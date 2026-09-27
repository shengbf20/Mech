#!/usr/bin/env python3
"""
Stage 3 orchestrator: re-expose to positive IMDb and compare curves.

Arms (same data / seed / schedule):
  primed   — start from Stage-2 matched checkpoint (rehearsal)
  control  — start from warmup W0 (first-time positive learning)

Eval schedule (config.stage3_eval):
  every 8 steps for t ∈ (0, 128], then every 32 steps thereafter.
Also scores step 0 (the starting checkpoint) for each arm.

Trial default: --max-steps = research.trial_metric_window_steps (128).

Primary summary metric: baseline-adjusted curve area of
  ΔS(t) = S_primed(t) − S_control(t)  over [0, window]
via trapezoidal rule (secondary: steps for each arm to reach a target S).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import load_config, write_json
from match_checkpoint import eval_one


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stage3_eval_steps(max_steps: int, every_until: int, every_after: int) -> list[int]:
    if max_steps <= 0:
        return []
    steps: list[int] = []
    t = every_until
    while t <= min(128, max_steps):
        steps.append(t)
        t += every_until
    t = 128 + every_after
    while t <= max_steps:
        steps.append(t)
        t += every_after
    if max_steps not in steps:
        steps.append(max_steps)
    return sorted(set(s for s in steps if 0 < s <= max_steps))


def resolve_matched_ckpt(root: Path, cfg: dict[str, Any], seed: int, override: Path | None) -> Path:
    if override is not None:
        p = override if override.is_absolute() else root / override
        if not (p / "config.json").exists():
            raise SystemExit(f"missing Stage-2 matched ckpt: {p}")
        return p
    stage2_dir = root / cfg["paths"]["output_dir"] / "stage2" / f"main_neg_seed{seed}"
    matched_path = stage2_dir / "MATCHED.json"
    if not matched_path.exists():
        raise SystemExit(f"missing {matched_path}; pass --primed explicitly")
    matched = load_json(matched_path)
    if not matched.get("matched", False):
        raise SystemExit(
            f"{matched_path} has matched=false; exclude this branch or pass --primed manually"
        )
    step = int(matched["selected_step"])
    p = stage2_dir / f"depth_{step}"
    if not (p / "config.json").exists():
        raise SystemExit(f"MATCHED step={step} but missing {p}")
    return p


def train_imdb(
    *,
    root: Path,
    config: Path,
    model: Path,
    data: Path,
    output: Path,
    max_steps: int,
    save_depths: list[int],
    seed: int,
    train_gpus: str,
    num_gpus: int,
) -> None:
    depths = sorted({d for d in save_depths if 0 < d <= max_steps})
    if not depths:
        raise SystemExit(f"no save depths in (0, {max_steps}]")
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = train_gpus
    cmd = [
        "deepspeed",
        f"--num_gpus={num_gpus}",
        str(root / "train_sft.py"),
        "--mode",
        "imdb",
        "--config",
        str(config),
        "--model",
        str(model),
        "--data",
        str(data),
        "--output",
        str(output),
        "--max-steps",
        str(max_steps),
        "--save-depths",
        ",".join(str(d) for d in depths),
        "--seed",
        str(seed),
    ]
    run(cmd, env=env)


def need_train(out_dir: Path, steps: list[int], force: bool) -> bool:
    if force:
        return True
    return any(not (out_dir / f"depth_{s}" / "config.json").exists() for s in steps)


def score_arm(
    *,
    root: Path,
    config: Path,
    arm_name: str,
    start_model: Path,
    ckpt_dir: Path,
    eval_dir: Path,
    steps: list[int],
    prompts: Path,
    seed: int,
    gen_gpus: str,
    score_gpu: str,
    num_gpus: int,
    samples: int,
    limit: int | None,
    delete_ckpts: bool,
) -> list[dict[str, Any]]:
    """Score step 0 (start_model) + each depth_* in steps."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    # step 0
    stem0 = eval_dir / f"step_0_n{samples}"
    print(f"[stage3:{arm_name}] eval step=0 ← {start_model}", flush=True)
    summary0 = eval_one(
        root=root,
        config=config,
        model_dir=start_model,
        prompts=prompts,
        out_stem=stem0,
        samples=samples,
        gen_gpus=gen_gpus,
        score_gpu=score_gpu,
        num_gpus=num_gpus,
        seed=seed,
        limit=limit,
    )
    rows.append(
        {
            "arm": arm_name,
            "step": 0,
            "S": float(summary0["positive_rate_S"]),
            "samples_per_prompt": samples,
            "model": str(start_model),
            "summary": str(stem0.with_suffix(".scored.summary.json")),
        }
    )

    for step in steps:
        model_dir = ckpt_dir / f"depth_{step}"
        if not (model_dir / "config.json").exists():
            raise SystemExit(f"missing {model_dir}")
        stem = eval_dir / f"step_{step}_n{samples}"
        print(f"[stage3:{arm_name}] eval step={step}", flush=True)
        summary = eval_one(
            root=root,
            config=config,
            model_dir=model_dir,
            prompts=prompts,
            out_stem=stem,
            samples=samples,
            gen_gpus=gen_gpus,
            score_gpu=score_gpu,
            num_gpus=num_gpus,
            seed=seed,
            limit=limit,
        )
        rows.append(
            {
                "arm": arm_name,
                "step": step,
                "S": float(summary["positive_rate_S"]),
                "samples_per_prompt": samples,
                "model": str(model_dir),
                "summary": str(stem.with_suffix(".scored.summary.json")),
            }
        )
        if delete_ckpts and step != steps[-1]:
            # Keep final depth; drop intermediates to save disk after scoring.
            shutil.rmtree(model_dir, ignore_errors=True)
            print(f"[stage3:{arm_name}] deleted {model_dir} after eval", flush=True)

    write_json(eval_dir / "curve.json", {"arm": arm_name, "points": rows})
    return rows


def trapezoid_area(xs: list[int], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    area = 0.0
    for i in range(len(xs) - 1):
        area += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i])
    return float(area)


def steps_to_target(points: list[dict[str, Any]], target: float) -> int | None:
    """First step where S >= target (or None)."""
    for p in sorted(points, key=lambda r: int(r["step"])):
        if float(p["S"]) >= target:
            return int(p["step"])
    return None


def analyze(
    primed: list[dict[str, Any]],
    control: list[dict[str, Any]],
    window: int,
    s0: float,
    target: float | None,
) -> dict[str, Any]:
    def clip(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [p for p in sorted(points, key=lambda r: int(r["step"])) if int(p["step"]) <= window]

    p = clip(primed)
    c = clip(control)
    # Align on shared steps
    c_by = {int(r["step"]): float(r["S"]) for r in c}
    shared = [r for r in p if int(r["step"]) in c_by]
    xs = [int(r["step"]) for r in shared]
    yp = [float(r["S"]) for r in shared]
    yc = [c_by[x] for x in xs]
    delta = [a - b for a, b in zip(yp, yc)]
    # Baseline-adjusted: area of (S_arm(t) - S_arm(0)) and of delta
    p0 = yp[0] if yp else 0.0
    c0 = yc[0] if yc else 0.0
    area_delta = trapezoid_area(xs, delta)
    area_primed_adj = trapezoid_area(xs, [y - p0 for y in yp])
    area_control_adj = trapezoid_area(xs, [y - c0 for y in yc])

    report: dict[str, Any] = {
        "window_steps": window,
        "S0": s0,
        "shared_steps": xs,
        "S_primed": yp,
        "S_control": yc,
        "delta_primed_minus_control": delta,
        "primary_metric": "baseline_adjusted_curve_area",
        "area_delta_S": area_delta,
        "area_primed_minus_S0_arm": area_primed_adj,
        "area_control_minus_S0_arm": area_control_adj,
        "area_primed_minus_control_adj": area_primed_adj - area_control_adj,
        "S_primed_at_0": p0,
        "S_control_at_0": c0,
        "interpretation": (
            "Positive area_delta_S means primed stays above control on average "
            "over the window (faster / higher relearning under this protocol)."
        ),
    }
    if target is not None:
        report["target_score"] = target
        report["steps_to_target_primed"] = steps_to_target(p, target)
        report["steps_to_target_control"] = steps_to_target(c, target)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--phase",
        choices=["train", "eval", "analyze", "all"],
        default="all",
        help="train both arms; eval curves; analyze; or all",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--primed",
        type=Path,
        default=None,
        help="Primed start ckpt (default: Stage2 MATCHED depth_*)",
    )
    parser.add_argument(
        "--control",
        type=Path,
        default=None,
        help="Control start ckpt (default: outputs/warmup/W0)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Positive jsonl (default: data/splits/main_positive_8192.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Root output (default: outputs/stage3/seed{seed})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Train length (default: trial_metric_window_steps=128)",
    )
    parser.add_argument("--train-gpus", type=str, default="0,1,2,3")
    parser.add_argument("--gen-gpus", type=str, default="0,1,2,3")
    parser.add_argument("--score-gpu", type=str, default="0")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument(
        "--delete-ckpts-after-eval",
        action="store_true",
        help="After scoring, delete intermediate depth_* (keeps last step)",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default="primed,control",
        help="Comma list: primed and/or control",
    )
    args = parser.parse_args()

    root = args.config.resolve().parent
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg["train"]["seeds"][0])
    config_path = args.config if args.config.is_absolute() else root / args.config

    primed_start = resolve_matched_ckpt(root, cfg, seed, args.primed)
    control_start = args.control or (root / cfg["paths"]["output_dir"] / "warmup" / "W0")
    control_start = control_start if control_start.is_absolute() else root / control_start
    if not (control_start / "config.json").exists():
        raise SystemExit(f"missing control ckpt: {control_start}")

    data = args.data or (root / cfg["paths"]["splits_dir"] / "main_positive_8192.jsonl")
    data = data if data.is_absolute() else root / data

    out_root = args.output or (root / cfg["paths"]["output_dir"] / "stage3" / f"seed{seed}")
    out_root = out_root if out_root.is_absolute() else root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    max_steps = int(
        args.max_steps
        if args.max_steps is not None
        else cfg["research"]["trial_metric_window_steps"]
    )
    every_until = int(cfg["stage3_eval"]["every_steps_until_128"])
    every_after = int(cfg["stage3_eval"]["every_steps_after_128"])
    eval_steps = stage3_eval_steps(max_steps, every_until, every_after)
    samples = int(cfg["generation"]["samples_per_prompt_default"])
    prompts = root / cfg["paths"]["splits_dir"] / "calib_2000.jsonl"

    s0_path = root / cfg["paths"]["output_dir"] / "warmup" / "S0_calib_scored.summary.json"
    s0 = float(load_json(s0_path)["positive_rate_S"])
    target = cfg["research"].get("target_score")
    target_f = float(target) if target is not None else None

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    arm_starts = {"primed": primed_start, "control": control_start}

    meta = {
        "seed": seed,
        "max_steps": max_steps,
        "eval_steps": eval_steps,
        "data": str(data),
        "primed_start": str(primed_start),
        "control_start": str(control_start),
        "S0": s0,
        "arms": arms,
    }
    write_json(out_root / "stage3_meta.json", meta)
    print(json.dumps(meta, indent=2), flush=True)

    # ---- train ----
    if args.phase in ("train", "all"):
        for arm in arms:
            arm_dir = out_root / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            if need_train(arm_dir, eval_steps, args.force_retrain):
                print(f"[stage3] training arm={arm} from {arm_starts[arm]}", flush=True)
                train_imdb(
                    root=root,
                    config=config_path,
                    model=arm_starts[arm],
                    data=data,
                    output=arm_dir,
                    max_steps=max_steps,
                    save_depths=eval_steps,
                    seed=seed,
                    train_gpus=args.train_gpus,
                    num_gpus=args.num_gpus,
                )
            else:
                print(f"[stage3] reuse existing depths for arm={arm}", flush=True)

    # ---- eval ----
    curves: dict[str, list[dict[str, Any]]] = {}
    if args.phase in ("eval", "all"):
        for arm in arms:
            arm_dir = out_root / arm
            curves[arm] = score_arm(
                root=root,
                config=config_path,
                arm_name=arm,
                start_model=arm_starts[arm],
                ckpt_dir=arm_dir,
                eval_dir=out_root / "eval" / arm,
                steps=eval_steps,
                prompts=prompts,
                seed=seed,
                gen_gpus=args.gen_gpus,
                score_gpu=args.score_gpu,
                num_gpus=args.num_gpus,
                samples=samples,
                limit=args.limit,
                delete_ckpts=bool(args.delete_ckpts_after_eval),
            )

    # ---- analyze ----
    if args.phase in ("analyze", "all"):
        if not curves:
            # reload from disk
            for arm in arms:
                curve_path = out_root / "eval" / arm / "curve.json"
                if not curve_path.exists():
                    raise SystemExit(f"missing {curve_path}; run --phase eval first")
                curves[arm] = load_json(curve_path)["points"]
        if "primed" not in curves or "control" not in curves:
            raise SystemExit("analyze requires both primed and control curves")
        report = analyze(
            curves["primed"],
            curves["control"],
            window=int(cfg["research"]["trial_metric_window_steps"]),
            s0=s0,
            target=target_f,
        )
        write_json(out_root / "curve_report.json", report)
        print("=== Stage-3 curve report ===", flush=True)
        print(json.dumps(report, indent=2), flush=True)
        print(f"wrote {out_root / 'curve_report.json'}", flush=True)


if __name__ == "__main__":
    main()
