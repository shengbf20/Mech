#!/usr/bin/env python3
"""
Stage-2 checkpoint matching on the calibration set.

Protocol (config.matching / README §1.1):
  1. Score each depth_* checkpoint (1 sample/prompt by default).
  2. Prefer checkpoints with |S - S0| < abs_diff_threshold (0.005).
  3. When |S - S0| < refine_when_abs_diff_below (0.02), re-score with
     samples_per_prompt_refine (5) and re-rank.
  4. Write match_report.json; mark unreliable if nothing meets the threshold.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import load_config, write_json

DEPTH_RE = re.compile(r"^depth_(\d+)$")


def discover_depths(ckpt_dir: Path) -> list[int]:
    depths: list[int] = []
    for p in ckpt_dir.iterdir():
        if not p.is_dir():
            continue
        m = DEPTH_RE.match(p.name)
        if m and (p / "config.json").exists():
            depths.append(int(m.group(1)))
    return sorted(depths)


def load_s0(path: Path) -> float:
    obj = load_json(path)
    if "positive_rate_S" not in obj:
        raise SystemExit(f"missing positive_rate_S in {path}")
    return float(obj["positive_rate_S"])


def load_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def eval_one(
    *,
    root: Path,
    config: Path,
    model_dir: Path,
    prompts: Path,
    out_stem: Path,
    samples: int,
    gen_gpus: str,
    score_gpu: str,
    num_gpus: int,
    seed: int,
    limit: int | None,
) -> dict[str, Any]:
    """Generate + score one checkpoint; return summary dict."""
    gen_path = out_stem.with_suffix(".generations.jsonl")
    scored_path = out_stem.with_suffix(".scored.jsonl")
    summary_path = scored_path.with_suffix(".summary.json")  # .scored.summary.json

    # pathlib: Path('x.scored.jsonl').with_suffix('.summary.json') → x.scored.summary.json
    # score_eval uses out_path.with_suffix(".summary.json") on the scored jsonl path.

    if not summary_path.exists() or not scored_path.exists():
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = gen_gpus
        gen_cmd = [
            sys.executable,
            str(root / "generate_eval.py"),
            "--config",
            str(config),
            "--model",
            str(model_dir),
            "--prompts",
            str(prompts),
            "--output",
            str(gen_path),
            "--samples-per-prompt",
            str(samples),
            "--seed",
            str(seed),
            "--num-gpus",
            str(num_gpus),
        ]
        if limit is not None:
            gen_cmd.extend(["--limit", str(limit)])
        run_cmd(gen_cmd, env=env)

        env2 = os.environ.copy()
        env2["HF_HUB_OFFLINE"] = "1"
        env2["TRANSFORMERS_OFFLINE"] = "1"
        env2["CUDA_VISIBLE_DEVICES"] = score_gpu
        run_cmd(
            [
                sys.executable,
                str(root / "score_eval.py"),
                "--config",
                str(config),
                "--generations",
                str(gen_path),
                "--output",
                str(scored_path),
            ],
            env=env2,
        )

    summary = load_json(summary_path)
    return summary


def select_match(
    rows: list[dict[str, Any]],
    s0: float,
    abs_thr: float,
) -> dict[str, Any]:
    """Pick checkpoint minimizing |S-S0|; flag whether it meets threshold."""
    if not rows:
        raise SystemExit("no scored checkpoints to match")
    ranked = sorted(rows, key=lambda r: (abs(float(r["S"]) - s0), int(r["step"])))
    best = ranked[0]
    diff = abs(float(best["S"]) - s0)
    return {
        "selected_step": int(best["step"]),
        "selected_S": float(best["S"]),
        "S0": s0,
        "abs_diff": diff,
        "matched": diff < abs_thr,
        "ranking": [
            {
                "step": int(r["step"]),
                "S": float(r["S"]),
                "abs_diff": abs(float(r["S"]) - s0),
                "samples_per_prompt": int(r.get("samples_per_prompt", 1)),
                "phase": r.get("phase"),
            }
            for r in ranked
        ],
    }


def find_cross_window(
    coarse: list[dict[str, Any]], s0: float, coarse_every: int
) -> dict[str, int]:
    """
    Locate (lo, hi] step window for stepwise refinement.
    Prefer first bracket where S crosses S0; else bracket around closest coarse point.
    """
    pts = sorted(coarse, key=lambda r: int(r["step"]))
    if not pts:
        raise SystemExit("empty coarse list")

    # Crossing: consecutive coarse points with (S-s0) sign change, or equal.
    for a, b in zip(pts, pts[1:]):
        da = float(a["S"]) - s0
        db = float(b["S"]) - s0
        if da == 0 or db == 0 or da * db < 0:
            return {
                "window_lo": int(a["step"]),
                "window_hi": int(b["step"]),
                "reason": "sign_cross_or_hit",
            }

    # No cross: window around closest coarse point.
    closest = min(pts, key=lambda r: abs(float(r["S"]) - s0))
    step = int(closest["step"])
    lo = max(0, step - coarse_every)
    # If closest is first point, window is (0, first]; else (step-every, step]
    # Prefer the adjacent interval that contains the closest approach.
    if step == int(pts[0]["step"]) and len(pts) >= 2:
        return {
            "window_lo": 0,
            "window_hi": step,
            "reason": "closest_first_bracket",
        }
    prev = step - coarse_every
    if prev < 0:
        prev = 0
    return {
        "window_lo": prev,
        "window_hi": step,
        "reason": "closest_bracket",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        required=True,
        help="Directory containing depth_<step>/ HF checkpoints",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Where to write generations/scores (default: <ckpt-dir>/match_eval)",
    )
    parser.add_argument(
        "--s0-summary",
        type=Path,
        default=None,
        help="S0 summary JSON (default: outputs/warmup/S0_calib_scored.summary.json)",
    )
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Optional comma-separated steps to eval (default: all depth_* present)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="auto",
        help="Label stored in report (coarse|medium|fine|auto)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override samples/prompt (default: matching.samples_per_prompt_default)",
    )
    parser.add_argument(
        "--skip-refine",
        action="store_true",
        help="Do not auto-run 5-sample refine near the match band",
    )
    parser.add_argument("--gen-gpus", type=str, default="0,1,2,3")
    parser.add_argument("--score-gpu", type=str, default="0")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap prompts")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output report path (default: <eval-dir>/match_report.json)",
    )
    args = parser.parse_args()

    root = args.config.resolve().parent
    cfg = load_config(args.config)
    mcfg = cfg["matching"]
    seed = int(args.seed if args.seed is not None else cfg["train"]["seeds"][0])
    ckpt_dir = args.ckpt_dir if args.ckpt_dir.is_absolute() else root / args.ckpt_dir
    eval_dir = (
        args.eval_dir
        if args.eval_dir is not None
        else ckpt_dir / "match_eval"
    )
    eval_dir = eval_dir if eval_dir.is_absolute() else root / eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)

    s0_path = args.s0_summary or (
        root / cfg["paths"]["output_dir"] / "warmup" / "S0_calib_scored.summary.json"
    )
    s0_path = s0_path if s0_path.is_absolute() else root / s0_path
    s0 = load_s0(s0_path)

    prompts = args.prompts or (root / cfg["paths"]["splits_dir"] / "calib_2000.jsonl")
    prompts = prompts if prompts.is_absolute() else root / prompts

    if args.steps:
        steps = sorted({int(x.strip()) for x in args.steps.split(",") if x.strip()})
    else:
        steps = discover_depths(ckpt_dir)
    if not steps:
        raise SystemExit(f"no depth_* checkpoints under {ckpt_dir}")

    samples_default = int(
        args.samples
        if args.samples is not None
        else mcfg["samples_per_prompt_default"]
    )
    samples_refine = int(mcfg["samples_per_prompt_refine"])
    abs_thr = float(mcfg["abs_diff_threshold"])
    refine_band = float(mcfg["refine_when_abs_diff_below"])
    coarse_every = int(mcfg["coarse_eval_every_steps"])

    rows: list[dict[str, Any]] = []
    for step in steps:
        model_dir = ckpt_dir / f"depth_{step}"
        if not (model_dir / "config.json").exists():
            raise SystemExit(f"missing checkpoint {model_dir}")
        stem = eval_dir / f"step_{step}_n{samples_default}"
        summary = eval_one(
            root=root,
            config=args.config if args.config.is_absolute() else root / args.config,
            model_dir=model_dir,
            prompts=prompts,
            out_stem=stem,
            samples=samples_default,
            gen_gpus=args.gen_gpus,
            score_gpu=args.score_gpu,
            num_gpus=args.num_gpus,
            seed=seed,
            limit=args.limit,
        )
        s = float(summary["positive_rate_S"])
        rows.append(
            {
                "step": step,
                "S": s,
                "abs_diff": abs(s - s0),
                "samples_per_prompt": samples_default,
                "phase": args.phase,
                "summary": str(stem.with_suffix(".scored.summary.json")),
            }
        )
        print(
            f"[match] step={step} S={s:.4f} |S-S0|={abs(s - s0):.4f} "
            f"(S0={s0:.4f}, n={samples_default})",
            flush=True,
        )

    # Optional refine near band
    refine_rows: list[dict[str, Any]] = []
    if not args.skip_refine:
        near = [r for r in rows if float(r["abs_diff"]) < refine_band]
        # Always refine the current best if inside band or if any candidate is close
        if near:
            for r in sorted(near, key=lambda x: float(x["abs_diff"]))[:5]:
                step = int(r["step"])
                stem = eval_dir / f"step_{step}_n{samples_refine}"
                summary = eval_one(
                    root=root,
                    config=args.config if args.config.is_absolute() else root / args.config,
                    model_dir=ckpt_dir / f"depth_{step}",
                    prompts=prompts,
                    out_stem=stem,
                    samples=samples_refine,
                    gen_gpus=args.gen_gpus,
                    score_gpu=args.score_gpu,
                    num_gpus=args.num_gpus,
                    seed=seed,
                    limit=args.limit,
                )
                s = float(summary["positive_rate_S"])
                refine_rows.append(
                    {
                        "step": step,
                        "S": s,
                        "abs_diff": abs(s - s0),
                        "samples_per_prompt": samples_refine,
                        "phase": f"{args.phase}_refine",
                        "summary": str(stem.with_suffix(".scored.summary.json")),
                    }
                )
                print(
                    f"[match-refine] step={step} S={s:.4f} |S-S0|={abs(s - s0):.4f} "
                    f"(n={samples_refine})",
                    flush=True,
                )

    pool = refine_rows if refine_rows else rows
    selection = select_match(pool, s0, abs_thr)
    window = find_cross_window(rows, s0, coarse_every)

    report = {
        "ckpt_dir": str(ckpt_dir),
        "eval_dir": str(eval_dir),
        "S0": s0,
        "s0_summary": str(s0_path),
        "abs_diff_threshold": abs_thr,
        "refine_band": refine_band,
        "coarse_points": rows,
        "refine_points": refine_rows,
        "selection": selection,
        "suggested_window": window,
        "reliable": bool(selection["matched"]),
        "exclude_if_unreliable": bool(mcfg.get("exclude_if_unreliable", True)),
    }
    report_path = args.report or (eval_dir / "match_report.json")
    report_path = report_path if report_path.is_absolute() else root / report_path
    write_json(report_path, report)

    print("=== match report ===", flush=True)
    print(
        f"selected step={selection['selected_step']} S={selection['selected_S']:.4f} "
        f"|S-S0|={selection['abs_diff']:.4f} matched={selection['matched']}",
        flush=True,
    )
    print(f"suggested_window={window}", flush=True)
    print(f"wrote {report_path}", flush=True)
    if not selection["matched"] and mcfg.get("exclude_if_unreliable", True):
        print(
            "WARNING: no checkpoint met |S-S0|<threshold; "
            "branch should be excluded from main comparison if still unmatched "
            "after fine scan.",
            flush=True,
        )


if __name__ == "__main__":
    main()
