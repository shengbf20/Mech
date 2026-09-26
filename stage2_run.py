#!/usr/bin/env python3
"""
Stage 2 orchestrator: reverse SFT on negatives + hierarchical matching to S0.

Phases:
  coarse  — train from Stage-1 depth, save every 128, score, propose window
  medium  — retrain same trajectory, save every --medium-every inside window
  fine    — densify ±--fine-radius around best medium step (every 1 step)
  match   — score existing depth_* under --output (no training)
  all     — coarse → medium → fine (stops early if matched)

Disk note: fine saves (~2B HF) only for a small radius (default 8), not the full 128 window.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import load_config, write_json


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        raise SystemExit(f"no save depths in (0, {max_steps}] from {save_depths}")
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


def run_match(
    *,
    root: Path,
    config: Path,
    ckpt_dir: Path,
    eval_dir: Path,
    steps: list[int] | None,
    phase: str,
    seed: int,
    gen_gpus: str,
    score_gpu: str,
    num_gpus: int,
    skip_refine: bool,
    report: Path,
    limit: int | None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(root / "match_checkpoint.py"),
        "--config",
        str(config),
        "--ckpt-dir",
        str(ckpt_dir),
        "--eval-dir",
        str(eval_dir),
        "--phase",
        phase,
        "--seed",
        str(seed),
        "--gen-gpus",
        gen_gpus,
        "--score-gpu",
        score_gpu,
        "--num-gpus",
        str(num_gpus),
        "--report",
        str(report),
    ]
    if steps is not None:
        cmd.extend(["--steps", ",".join(str(s) for s in steps)])
    if skip_refine:
        cmd.append("--skip-refine")
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    run(cmd)
    return load_json(report)


def medium_depths(window_lo: int, window_hi: int, every: int) -> list[int]:
    if every <= 0:
        raise ValueError("medium every must be > 0")
    if window_hi <= window_lo:
        raise ValueError(f"bad window ({window_lo}, {window_hi}]")
    steps = list(range(window_lo + every, window_hi + 1, every))
    if not steps or steps[-1] != window_hi:
        steps.append(window_hi)
    # Also keep window_hi's left edge checkpoint if present from coarse
    return sorted(set(steps))


def fine_depths(center: int, radius: int, lo: int, hi: int) -> list[int]:
    a = max(lo + 1, center - radius)
    b = min(hi, center + radius)
    return list(range(a, b + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--phase",
        choices=["coarse", "medium", "fine", "match", "all"],
        default="all",
    )
    parser.add_argument(
        "--stage1",
        type=Path,
        default=None,
        help="Stage-1 HF ckpt (default: outputs/stage1/main_pos_seed11/depth_512)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Negative jsonl (default: data/splits/main_negative_8192.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Stage-2 output dir (default: outputs/stage2/main_neg_seed11)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Coarse/full train length (default: 512 = one pass over 8192 @ gbs 16)",
    )
    parser.add_argument("--medium-every", type=int, default=16)
    parser.add_argument("--fine-radius", type=int, default=8)
    parser.add_argument("--train-gpus", type=str, default="0,1,2,3")
    parser.add_argument("--gen-gpus", type=str, default="0,1,2,3")
    parser.add_argument("--score-gpu", type=str, default="0")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Debug prompt cap for match")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even if depth_* already exist for requested steps",
    )
    args = parser.parse_args()

    root = args.config.resolve().parent
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg["train"]["seeds"][0])
    locked = int(cfg["imdb_sft"].get("stage1_depth_locked", 512))
    stage1 = args.stage1 or (
        root
        / cfg["paths"]["output_dir"]
        / "stage1"
        / f"main_pos_seed{seed}"
        / f"depth_{locked}"
    )
    stage1 = stage1 if stage1.is_absolute() else root / stage1
    data = args.data or (root / cfg["paths"]["splits_dir"] / "main_negative_8192.jsonl")
    data = data if data.is_absolute() else root / data
    out_dir = args.output or (
        root / cfg["paths"]["output_dir"] / "stage2" / f"main_neg_seed{seed}"
    )
    out_dir = out_dir if out_dir.is_absolute() else root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    coarse_every = int(cfg["matching"]["coarse_eval_every_steps"])
    max_steps = int(
        args.max_steps
        if args.max_steps is not None
        else cfg["imdb_sft"]["n_per_stage_main"] // cfg["train"]["global_batch_size"]
    )
    config_path = args.config if args.config.is_absolute() else root / args.config

    state_path = out_dir / "stage2_state.json"
    state: dict[str, Any] = load_json(state_path) if state_path.exists() else {}

    def save_state() -> None:
        write_json(state_path, state)

    def need_train(steps: list[int]) -> bool:
        if args.force_retrain:
            return True
        return any(not (out_dir / f"depth_{s}" / "config.json").exists() for s in steps)

    # ---- match-only ----
    if args.phase == "match":
        report = run_match(
            root=root,
            config=config_path,
            ckpt_dir=out_dir,
            eval_dir=out_dir / "match_eval",
            steps=None,
            phase="existing",
            seed=seed,
            gen_gpus=args.gen_gpus,
            score_gpu=args.score_gpu,
            num_gpus=args.num_gpus,
            skip_refine=False,
            report=out_dir / "match_eval" / "match_report.json",
            limit=args.limit,
        )
        state["last_match"] = report["selection"]
        save_state()
        return

    # ---- coarse ----
    if args.phase in ("coarse", "all"):
        coarse_steps = list(range(coarse_every, max_steps + 1, coarse_every))
        if need_train(coarse_steps):
            train_imdb(
                root=root,
                config=config_path,
                model=stage1,
                data=data,
                output=out_dir,
                max_steps=max_steps,
                save_depths=coarse_steps,
                seed=seed,
                train_gpus=args.train_gpus,
                num_gpus=args.num_gpus,
            )
        else:
            print(f"[stage2] reuse existing coarse depths {coarse_steps}", flush=True)

        coarse_report = run_match(
            root=root,
            config=config_path,
            ckpt_dir=out_dir,
            eval_dir=out_dir / "match_eval" / "coarse",
            steps=coarse_steps,
            phase="coarse",
            seed=seed,
            gen_gpus=args.gen_gpus,
            score_gpu=args.score_gpu,
            num_gpus=args.num_gpus,
            skip_refine=True,  # refine after fine densify
            report=out_dir / "match_eval" / "coarse_report.json",
            limit=args.limit,
        )
        state["coarse"] = {
            "steps": coarse_steps,
            "selection": coarse_report["selection"],
            "suggested_window": coarse_report["suggested_window"],
            "points": coarse_report["coarse_points"],
        }
        save_state()

        if coarse_report["selection"]["matched"]:
            # Still run refine on that single point
            refine = run_match(
                root=root,
                config=config_path,
                ckpt_dir=out_dir,
                eval_dir=out_dir / "match_eval" / "coarse_refine",
                steps=[int(coarse_report["selection"]["selected_step"])],
                phase="coarse",
                seed=seed,
                gen_gpus=args.gen_gpus,
                score_gpu=args.score_gpu,
                num_gpus=args.num_gpus,
                skip_refine=False,
                report=out_dir / "match_eval" / "coarse_refine_report.json",
                limit=args.limit,
            )
            state["matched"] = refine["selection"]
            state["matched_phase"] = "coarse"
            save_state()
            write_json(out_dir / "MATCHED.json", state["matched"])
            print("[stage2] matched at coarse granularity", flush=True)
            if args.phase == "coarse":
                return
            if refine["selection"]["matched"]:
                print("[stage2] done (coarse match reliable)", flush=True)
                return

    if args.phase == "coarse":
        return

    # Need window from coarse state
    if "coarse" not in state:
        raise SystemExit("missing coarse state; run --phase coarse|all first")
    window = state["coarse"]["suggested_window"]
    w_lo, w_hi = int(window["window_lo"]), int(window["window_hi"])
    print(f"[stage2] window=({w_lo}, {w_hi}] reason={window.get('reason')}", flush=True)

    # ---- medium ----
    if args.phase in ("medium", "all"):
        med_steps = medium_depths(w_lo, w_hi, int(args.medium_every))
        # Must retrain from stage1 through w_hi to keep one trajectory; only those steps saved.
        if need_train(med_steps):
            train_imdb(
                root=root,
                config=config_path,
                model=stage1,
                data=data,
                output=out_dir,
                max_steps=w_hi if w_hi > 0 else max_steps,
                save_depths=med_steps,
                seed=seed,
                train_gpus=args.train_gpus,
                num_gpus=args.num_gpus,
            )
        else:
            print(f"[stage2] reuse existing medium depths {med_steps}", flush=True)

        med_report = run_match(
            root=root,
            config=config_path,
            ckpt_dir=out_dir,
            eval_dir=out_dir / "match_eval" / "medium",
            steps=med_steps,
            phase="medium",
            seed=seed,
            gen_gpus=args.gen_gpus,
            score_gpu=args.score_gpu,
            num_gpus=args.num_gpus,
            skip_refine=True,
            report=out_dir / "match_eval" / "medium_report.json",
            limit=args.limit,
        )
        state["medium"] = {
            "steps": med_steps,
            "selection": med_report["selection"],
            "points": med_report["coarse_points"],
        }
        save_state()
        if med_report["selection"]["matched"] and args.phase == "medium":
            refine = run_match(
                root=root,
                config=config_path,
                ckpt_dir=out_dir,
                eval_dir=out_dir / "match_eval" / "medium_refine",
                steps=[int(med_report["selection"]["selected_step"])],
                phase="medium",
                seed=seed,
                gen_gpus=args.gen_gpus,
                score_gpu=args.score_gpu,
                num_gpus=args.num_gpus,
                skip_refine=False,
                report=out_dir / "match_eval" / "medium_refine_report.json",
                limit=args.limit,
            )
            state["matched"] = refine["selection"]
            state["matched_phase"] = "medium"
            save_state()
            write_json(out_dir / "MATCHED.json", state["matched"])
            return

    if args.phase == "medium":
        return

    # ---- fine ----
    if args.phase in ("fine", "all"):
        if "medium" in state:
            center = int(state["medium"]["selection"]["selected_step"])
        else:
            center = int(state["coarse"]["selection"]["selected_step"])
        fine_steps = fine_depths(center, int(args.fine_radius), w_lo, w_hi)
        print(f"[stage2] fine steps around {center}: {fine_steps}", flush=True)
        if need_train(fine_steps):
            train_imdb(
                root=root,
                config=config_path,
                model=stage1,
                data=data,
                output=out_dir,
                max_steps=max(fine_steps),
                save_depths=fine_steps,
                seed=seed,
                train_gpus=args.train_gpus,
                num_gpus=args.num_gpus,
            )
        else:
            print(f"[stage2] reuse existing fine depths {fine_steps}", flush=True)

        fine_report = run_match(
            root=root,
            config=config_path,
            ckpt_dir=out_dir,
            eval_dir=out_dir / "match_eval" / "fine",
            steps=fine_steps,
            phase="fine",
            seed=seed,
            gen_gpus=args.gen_gpus,
            score_gpu=args.score_gpu,
            num_gpus=args.num_gpus,
            skip_refine=False,
            report=out_dir / "match_eval" / "fine_report.json",
            limit=args.limit,
        )
        state["fine"] = {
            "steps": fine_steps,
            "selection": fine_report["selection"],
            "points": fine_report["coarse_points"],
            "refine_points": fine_report["refine_points"],
            "reliable": fine_report["reliable"],
        }
        state["matched"] = fine_report["selection"]
        state["matched_phase"] = "fine"
        save_state()
        write_json(out_dir / "MATCHED.json", fine_report["selection"])
        if fine_report["reliable"]:
            print(
                f"[stage2] MATCH OK → depth_{fine_report['selection']['selected_step']} "
                f"S={fine_report['selection']['selected_S']:.4f}",
                flush=True,
            )
        else:
            print(
                "[stage2] NO reliable match (|S-S0|>=threshold after refine). "
                "Exclude this branch from main Stage-3 comparison.",
                flush=True,
            )


if __name__ == "__main__":
    main()
