#!/usr/bin/env python3
"""Generate continuations from fixed IMDb prefixes (calib / final_test).

Multi-GPU: shard prompts by global index (i % num_gpus == rank). Per-example
RNG uses the global index so seeds match single-GPU runs for the same prompt.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from common import load_config, load_jsonl, write_json, write_jsonl


def _resolve_paths(
    args: argparse.Namespace,
) -> tuple[int, Path, Path, Path, int, dict]:
    root = args.config.resolve().parent
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg["train"]["seeds"][0])
    prompts_path = args.prompts or (root / cfg["paths"]["splits_dir"] / "calib_2000.jsonl")
    prompts_path = prompts_path if prompts_path.is_absolute() else root / prompts_path
    out_path = args.output if args.output.is_absolute() else root / args.output
    model_path = args.model if args.model.is_absolute() else root / args.model
    n_samples = int(
        args.samples_per_prompt
        if args.samples_per_prompt is not None
        else cfg["generation"]["samples_per_prompt_default"]
    )
    return seed, prompts_path, out_path, model_path, n_samples, cfg["generation"]


def _load_rows(prompts_path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = load_jsonl(prompts_path)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _generate_on_device(
    *,
    model_path: Path,
    device: torch.device,
    rows: list[dict[str, Any]],
    global_indices: list[int],
    seed: int,
    n_samples: int,
    g: dict[str, Any],
    progress_tag: str = "",
) -> list[dict[str, Any]]:
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    generations: list[dict[str, Any]] = []
    with torch.inference_mode():
        for local_k, (i, row) in enumerate(zip(global_indices, rows)):
            prefix = row["prefix_text"]
            inputs = tok(prefix, return_tensors="pt").to(device)
            for s in range(n_samples):
                # Deterministic-ish per (global example index, sample index)
                torch.manual_seed(seed + i * 1009 + s)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed + i * 1009 + s)
                out = model.generate(
                    **inputs,
                    do_sample=bool(g["do_sample"]),
                    temperature=float(g["temperature"]),
                    top_p=float(g["top_p"]),
                    top_k=int(g["top_k"]) if int(g["top_k"]) > 0 else None,
                    max_new_tokens=int(g["max_new_tokens"]),
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,
                )
                cont = tok.decode(
                    out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                generations.append(
                    {
                        "example_id": row["example_id"],
                        "source_index": row.get("source_index"),
                        "label_imdb": row.get("label"),
                        "prefix_len": row.get("prefix_len"),
                        "prefix_text": prefix,
                        "sample_idx": s,
                        "continuation": cont,
                        "full_text": prefix + cont,
                        "model": str(model_path),
                        "_global_idx": i,
                    }
                )
            if (local_k + 1) % 50 == 0:
                print(
                    f"{progress_tag}generated {local_k + 1}/{len(rows)} "
                    f"(last global_idx={i})",
                    flush=True,
                )
    return generations


def _visible_device_ids(num_gpus: int) -> list[str]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() != "":
        ids = [x.strip() for x in cvd.split(",") if x.strip() != ""]
    else:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA required for --num-gpus > 1")
        ids = [str(i) for i in range(torch.cuda.device_count())]
    if len(ids) < num_gpus:
        raise SystemExit(
            f"Requested --num-gpus={num_gpus} but only {len(ids)} visible "
            f"(CUDA_VISIBLE_DEVICES={cvd!r})"
        )
    return ids[:num_gpus]


def _run_multi_gpu(
    args: argparse.Namespace,
    num_gpus: int,
    rows: list[dict[str, Any]],
    n_samples: int,
    out_path: Path,
) -> list[dict[str, Any]]:
    """Launch one subprocess per GPU (CVD set before Python/CUDA init), then merge."""
    visible = _visible_device_ids(num_gpus)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shard_paths = [
        out_path.parent / f".{out_path.stem}.shard{r}.jsonl" for r in range(num_gpus)
    ]

    procs: list[subprocess.Popen[str]] = []
    for r in range(num_gpus):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(args.config),
            "--model",
            str(args.model),
            "--output",
            str(shard_paths[r]),
            "--num-gpus",
            "1",
            "--shard-rank",
            str(r),
            "--shard-world",
            str(num_gpus),
        ]
        if args.prompts is not None:
            cmd.extend(["--prompts", str(args.prompts)])
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.samples_per_prompt is not None:
            cmd.extend(["--samples-per-prompt", str(args.samples_per_prompt)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible[r]
        print(f"[launcher] rank {r} → GPU {visible[r]}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))

    failed = False
    for r, p in enumerate(procs):
        code = p.wait()
        if code != 0:
            print(f"[launcher] rank {r} failed with exit {code}", flush=True)
            failed = True
    if failed:
        raise SystemExit("one or more generate shards failed")

    merged: list[dict[str, Any]] = []
    for r, sp in enumerate(shard_paths):
        if not sp.exists():
            raise SystemExit(f"missing shard output: {sp}")
        shard = load_jsonl(sp)
        expected_idx = [i for i in range(len(rows)) if i % num_gpus == r]
        # One record per (prompt, sample); validate prompt index sequence.
        got_prompt_idx = [int(x["_global_idx"]) for x in shard if int(x["sample_idx"]) == 0]
        if got_prompt_idx != expected_idx:
            raise SystemExit(
                f"shard {r} index mismatch: expected {expected_idx[:8]}... "
                f"got {got_prompt_idx[:8]}... "
                f"(n={len(got_prompt_idx)} vs {len(expected_idx)})"
            )
        if len(shard) != len(expected_idx) * n_samples:
            raise SystemExit(
                f"shard {r} size {len(shard)} != "
                f"{len(expected_idx)} prompts × {n_samples} samples"
            )
        merged.extend(shard)
        sp.unlink(missing_ok=True)

    merged.sort(key=lambda r: (int(r["_global_idx"]), int(r["sample_idx"])))
    for r in merged:
        r.pop("_global_idx", None)
    if len(merged) != len(rows) * n_samples:
        raise SystemExit(
            f"merged size {len(merged)} != {len(rows)} prompts × {n_samples} samples"
        )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--model", type=Path, required=True, help="Local checkpoint dir")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Prompt jsonl (default: data/splits/calib_2000.jsonl)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output generations jsonl")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for debugging")
    parser.add_argument("--samples-per-prompt", type=int, default=None)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Shard prompts across this many visible GPUs (default: 1).",
    )
    parser.add_argument(
        "--shard-rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # internal: set by multi-GPU launcher
    )
    parser.add_argument(
        "--shard-world",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    seed, prompts_path, out_path, model_path, n_samples, g = _resolve_paths(args)
    set_seed(seed)
    rows = _load_rows(prompts_path, args.limit)
    num_gpus = int(args.num_gpus)

    if num_gpus < 1:
        raise SystemExit("--num-gpus must be >= 1")

    # Multi-GPU parent: spawn shard subprocesses and merge.
    if num_gpus > 1 and args.shard_rank is None:
        if not torch.cuda.is_available():
            raise SystemExit("--num-gpus > 1 requires CUDA")
        generations = _run_multi_gpu(args, num_gpus, rows, n_samples, out_path)
        write_jsonl(out_path, generations)
        write_json(
            out_path.with_suffix(".meta.json"),
            {
                "model": str(model_path),
                "prompts": str(prompts_path),
                "n_prompts": len(rows),
                "samples_per_prompt": n_samples,
                "n_generations": len(generations),
                "seed": seed,
                "num_gpus": num_gpus,
            },
        )
        print(f"Wrote {len(generations)} generations → {out_path} (num_gpus={num_gpus})")
        return

    # Single process (full run or one shard).
    shard_rank = args.shard_rank
    shard_world = args.shard_world
    if shard_rank is not None:
        if shard_world is None or shard_world < 1:
            raise SystemExit("--shard-world required with --shard-rank")
        global_indices = [i for i in range(len(rows)) if i % shard_world == shard_rank]
        shard_rows = [rows[i] for i in global_indices]
        tag = f"[shard {shard_rank}/{shard_world}] "
    else:
        global_indices = list(range(len(rows)))
        shard_rows = rows
        tag = ""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generations = _generate_on_device(
        model_path=model_path,
        device=device,
        rows=shard_rows,
        global_indices=global_indices,
        seed=seed,
        n_samples=n_samples,
        g=g,
        progress_tag=tag,
    )

    # Keep _global_idx for shard merge validation; strip for normal single-GPU writes.
    if shard_rank is None:
        for r in generations:
            r.pop("_global_idx", None)

    write_jsonl(out_path, generations)
    if shard_rank is None:
        write_json(
            out_path.with_suffix(".meta.json"),
            {
                "model": str(model_path),
                "prompts": str(prompts_path),
                "n_prompts": len(rows),
                "samples_per_prompt": n_samples,
                "n_generations": len(generations),
                "seed": seed,
                "num_gpus": 1,
            },
        )
    print(
        f"{tag}Wrote {len(generations)} generations → {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
