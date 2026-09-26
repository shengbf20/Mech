#!/usr/bin/env python3
"""Generate continuations from fixed IMDb prefixes (calib / final_test)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from common import load_config, load_jsonl, write_json, write_jsonl


@torch.inference_mode()
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
    args = parser.parse_args()

    root = args.config.resolve().parent
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg["train"]["seeds"][0])
    set_seed(seed)

    prompts_path = args.prompts or (root / cfg["paths"]["splits_dir"] / "calib_2000.jsonl")
    prompts_path = prompts_path if prompts_path.is_absolute() else root / prompts_path
    out_path = args.output if args.output.is_absolute() else root / args.output
    model_path = args.model if args.model.is_absolute() else root / args.model

    n_samples = int(
        args.samples_per_prompt
        if args.samples_per_prompt is not None
        else cfg["generation"]["samples_per_prompt_default"]
    )
    g = cfg["generation"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    rows = load_jsonl(prompts_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    generations = []
    for i, row in enumerate(rows):
        prefix = row["prefix_text"]
        inputs = tok(prefix, return_tensors="pt").to(device)
        for s in range(n_samples):
            # Deterministic-ish per (example, sample index)
            torch.manual_seed(seed + i * 1009 + s)
            if torch.cuda.is_available():
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
            cont = tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
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
                }
            )
        if (i + 1) % 50 == 0:
            print(f"generated {i + 1}/{len(rows)}")

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
        },
    )
    print(f"Wrote {len(generations)} generations → {out_path}")


if __name__ == "__main__":
    main()
