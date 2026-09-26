#!/usr/bin/env python3
"""
Full-parameter SFT via TRL + DeepSpeed ZeRO-2.

Modes:
  warmup  — Alpaca-cleaned, 1 epoch, completion-only loss → outputs/warmup/W0
  imdb    — language-modeling on review text (Stage 1/2/3); start from --model

Launch with DeepSpeed, e.g.:
  deepspeed --num_gpus=4 train_sft.py --mode warmup --config config.yaml

Stage 1 (positive) example — one run, consolidated HF weights at candidate depths:
  deepspeed --num_gpus=4 train_sft.py --mode imdb \\
    --config config.yaml \\
    --model outputs/warmup/W0 \\
    --data data/splits/main_positive_8192.jsonl \\
    --output outputs/stage1/main_pos_seed11 \\
    --max-steps 512 \\
    --save-depths 128,256,512 \\
    --seed 11
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from common import alpaca_prompt_completion, is_main_process, load_config, load_jsonl, write_json


class SaveDepthsCallback(TrainerCallback):
    """Save consolidated HF weights at Stage-1 candidate depths (usable by generate_eval)."""

    def __init__(self, depths: set[int], out_dir: Path) -> None:
        self.depths = depths
        self.out_dir = out_dir
        self.trainer: SFTTrainer | None = None
        self.saved: list[int] = []

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        step = int(state.global_step)
        if step not in self.depths or self.trainer is None:
            return
        dest = self.out_dir / f"depth_{step}"
        dest.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(str(dest))
        tok = self.trainer.processing_class
        if tok is not None:
            tok.save_pretrained(str(dest))
        if is_main_process():
            self.saved.append(step)
            print(f"[save-depths] consolidated weights → {dest}")


def build_dataset(cfg: dict[str, Any], root: Path, mode: str, data_path: Path) -> Dataset:
    rows = load_jsonl(data_path)
    if mode == "warmup":
        records = []
        for r in rows:
            prompt, completion = alpaca_prompt_completion(r)
            records.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "example_id": r["example_id"],
                }
            )
        return Dataset.from_list(records)
    if mode == "imdb":
        return Dataset.from_list(
            [{"text": r["text"], "example_id": r["example_id"]} for r in rows]
        )
    raise ValueError(f"Unknown mode: {mode}")


def parse_depths(raw: str | None) -> set[int]:
    if not raw:
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def make_sft_args(
    cfg: dict[str, Any],
    root: Path,
    out_dir: Path,
    mode: str,
    max_steps: int | None,
    seed: int,
) -> SFTConfig:
    t = cfg["train"]
    # Intermediate ZeRO trainer checkpoints are optional; Stage-1 depths use SaveDepthsCallback.
    common = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(t["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=int(t["warmup_steps"]),
        weight_decay=float(t["weight_decay"]),
        adam_beta1=float(t["adam_beta1"]),
        adam_beta2=float(t["adam_beta2"]),
        adam_epsilon=float(t["adam_epsilon"]),
        max_grad_norm=float(t["max_grad_norm"]),
        bf16=bool(t["bf16"]),
        gradient_checkpointing=bool(t["gradient_checkpointing"]),
        max_length=int(t["max_seq_length"]),
        packing=bool(t["packing"]),
        logging_steps=10,
        save_strategy="no" if max_steps is not None else "epoch",
        save_total_limit=1,
        report_to="none",
        deepspeed=str(root / t["deepspeed_config"]),
        seed=seed,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        model_init_kwargs={
            "revision": cfg["model"]["revision"] if mode == "warmup" else None,
            "torch_dtype": torch.bfloat16,
            "local_files_only": True,
            "attn_implementation": "sdpa",
        },
    )
    # Drop None revision when loading from local checkpoint path
    if common["model_init_kwargs"]["revision"] is None:
        del common["model_init_kwargs"]["revision"]

    if mode == "warmup":
        common.update(
            num_train_epochs=float(cfg["warmup"]["num_epochs"]),
            completion_only_loss=True,
            save_strategy="epoch",
        )
    else:
        common.update(
            dataset_text_field="text",
            completion_only_loss=False,
        )
        if max_steps is not None:
            common["max_steps"] = max_steps
            common["num_train_epochs"] = 1.0
        else:
            # One pass over the provided jsonl (ordered).
            common["num_train_epochs"] = 1.0

    return SFTConfig(**common)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--mode", choices=["warmup", "imdb"], required=True)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HF id or local ckpt. Default: config model (warmup) or required for imdb.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="jsonl path. Default: alpaca_warmup_order (warmup) or must pass for imdb.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dir. Default: outputs/warmup/W0 or outputs/imdb/<name>",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--save-depths",
        type=str,
        default=None,
        help="Comma-separated steps to dump consolidated HF weights (e.g. 128,256,512).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    root = args.config.resolve().parent
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg["train"]["seeds"][0])
    set_seed(seed)
    depths = parse_depths(args.save_depths)

    if args.mode == "warmup":
        data_path = args.data or (root / cfg["paths"]["splits_dir"] / "alpaca_warmup_order.jsonl")
        out_dir = args.output or (root / cfg["paths"]["output_dir"] / "warmup" / "W0")
        model_name = args.model or cfg["model"]["name_or_path"]
    else:
        if args.data is None:
            raise SystemExit("--data is required for --mode imdb")
        data_path = args.data if args.data.is_absolute() else root / args.data
        out_dir = args.output or (
            root / cfg["paths"]["output_dir"] / "imdb" / data_path.stem
        )
        if args.model is None:
            raise SystemExit("--model is required for --mode imdb (start from W0 or stage ckpt)")
        model_name = args.model

    out_dir = out_dir if out_dir.is_absolute() else root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_path if data_path.is_absolute() else root / data_path

    if depths and args.max_steps is not None:
        bad = sorted(d for d in depths if d > args.max_steps or d <= 0)
        if bad:
            raise SystemExit(f"--save-depths out of range for max_steps={args.max_steps}: {bad}")

    ds = build_dataset(cfg, root, args.mode, data_path)
    sft_args = make_sft_args(cfg, root, out_dir, args.mode, args.max_steps, seed)

    tok = AutoTokenizer.from_pretrained(
        model_name if Path(model_name).exists() else cfg["model"]["name_or_path"],
        revision=None if Path(model_name).exists() else cfg["model"]["revision"],
        local_files_only=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # When resuming from local W0, pass path string without hub revision.
    model_arg: Any = model_name

    depth_cb = SaveDepthsCallback(depths, out_dir) if depths else None
    trainer = SFTTrainer(
        model=model_arg,
        args=sft_args,
        train_dataset=ds,
        processing_class=tok,
        callbacks=[depth_cb] if depth_cb is not None else None,
    )
    if depth_cb is not None:
        depth_cb.trainer = trainer

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    elapsed = time.time() - t0
    peak_gb = (
        torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None
    )

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    if is_main_process():
        metrics = {
            "mode": args.mode,
            "model": model_name,
            "data": str(data_path.relative_to(root)) if root in data_path.parents else str(data_path),
            "output": str(out_dir.relative_to(root)),
            "seed": seed,
            "max_steps": args.max_steps,
            "save_depths": sorted(depths) if depths else [],
            "train_loss": float(result.training_loss)
            if hasattr(result, "training_loss")
            else None,
            "train_runtime_sec": elapsed,
            "peak_cuda_mem_gb_rank_local": peak_gb,
            "n_examples": len(ds),
            "completion_only_loss": args.mode == "warmup",
            "optimizer_reset": True,
        }
        write_json(out_dir / "train_metrics.json", metrics)
        print(metrics)


if __name__ == "__main__":
    main()
