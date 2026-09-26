#!/usr/bin/env python3
"""Step 2 pathway smoke: short SFT → save → reload → generate → classify."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

# Prefer local cache when offline / no Hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def train_pathway(cfg: dict[str, Any], root: Path, max_steps: int) -> Path:
    set_seed(int(cfg["train"]["seeds"][0]))
    out_dir = root / cfg["paths"]["output_dir"] / "pathway" / "ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = root / cfg["paths"]["splits_dir"] / "main_positive_8192.jsonl"
    rows = load_jsonl(train_path)
    # 32 steps × global batch 16 = 512 examples; take a bit more for safety.
    n_need = max_steps * int(cfg["train"]["global_batch_size"]) + 64
    rows = rows[:n_need]
    ds = Dataset.from_list([{"text": r["text"]} for r in rows])

    model_kwargs = {
        "revision": cfg["model"]["revision"],
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
        "attn_implementation": "sdpa",
    }

    args = SFTConfig(
        output_dir=str(out_dir),
        max_steps=max_steps,
        per_device_train_batch_size=int(cfg["train"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["train"]["gradient_accumulation_steps"]),
        learning_rate=float(cfg["train"]["learning_rate"]),
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        warmup_steps=int(cfg["train"]["warmup_steps"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        adam_beta1=float(cfg["train"]["adam_beta1"]),
        adam_beta2=float(cfg["train"]["adam_beta2"]),
        adam_epsilon=float(cfg["train"]["adam_epsilon"]),
        max_grad_norm=float(cfg["train"]["max_grad_norm"]),
        bf16=bool(cfg["train"]["bf16"]),
        gradient_checkpointing=bool(cfg["train"]["gradient_checkpointing"]),
        max_length=int(cfg["train"]["max_seq_length"]),
        packing=bool(cfg["train"]["packing"]),
        dataset_text_field="text",
        logging_steps=1,
        save_strategy="steps",
        save_steps=max_steps,
        save_total_limit=1,
        report_to="none",
        deepspeed=str(root / cfg["train"]["deepspeed_config"]),
        seed=int(cfg["train"]["seeds"][0]),
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        model_init_kwargs=model_kwargs,
    )

    trainer = SFTTrainer(
        model=cfg["model"]["name_or_path"],
        args=args,
        train_dataset=ds,
        processing_class=AutoTokenizer.from_pretrained(
            cfg["model"]["name_or_path"],
            revision=cfg["model"]["revision"],
            local_files_only=True,
        ),
    )
    # Ensure pad token
    tok = trainer.processing_class
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    elapsed = time.time() - t0
    peak_gb = None
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    metrics = {
        "max_steps": max_steps,
        "train_loss": float(train_result.training_loss)
        if hasattr(train_result, "training_loss")
        else None,
        "train_runtime_sec": elapsed,
        "sec_per_step": elapsed / max_steps,
        "peak_cuda_mem_gb_rank_local": peak_gb,
        "global_batch_size": cfg["train"]["global_batch_size"],
        "checkpoint": str(out_dir.relative_to(root)),
    }
    if is_main_process():
        metrics_path = root / cfg["paths"]["output_dir"] / "pathway" / "train_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
    return out_dir


@torch.inference_mode()
def eval_pathway(cfg: dict[str, Any], root: Path, ckpt: Path, n_prompts: int = 8) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen_cfg = cfg["generation"]
    clf_cfg = cfg["classifier"]

    tok = AutoTokenizer.from_pretrained(str(ckpt), local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()

    clf_tok = AutoTokenizer.from_pretrained(
        clf_cfg["name_or_path"],
        revision=clf_cfg["revision"],
        local_files_only=True,
    )
    clf = AutoModelForSequenceClassification.from_pretrained(
        clf_cfg["name_or_path"],
        revision=clf_cfg["revision"],
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(device)
    clf.eval()
    id2label = {int(k): v for k, v in clf.config.id2label.items()}

    calib = load_jsonl(root / cfg["paths"]["splits_dir"] / "calib_2000.jsonl")[:n_prompts]
    generations = []
    labels = []
    for row in calib:
        prefix = row["prefix_text"]
        inputs = tok(prefix, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            do_sample=bool(gen_cfg["do_sample"]),
            temperature=float(gen_cfg["temperature"]),
            top_p=float(gen_cfg["top_p"]),
            top_k=int(gen_cfg["top_k"]) if int(gen_cfg["top_k"]) > 0 else None,
            max_new_tokens=int(gen_cfg["max_new_tokens"]),
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        continuation = tok.decode(new_tokens, skip_special_tokens=True)
        full_text = prefix + continuation

        clf_in = clf_tok(
            full_text[:2000],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)
        logits = clf(**clf_in).logits[0]
        pred_id = int(torch.argmax(logits).item())
        pred_label = id2label[pred_id]
        score = int(clf_cfg["score_map"].get(pred_label, 0))
        generations.append(
            {
                "example_id": row["example_id"],
                "prefix_text": prefix,
                "continuation": continuation,
                "pred_label": pred_label,
                "positive_score": score,
            }
        )
        labels.append(score)

    summary = {
        "n_prompts": len(labels),
        "positive_rate": sum(labels) / max(len(labels), 1),
        "checkpoint": str(ckpt.relative_to(root)),
        "generations": generations,
    }
    out_path = root / cfg["paths"]["output_dir"] / "pathway" / "eval_smoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_prompts", "positive_rate", "checkpoint")}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument(
        "--stage",
        choices=["train", "eval", "all"],
        default="all",
        help="train under deepspeed/accelerate; eval is single-process",
    )
    # Injected by DeepSpeed / torch.distributed launchers.
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    root = args.config.resolve().parent
    cfg = load_config(args.config)

    ckpt = root / cfg["paths"]["output_dir"] / "pathway" / "ckpt"
    if args.stage in ("train", "all"):
        ckpt = train_pathway(cfg, root, args.max_steps)
        # Multi-process train: only rank0 continues to eval when --stage all
        if args.stage == "all" and not is_main_process():
            return
        # Barrier-ish: give rank0 a moment; deepspeed may tear down differently
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    if args.stage in ("eval", "all"):
        if args.stage == "all" and not is_main_process():
            return
        # Eval should be single-GPU; if still in distributed group, skip non-zero
        if int(os.environ.get("LOCAL_RANK", "0")) != 0:
            return
        eval_pathway(cfg, root, ckpt)
        report = {
            "step": 2,
            "status": "pathway_ok",
            "train_metrics": str(
                (root / cfg["paths"]["output_dir"] / "pathway" / "train_metrics.json").relative_to(root)
            ),
            "eval_smoke": str(
                (root / cfg["paths"]["output_dir"] / "pathway" / "eval_smoke.json").relative_to(root)
            ),
        }
        report_path = root / cfg["paths"]["output_dir"] / "pathway" / "STEP2_REPORT.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Step 2 pathway finished OK →", report_path)


if __name__ == "__main__":
    main()
