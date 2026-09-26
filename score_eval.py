#!/usr/bin/env python3
"""Score generations with Sentiment-RoBERTa; positive→1, else 0."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import load_config, load_jsonl, write_json, write_jsonl


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Scored jsonl + summary json")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    root = args.config.resolve().parent
    cfg = load_config(args.config)
    gen_path = args.generations if args.generations.is_absolute() else root / args.generations
    out_path = args.output if args.output.is_absolute() else root / args.output

    clf_cfg = cfg["classifier"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(
        clf_cfg["name_or_path"],
        revision=clf_cfg["revision"],
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        clf_cfg["name_or_path"],
        revision=clf_cfg["revision"],
        local_files_only=True,
    ).to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    score_map = clf_cfg["score_map"]

    rows = load_jsonl(gen_path)
    scored = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        texts = [r["full_text"][:2000] for r in batch]
        inputs = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).tolist()
        for r, pred_id in zip(batch, preds):
            label = id2label[int(pred_id)]
            s = int(score_map.get(label, 0))
            scored.append({**r, "pred_label": label, "positive_score": s})

    # Aggregate: mean over prompts (average sample scores per prompt, then mean)
    by_id: dict[str, list[int]] = defaultdict(list)
    for r in scored:
        by_id[r["example_id"]].append(int(r["positive_score"]))
    per_prompt = {eid: sum(v) / len(v) for eid, v in by_id.items()}
    s_mean = sum(per_prompt.values()) / max(len(per_prompt), 1)

    write_jsonl(out_path, scored)
    summary = {
        "generations": str(gen_path),
        "n_generations": len(scored),
        "n_prompts": len(per_prompt),
        "positive_rate_S": s_mean,
        "score_map": score_map,
        "classifier": clf_cfg["name_or_path"],
        "classifier_revision": clf_cfg["revision"],
    }
    write_json(out_path.with_suffix(".summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
