#!/usr/bin/env python3
"""Step 1: freeze IMDb / Alpaca splits, prefixes, and token–label inspections."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def clean_imdb_text(text: str) -> str:
    text = BR_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def example_id_hash_prefix_len(example_id: str, lo: int, hi: int) -> int:
    digest = hashlib.sha256(example_id.encode("utf-8")).hexdigest()
    # Map to inclusive [lo, hi]
    return lo + (int(digest[:8], 16) % (hi - lo + 1))


def format_alpaca(example: dict[str, str], template: str) -> str:
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()
    if inp:
        text = template.format(instruction=instruction, input=inp, output=output)
    else:
        # Drop the Input block when empty (config note).
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
    return text


def sample_indices(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if n > len(pool):
        raise ValueError(f"Need {n} samples but pool only has {len(pool)}")
    chosen = rng.choice(pool, size=n, replace=False)
    return np.sort(chosen)


def balanced_sample_by_label(
    labels: np.ndarray,
    n_total: int,
    rng: np.random.Generator,
    exclude: set[int] | None = None,
) -> np.ndarray:
    assert n_total % 2 == 0
    n_each = n_total // 2
    exclude = exclude or set()
    pos = np.array([i for i, y in enumerate(labels) if y == 1 and i not in exclude], dtype=np.int64)
    neg = np.array([i for i, y in enumerate(labels) if y == 0 and i not in exclude], dtype=np.int64)
    if len(pos) < n_each or len(neg) < n_each:
        raise ValueError(
            f"Not enough labeled examples for balanced {n_total} "
            f"(pos={len(pos)}, neg={len(neg)}, need {n_each} each)"
        )
    take_pos = sample_indices(pos, n_each, rng)
    take_neg = sample_indices(neg, n_each, rng)
    out = np.concatenate([take_pos, take_neg])
    rng.shuffle(out)
    return out


def build_imdb_records(
    texts: list[str],
    labels: list[int],
    indices: np.ndarray,
    split_name: str,
    tokenizer,
    prefix_lo: int,
    prefix_hi: int,
) -> list[dict[str, Any]]:
    records = []
    for idx in indices.tolist():
        text = clean_imdb_text(texts[idx])
        label = int(labels[idx])
        eid = f"imdb_{split_name}_{idx}"
        prefix_len = example_id_hash_prefix_len(eid, prefix_lo, prefix_hi)
        # Tokenize without special tokens for prefix extraction (continuation prompt).
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < prefix_lo:
            # Skip ultra-short; caller should ensure enough remain — keep with clamped len.
            prefix_len = max(1, min(prefix_len, len(token_ids)))
        else:
            prefix_len = min(prefix_len, len(token_ids))
        prefix_ids = token_ids[:prefix_len]
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
        records.append(
            {
                "example_id": eid,
                "source_split": split_name,
                "source_index": idx,
                "label": label,  # 0 neg, 1 pos (IMDb)
                "text": text,
                "text_sha256": sha256_text(text),
                "prefix_len": prefix_len,
                "prefix_token_ids": prefix_ids,
                "prefix_text": prefix_text,
            }
        )
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    blob = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    path.write_bytes(blob)
    return sha256_file_bytes(blob)


def inspect_imdb(tokenizer, records: list[dict[str, Any]], n: int = 10) -> list[dict[str, Any]]:
    out = []
    for r in records[:n]:
        ids = tokenizer.encode(r["text"], add_special_tokens=True)
        # BOS + tokens + EOS pattern for training (as locked); labels ignore BOS/pad.
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
        if eos is not None and (not ids or ids[-1] != eos):
            ids = ids + [eos]
        labels = [-100 if (i == 0 and bos is not None) else tid for i, tid in enumerate(ids)]
        # First token is BOS → -100; rest including EOS contribute to loss.
        if bos is not None:
            labels[0] = -100
        out.append(
            {
                "example_id": r["example_id"],
                "n_tokens": len(ids),
                "token_ids_head": ids[:32],
                "labels_head": labels[:32],
                "decoded_head": tokenizer.decode(ids[:32], skip_special_tokens=False),
                "prefix_len": r["prefix_len"],
                "prefix_text": r["prefix_text"],
            }
        )
    return out


def inspect_alpaca(tokenizer, formatted: list[dict[str, Any]], n: int = 10) -> list[dict[str, Any]]:
    out = []
    for row in formatted[:n]:
        full = row["text"]
        # Find response span for loss_on=output_only
        marker = "### Response:\n"
        pos = full.rfind(marker)
        assert pos >= 0
        prompt_part = full[: pos + len(marker)]
        response_part = full[pos + len(marker) :]
        prompt_ids = tokenizer.encode(prompt_part, add_special_tokens=True)
        # Continuation without re-adding BOS
        resp_ids = tokenizer.encode(response_part, add_special_tokens=False)
        eos = tokenizer.eos_token_id
        if eos is not None and (not resp_ids or resp_ids[-1] != eos):
            resp_ids = resp_ids + [eos]
        input_ids = prompt_ids + resp_ids
        labels = [-100] * len(prompt_ids) + resp_ids
        out.append(
            {
                "example_id": row["example_id"],
                "n_tokens": len(input_ids),
                "n_prompt_tokens": len(prompt_ids),
                "n_response_tokens": len(resp_ids),
                "token_ids_head": input_ids[:40],
                "labels_head": labels[:40],
                "decoded_head": tokenizer.decode(input_ids[:40], skip_special_tokens=False),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = args.config.resolve().parent
    splits_dir = root / cfg["paths"]["splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    data_seed = int(cfg["train"]["seeds"][0])
    rng = np.random.default_rng(data_seed)

    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["name_or_path"],
        revision=cfg["model"]["revision"],
        local_files_only=True,
    )

    print("Loading IMDb…")
    imdb = load_dataset(
        cfg["imdb"]["name"],
        revision=cfg["imdb"]["revision"],
    )
    train = imdb["train"]
    test = imdb["test"]
    train_texts = [clean_imdb_text(t) for t in train["text"]]
    train_labels = list(train["label"])
    test_texts = [clean_imdb_text(t) for t in test["text"]]
    test_labels = list(test["label"])

    train_labels_np = np.array(train_labels, dtype=np.int64)
    test_labels_np = np.array(test_labels, dtype=np.int64)
    pos_train = np.where(train_labels_np == 1)[0]
    neg_train = np.where(train_labels_np == 0)[0]

    n_main = int(cfg["imdb_sft"]["n_per_stage_main"])
    n_disjoint = int(cfg["research"]["disjoint_n_per_stage"])
    prefix_lo = int(cfg["data_splits"]["prefix_len_min"])
    prefix_hi = int(cfg["data_splits"]["prefix_len_max"])

    # Main experiment: reused positive 8192 + negative 8192
    main_pos = sample_indices(pos_train, n_main, rng)
    main_neg = sample_indices(neg_train, n_main, rng)

    # Disjoint control: two non-overlapping positive sets of 6144
    # (independent arm; may overlap main_pos membership)
    shuffled_pos = pos_train.copy()
    rng.shuffle(shuffled_pos)
    if 2 * n_disjoint > len(shuffled_pos):
        raise ValueError("Not enough positives for disjoint control")
    disjoint_s1 = np.sort(shuffled_pos[:n_disjoint])
    disjoint_s3 = np.sort(shuffled_pos[n_disjoint : 2 * n_disjoint])
    assert len(set(disjoint_s1.tolist()) & set(disjoint_s3.tolist())) == 0

    # Same negatives reused for disjoint arm stages (matched n via config research flag)
    # Use a separate negative draw for the control arm for independence from main order.
    disjoint_neg = sample_indices(neg_train, n_disjoint, rng)

    # Eval: 2000 calib + 2000 final from test, balanced, disjoint
    calib_n = int(cfg["data_splits"]["calib_n"])
    test_n = int(cfg["data_splits"]["test_n"])
    calib_idx = balanced_sample_by_label(test_labels_np, calib_n, rng)
    test_idx = balanced_sample_by_label(
        test_labels_np, test_n, rng, exclude=set(calib_idx.tolist())
    )
    assert len(set(calib_idx.tolist()) & set(test_idx.tolist())) == 0

    def pack(name: str, indices: np.ndarray, split: str, texts, labels):
        recs = build_imdb_records(
            texts, labels, indices, split, tokenizer, prefix_lo, prefix_hi
        )
        path = splits_dir / f"{name}.jsonl"
        file_hash = write_jsonl(path, recs)
        return {
            "name": name,
            "path": str(path.relative_to(root)),
            "n": len(recs),
            "file_sha256": file_hash,
            "label_counts": {
                "neg": sum(1 for r in recs if r["label"] == 0),
                "pos": sum(1 for r in recs if r["label"] == 1),
            },
            "ids_sha256": sha256_text(",".join(r["example_id"] for r in recs)),
        }, recs

    manifest_splits = {}
    all_recs = {}

    meta, recs = pack("main_positive_8192", main_pos, "train", train_texts, train_labels)
    manifest_splits[meta["name"]] = meta
    all_recs[meta["name"]] = recs

    meta, recs = pack("main_negative_8192", main_neg, "train", train_texts, train_labels)
    manifest_splits[meta["name"]] = meta
    all_recs[meta["name"]] = recs

    meta, recs = pack(
        "disjoint_stage1_positive_6144", disjoint_s1, "train", train_texts, train_labels
    )
    manifest_splits[meta["name"]] = meta
    all_recs[meta["name"]] = recs

    meta, recs = pack(
        "disjoint_stage3_positive_6144", disjoint_s3, "train", train_texts, train_labels
    )
    manifest_splits[meta["name"]] = meta
    all_recs[meta["name"]] = recs

    meta, recs = pack(
        "disjoint_negative_6144", disjoint_neg, "train", train_texts, train_labels
    )
    manifest_splits[meta["name"]] = meta
    all_recs[meta["name"]] = recs

    meta, recs = pack("calib_2000", calib_idx, "test", test_texts, test_labels)
    manifest_splits[meta["name"]] = meta
    all_recs[meta["name"]] = recs

    meta, recs = pack("final_test_2000", test_idx, "test", test_texts, test_labels)
    manifest_splits[meta["name"]] = meta
    all_recs[meta["name"]] = recs

    # Alpaca: freeze order of all indices (full set for 1-epoch warmup)
    print("Loading Alpaca-cleaned…")
    alpaca = load_dataset(
        cfg["warmup_data"]["name"],
        revision=cfg["warmup_data"]["revision"],
    )["train"]
    template = cfg["warmup"]["template"]
    alpaca_rows = []
    for i in range(len(alpaca)):
        ex = {
            "instruction": alpaca[i]["instruction"],
            "input": alpaca[i]["input"],
            "output": alpaca[i]["output"],
        }
        text = format_alpaca(ex, template)
        if cfg["warmup"].get("add_eos"):
            # EOS added at tokenize time; store plain text here.
            pass
        alpaca_rows.append(
            {
                "example_id": f"alpaca_{i}",
                "source_index": i,
                "instruction": ex["instruction"],
                "input": ex["input"],
                "output": ex["output"],
                "text": text,
                "text_sha256": sha256_text(text),
            }
        )
    # Deterministic shuffle for training order using data_seed
    order = np.arange(len(alpaca_rows))
    rng.shuffle(order)
    alpaca_ordered = [alpaca_rows[i] for i in order.tolist()]
    alpaca_path = splits_dir / "alpaca_warmup_order.jsonl"
    alpaca_hash = write_jsonl(alpaca_path, alpaca_ordered)
    manifest_splits["alpaca_warmup_order"] = {
        "name": "alpaca_warmup_order",
        "path": str(alpaca_path.relative_to(root)),
        "n": len(alpaca_ordered),
        "file_sha256": alpaca_hash,
        "ids_sha256": sha256_text(",".join(r["example_id"] for r in alpaca_ordered)),
    }

    # Inspections (10 each)
    imdb_insp = inspect_imdb(tokenizer, all_recs["main_positive_8192"], n=10)
    alpaca_insp = inspect_alpaca(tokenizer, alpaca_ordered, n=10)
    insp_path = splits_dir / "token_label_inspection.json"
    insp_payload = {
        "imdb_main_positive_10": imdb_insp,
        "alpaca_10": alpaca_insp,
    }
    insp_path.write_text(json.dumps(insp_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Sanity checks
    overlap_d = set(
        r["source_index"] for r in all_recs["disjoint_stage1_positive_6144"]
    ) & set(r["source_index"] for r in all_recs["disjoint_stage3_positive_6144"])
    overlap_eval = set(r["source_index"] for r in all_recs["calib_2000"]) & set(
        r["source_index"] for r in all_recs["final_test_2000"]
    )
    assert not overlap_d
    assert not overlap_eval

    manifest = {
        "data_seed": data_seed,
        "model_revision": cfg["model"]["revision"],
        "imdb_revision": cfg["imdb"]["revision"],
        "alpaca_revision": cfg["warmup_data"]["revision"],
        "splits": manifest_splits,
        "checks": {
            "disjoint_stage1_stage3_overlap": 0,
            "calib_final_test_overlap": 0,
            "main_positive_n": manifest_splits["main_positive_8192"]["n"],
            "main_negative_n": manifest_splits["main_negative_8192"]["n"],
            "disjoint_each_n": n_disjoint,
            "calib_label_counts": manifest_splits["calib_2000"]["label_counts"],
            "final_test_label_counts": manifest_splits["final_test_2000"]["label_counts"],
        },
        "inspection_path": str(insp_path.relative_to(root)),
    }
    man_path = splits_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable preview for manual check
    preview_path = splits_dir / "MANUAL_CHECK.md"
    lines = [
        "# Step 1 manual token/label check (first 3 IMDb + 3 Alpaca)",
        "",
        "## IMDb (main positive)",
        "",
    ]
    for row in imdb_insp[:3]:
        lines.append(f"### {row['example_id']} (prefix_len={row['prefix_len']})")
        lines.append(f"- prefix: {row['prefix_text']!r}")
        lines.append(f"- n_tokens: {row['n_tokens']}")
        lines.append(f"- labels_head: `{row['labels_head']}`")
        lines.append(f"- decoded_head: {row['decoded_head']!r}")
        lines.append("")
    lines.append("## Alpaca")
    lines.append("")
    for row in alpaca_insp[:3]:
        lines.append(
            f"### {row['example_id']} "
            f"(prompt={row['n_prompt_tokens']}, resp={row['n_response_tokens']})"
        )
        lines.append(f"- labels_head: `{row['labels_head']}`")
        lines.append(f"- decoded_head: {row['decoded_head']!r}")
        lines.append("")
    preview_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(manifest["checks"], indent=2))
    print(f"Wrote manifest → {man_path}")
    print(f"Inspection → {insp_path}")
    print(f"Manual preview → {preview_path}")
    print("Step 1 prepare_data finished OK.")


if __name__ == "__main__":
    main()
