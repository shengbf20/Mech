#!/usr/bin/env python3
"""Shared helpers for mech experiment scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def alpaca_prompt_completion(row: dict[str, Any]) -> tuple[str, str]:
    """Split frozen alpaca `text` into prompt (incl. Response header) and completion."""
    text = row["text"]
    marker = "### Response:\n"
    pos = text.rfind(marker)
    if pos < 0:
        raise ValueError(f"Missing Response marker in {row.get('example_id')}")
    prompt = text[: pos + len(marker)]
    completion = text[pos + len(marker) :]
    return prompt, completion
