#!/usr/bin/env python3
"""Build an SFT messages JSON from KTO and state-reconstruction datasets.

The KTO side contributes only label=True rows. The state-reconstruction side is
treated as auxiliary SFT data; if it has a label column, only label=True rows are
kept, otherwise all valid rows are kept.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _as_bool_label(value: Any) -> bool | None:
    if value is None or _is_nan(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _rows_from_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return [row for row in obj if isinstance(row, dict)]
    if isinstance(obj, dict):
        keys = list(obj.keys())
        size = max((len(obj[key]) for key in keys if isinstance(obj[key], list)), default=0)
        return [
            {
                key: obj[key][idx]
                if isinstance(obj[key], list) and idx < len(obj[key])
                else obj[key]
                for key in keys
            }
            for idx in range(size)
        ]
    raise ValueError(f"Unsupported JSON dataset type in {path}: {type(obj).__name__}")


def _load_rows(path: str, split: str) -> list[dict[str, Any]]:
    path = os.path.abspath(path)
    if os.path.isdir(path):
        dataset = load_from_disk(path)
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise ValueError(f"Split '{split}' not found in {path}; available={list(dataset.keys())}")
            dataset = dataset[split]
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Unsupported dataset object from {path}: {type(dataset).__name__}")
        return [dict(row) for row in dataset]

    if path.endswith(".json"):
        return _rows_from_json(path)

    raise ValueError(f"Expected a Hugging Face dataset directory or .json file: {path}")


def _valid_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or not messages:
        return False
    for message in messages:
        if not isinstance(message, dict):
            return False
        if str(message.get("content", "")).strip() == "":
            return False
    return True


def _row_to_sft_record(row: dict[str, Any], require_label_true: bool) -> dict[str, Any] | None:
    if "label" in row:
        label = _as_bool_label(row.get("label"))
        if label is None:
            return None
        if require_label_true and not label:
            return None
        if not require_label_true and label is False:
            return None
    elif require_label_true:
        return None

    messages = row.get("messages")
    if _valid_messages(messages):
        return {"messages": messages}

    prompt = row.get("prompt")
    completion = row.get("completion")
    if not isinstance(prompt, list) or not isinstance(completion, list):
        return None
    messages = prompt + completion
    if not _valid_messages(messages):
        return None
    return {"messages": messages}


def build_sft_records(
    kto_dataset_path: str | None,
    state_recon_dataset_path: str | None,
    split: str,
) -> list[dict[str, Any]]:
    kto_rows = _load_rows(kto_dataset_path, split) if kto_dataset_path else []
    kto_records = [
        record
        for row in kto_rows
        if (record := _row_to_sft_record(row, require_label_true=True)) is not None
    ]

    state_records: list[dict[str, Any]] = []
    state_rows_count = 0
    if state_recon_dataset_path:
        state_rows = _load_rows(state_recon_dataset_path, split)
        state_rows_count = len(state_rows)
        state_records = [
            record
            for row in state_rows
            if (record := _row_to_sft_record(row, require_label_true=False)) is not None
        ]

    print(f"kto_rows={len(kto_rows)}")
    print(f"kto_label_true_sft={len(kto_records)}")
    if state_recon_dataset_path:
        print(f"state_recon_rows={state_rows_count}")
        print(f"state_recon_sft={len(state_records)}")
    print(f"total_sft_samples={len(kto_records) + len(state_records)}")

    return kto_records + state_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert KTO/state-recon datasets into SFT messages JSON.")
    parser.add_argument("--kto_dataset_path", default=None)
    parser.add_argument("--state_recon_dataset_path", default=None)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--dataset_split", default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_sft_records(
        kto_dataset_path=args.kto_dataset_path,
        state_recon_dataset_path=args.state_recon_dataset_path,
        split=args.dataset_split,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"sft_path={args.out_path}")


if __name__ == "__main__":
    main()
