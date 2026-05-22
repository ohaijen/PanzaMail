#!/usr/bin/env python3
"""
Build prompt-completion JSONL files from ISTA-DASLab/Panza-emails.

This script is intended for MLX/HF LoRA training and writes:
  - train.jsonl
  - valid.jsonl
  - test.jsonl

Each row is in the form:
  {"prompt": "...", "completion": "...", "source_config": "..."}
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from datasets import get_dataset_config_names, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Panza-emails JSONL splits for LoRA training.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ISTA-DASLab/Panza-emails",
        help="Hugging Face dataset ID.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="david",
        help="Dataset config name (e.g., david/isabel/marcus) or 'all'.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where train/valid/test JSONL files will be written.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=41,
        help="Shuffle seed for split generation.",
    )
    parser.add_argument(
        "--subject-template",
        type=str,
        default="Write a concise professional email for this subject: {subject}",
        help="Template used to build the prompt from the source subject.",
    )
    parser.add_argument(
        "--merge-splits",
        action="store_true",
        default=False,
        help="If set, merge all source splits (train/test/valid) before re-splitting.",
    )
    return parser.parse_args()


def collect_rows(
    dataset_id: str,
    config_name: str,
    subject_template: str,
    merge_splits: bool,
) -> List[Dict[str, str]]:
    ds = load_dataset(dataset_id, config_name)
    split_names = list(ds.keys())
    if not merge_splits:
        split_names = [name for name in ["train", "validation", "valid", "test"] if name in ds]

    rows: List[Dict[str, str]] = []
    for split in split_names:
        for example in ds[split]:
            subject = str(example.get("subject", "")).strip()
            email = str(example.get("email", "")).strip()
            if not subject or not email:
                continue
            rows.append(
                {
                    "prompt": subject_template.format(subject=subject),
                    "completion": email,
                    "source_config": config_name,
                }
            )
    return rows


def split_records(
    records: List[Dict[str, str]],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, str]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must be in (0, 1).")
    if not 0 <= valid_ratio < 1:
        raise ValueError("--valid-ratio must be in [0, 1).")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("--train-ratio + --valid-ratio must be < 1.")
    if len(records) < 3:
        raise ValueError("Need at least 3 records to create train/valid/test splits.")

    shuffled = list(records)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    n_train = max(1, int(total * train_ratio))
    n_valid = max(1, int(total * valid_ratio))

    # Ensure all splits remain non-empty.
    if n_train + n_valid >= total:
        n_valid = max(1, total - n_train - 1)
    if n_train + n_valid >= total:
        n_train = max(1, total - n_valid - 1)

    train = shuffled[:n_train]
    valid = shuffled[n_train : n_train + n_valid]
    test = shuffled[n_train + n_valid :]

    if not train or not valid or not test:
        raise RuntimeError("Split generation failed; one split is empty.")

    return {"train": train, "valid": valid, "test": test}


def write_jsonl(path: Path, rows: Iterable[Dict[str, str]]) -> int:
    count = 0
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()

    config_names: List[str]
    if args.config == "all":
        config_names = get_dataset_config_names(args.dataset)
    else:
        config_names = [args.config]

    all_rows: List[Dict[str, str]] = []
    per_config_counts: Counter[str] = Counter()
    for cfg in config_names:
        rows = collect_rows(
            dataset_id=args.dataset,
            config_name=cfg,
            subject_template=args.subject_template,
            merge_splits=args.merge_splits,
        )
        all_rows.extend(rows)
        per_config_counts[cfg] += len(rows)

    if not all_rows:
        raise RuntimeError("No usable rows were collected from the dataset.")

    splits = split_records(
        records=all_rows,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, rows in splits.items():
        written[name] = write_jsonl(args.out_dir / f"{name}.jsonl", rows)

    print(f"Output directory: {args.out_dir}")
    print(f"Configs used: {', '.join(config_names)}")
    print(f"Rows per config: {dict(per_config_counts)}")
    print(f"Split sizes: {written}")


if __name__ == "__main__":
    main()
