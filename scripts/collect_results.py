#!/usr/bin/env python3
"""Collect BLEU/ROUGE/NUM_WORDS metrics from train/eval runs into a DataFrame."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from typing import Any

import pandas as pd

#panza_david_anonymous-Llama-3.2-1B-Instruct-bf16-bs16-lora-r32-lr6.3e-05-5ep-seed44
#panza_david_anonymous-Llama-3.2-1B-Instruct-bf16-bs16-lora-r16-lr0.00011-4ep-seed44

RUN_NAME_RE = re.compile(
    r"^panza_(?P<head>.+)-(?P<precision>[^-]+)-bs(?P<batch_size>\d+)-"
    r"(?P<finetuning>[^-]+)(?:-r(?P<rank>\d+))?-lr(?P<lr>.+)-"
    r"(?P<epochs>\d+)ep-seed(?P<seed>\d+)$"
)


def _extract_aggregate_metrics(path: str) -> dict[str, Any]:
    key = '"aggregate_metrics"'
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        prefix = f.read(65536)
    idx = prefix.find(key)
    if idx != -1:
        colon = prefix.find(":", idx + len(key))
        if colon != -1:
            try:
                return decoder.raw_decode(prefix[colon + 1 :].lstrip())[0]
            except json.JSONDecodeError:
                pass

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["aggregate_metrics"]


def _parse_run_name(run_name: str) -> dict[str, Any]:
    row: dict[str, Any] = {"run_name": run_name}
    match = RUN_NAME_RE.match(run_name)
    if not match:
        return row

    head = match.group("head")
    if "_anonymous-" in head:
        user, model = head.split("_anonymous-", 1)
        row["anonymous"] = True
    elif "-" in head:
        user, model = head.split("-", 1)
        row["anonymous"] = False
    else:
        user, model = head, None
        row["anonymous"] = False

    row.update(
        {
            "user": user,
            "model": model,
            "precision": match.group("precision"),
            "batch_size": int(match.group("batch_size")),
            "finetuning": match.group("finetuning"),
            "lora_rank": (
                int(match.group("rank")) if match.group("rank") is not None else None
            ),
            "learning_rate": float(match.group("lr")),
            "epochs": int(match.group("epochs")),
            "seed": int(match.group("seed")),
        }
    )
    return row


def _collect_file(path: str) -> dict[str, Any]:
    run_name = os.path.basename(os.path.dirname(path))
    row = _parse_run_name(run_name)
    row["result_path"] = path

    metrics = _extract_aggregate_metrics(path)
    row["BLEU"] = metrics.get("BLEU")
    row["NUM_WORDS"] = metrics.get("NUM_WORDS")
    row["NUM_WORDS_GOLDEN"] = metrics.get("NUM_WORDS_GOLDEN")

    for key, value in metrics.get("ROUGE", {}).items():
        row[f"ROUGE_{key}"] = value

    return row


def _resolve_max_workers(file_count: int, max_workers: int | None) -> int:
    if file_count <= 0:
        return 0
    if max_workers is not None:
        return min(file_count, max_workers)
    return min(file_count, 32, max(4, (os.cpu_count() or 1) * 4))


def collect_results(
    models_dir: str = "../checkpoints/models",
    result_filename: str = "test_outputs.json",
    max_workers: int | None = None,
    lora: bool = False
) -> pd.DataFrame:
    files: list[str] = []
    with os.scandir(models_dir) as entries:
        for entry in entries:
            if not entry.is_dir():
                continue
            candidate = os.path.join(entry.path, result_filename)
            if os.path.isfile(candidate):
                files.append(candidate)

    files.sort()
    print(f"there are {len(files)} files")
    if not files:
        return pd.DataFrame()

    if lora:
        fiq = 'lora'
    else:
        fiq = "fft"
    lora_files = [path for path in files if fiq in path]
    if not lora_files:
        return pd.DataFrame()

    worker_count = _resolve_max_workers(len(lora_files), max_workers)
    if worker_count == 1:
        rows = [_collect_file(path) for path in lora_files]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            rows = list(executor.map(_collect_file, lora_files))

    df = pd.DataFrame(rows)
    if "BLEU" in df.columns:
        df = df.sort_values("BLEU", ascending=False, ignore_index=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect BLEU/ROUGE/NUM_WORDS from run outputs into a pandas DataFrame."
        )
    )
    parser.add_argument(
        "--models-dir",
        default="../checkpoints/models",
        help="Directory that contains run folders.",
    )
    parser.add_argument(
        "--result-file",
        default="test_outputs.json",
        help="Result filename expected inside each run folder.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional thread count for reading result JSON files.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional output path (.csv or .parquet). If omitted, prints the DataFrame.",
    )
    args = parser.parse_args()
    if args.max_workers is not None and args.max_workers < 1:
        parser.error("--max-workers must be a positive integer.")

    df = collect_results(
        args.models_dir,
        args.result_file,
        max_workers=args.max_workers,
    )
    if args.out:
        if args.out.endswith(".parquet"):
            df.to_parquet(args.out, index=False)
        else:
            df.to_csv(args.out, index=False)
    else:
        print(df.to_string(index=False))

    print(df.columns)


if __name__ == "__main__":
    main()
