#!/usr/bin/env python3
"""Split JSONL summary fields into paragraph-based snippet records."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"\S+")
PARAGRAPH_BREAK_RE = re.compile(r"(?:\r?\n\s*){2,}")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def split_paragraphs(text: str) -> list[str]:
    return [
        paragraph.strip()
        for paragraph in PARAGRAPH_BREAK_RE.split(text.strip())
        if paragraph.strip()
    ]


def split_into_snippets(
    text: str, min_words: int, min_final_words: int = 50
) -> list[str]:
    """Group contiguous paragraphs into snippets of at least min_words when possible."""
    snippets: list[str] = []
    current_paragraphs: list[str] = []
    current_words = 0

    def flush_current() -> None:
        nonlocal current_paragraphs, current_words
        if current_paragraphs:
            snippets.append("\n\n".join(current_paragraphs))
            current_paragraphs = []
            current_words = 0

    for paragraph in split_paragraphs(text):
        paragraph_words = count_words(paragraph)
        if paragraph_words == 0:
            continue

        current_paragraphs.append(paragraph)
        current_words += paragraph_words

        if current_words >= min_words:
            flush_current()

    if current_paragraphs:
        final_snippet = "\n\n".join(current_paragraphs)
        if snippets and current_words < min_final_words:
            snippets[-1] = f"{snippets[-1]}\n\n{final_snippet}"
        else:
            snippets.append(final_snippet)

    return snippets


def split_record(
    record: dict[str, Any],
    *,
    min_words: int,
    min_final_words: int,
    summary_field: str,
    snippet_fields: list[str],
    full_text_field: str | None,
    keep_summary: bool,
    add_metadata: bool,
) -> list[dict[str, Any]]:
    if summary_field not in record:
        raise KeyError(f"Missing required field: {summary_field!r}")

    summary = record[summary_field]
    if not isinstance(summary, str):
        raise TypeError(f"Field {summary_field!r} must contain a string")

    snippets = split_into_snippets(summary, min_words, min_final_words)
    split_records: list[dict[str, Any]] = []

    for snippet_index, snippet in enumerate(snippets):
        split_record = copy.deepcopy(record)
        if full_text_field is not None:
            split_record[full_text_field] = summary
        for snippet_field in snippet_fields:
            split_record[snippet_field] = snippet
        if not keep_summary:
            if summary_field not in snippet_fields and summary_field != full_text_field:
                del split_record[summary_field]
        if add_metadata:
            split_record["snippet_index"] = snippet_index
            split_record["snippet_count"] = len(snippets)
            split_record["snippet_word_count"] = count_words(snippet)
        split_records.append(split_record)

    return split_records


def split_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    min_words: int,
    min_final_words: int,
    summary_field: str,
    snippet_fields: list[str],
    full_text_field: str | None,
    keep_summary: bool,
    add_metadata: bool,
) -> tuple[int, int]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")
    if not snippet_fields:
        raise ValueError("At least one snippet field is required")
    if full_text_field is not None and full_text_field in snippet_fields:
        raise ValueError("Full text field must be different from snippet fields")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_records = 0
    output_records = 0

    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue

            input_records += 1
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise TypeError("JSONL record must be an object")
                split_records = split_record(
                    record,
                    min_words=min_words,
                    min_final_words=min_final_words,
                    summary_field=summary_field,
                    snippet_fields=snippet_fields,
                    full_text_field=full_text_field,
                    keep_summary=keep_summary,
                    add_metadata=add_metadata,
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid record on line {line_number}: {exc}") from exc

            for split_item in split_records:
                outfile.write(json.dumps(split_item, ensure_ascii=False))
                outfile.write("\n")
                output_records += 1

    return input_records, output_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a JSONL file and expand each record into one record per summary "
            "snippet. Snippets are contiguous paragraphs grouped until they reach "
            "the target minimum word count."
        )
    )
    parser.add_argument("input", type=Path, help="Input JSONL path.")
    parser.add_argument("output", type=Path, help="Output JSONL path.")
    parser.add_argument(
        "--min-words",
        type=int,
        default=None,
        help="Target minimum words per snippet. Defaults to 150.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Deprecated alias for --min-words.",
    )
    parser.add_argument(
        "--min-final-words",
        type=int,
        default=50,
        help=(
            "If the last snippet has fewer words than this, merge it into the "
            "previous snippet. Defaults to 50."
        ),
    )
    parser.add_argument(
        "--summary-field",
        default="summary",
        help="Field to split into snippets. Defaults to 'summary'.",
    )
    parser.add_argument(
        "--snippet-field",
        nargs="+",
        action="append",
        default=None,
        help=(
            "Field or fields that receive each snippet. Can be passed once with "
            "multiple names or repeated. Defaults to 'snippet_text'."
        ),
    )
    parser.add_argument(
        "--full-text-field",
        default=None,
        help=(
            "Optional field that receives the full unsplit text from the summary "
            "field in each output record."
        ),
    )
    parser.add_argument(
        "--drop-summary",
        action="store_true",
        help="Remove the original summary field from each output record.",
    )
    parser.add_argument(
        "--add-metadata",
        action="store_true",
        help="Add snippet_index, snippet_count, and snippet_word_count fields.",
    )
    return parser.parse_args()


def normalize_snippet_fields(snippet_field_args: list[list[str]] | None) -> list[str]:
    if snippet_field_args is None:
        return ["snippet_text"]

    fields: list[str] = []
    seen: set[str] = set()
    for group in snippet_field_args:
        for field in group:
            if field in seen:
                continue
            fields.append(field)
            seen.add(field)
    return fields


def main() -> None:
    args = parse_args()
    min_words = args.min_words if args.min_words is not None else args.max_words
    snippet_fields = normalize_snippet_fields(args.snippet_field)
    if min_words is None:
        min_words = 150
    if min_words < 1:
        raise SystemExit("--min-words must be at least 1")
    if args.min_final_words < 1:
        raise SystemExit("--min-final-words must be at least 1")
    if args.full_text_field is not None and args.full_text_field in snippet_fields:
        raise SystemExit("--full-text-field must be different from --snippet-field values")

    input_records, output_records = split_jsonl(
        args.input,
        args.output,
        min_words=min_words,
        min_final_words=args.min_final_words,
        summary_field=args.summary_field,
        snippet_fields=snippet_fields,
        full_text_field=args.full_text_field,
        keep_summary=not args.drop_summary,
        add_metadata=args.add_metadata,
    )
    print(
        f"Wrote {output_records} records from {input_records} input records "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
