#!/usr/bin/env python3
"""Evaluate personalization labels against an OpenAI-compatible HTTP API.

The script:
1. reads a system prompt from prompt.txt,
2. reads query objects from queries.jsonl,
3. sends each query to an OpenAI-style chat completions endpoint,
4. predicts personalization by checking whether "personalize" appears
   in the model response, and
5. reports false positives and false negatives per category.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any
from urllib import error, request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run prompts from a JSONL file against an OpenAI-compatible HTTP API "
            "and report false positives and false negatives by category."
        )
    )
    parser.add_argument(
        "--prompt-file",
        default="prompt.txt",
        help="Path to the system prompt text file.",
    )
    parser.add_argument(
        "--queries-file",
        default="queries.jsonl",
        help="Path to the JSONL file containing prompt objects.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--endpoint",
        default="/v1/chat/completions",
        help="Endpoint path, or a full URL, for chat completions.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to send to the API.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional bearer token for the API.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature to send to the API.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens to request from the API.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional limit on how many query rows to process.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional path to write the JSON report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress to stderr while requests are running.",
    )
    args = parser.parse_args()

    if args.max_queries is not None and args.max_queries < 1:
        parser.error("--max-queries must be a positive integer.")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be a positive integer.")
    if args.timeout <= 0:
        parser.error("--timeout must be positive.")

    return args


def read_system_prompt(path: str) -> str:
    prompt = Path(path).read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"System prompt file is empty: {path}")
    return prompt


def read_queries(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} in {path}."
                )
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def resolve_url(base_url: str, endpoint: str) -> str:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def coerce_personalization_flag(row: dict[str, Any]) -> int:
    if "personalization" in row:
        value = row["personalization"]
    elif "pesonalization" in row:
        value = row["pesonalization"]
    else:
        raise KeyError(
            "Query row is missing both 'personalization' and 'pesonalization'."
        )

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return 1
    if normalized in {"0", "false", "no"}:
        return 0
    raise ValueError(f"Unsupported personalization value: {value!r}")


def flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    pieces.append(text)
        return "".join(pieces)
    return str(content)


def extract_response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("API response did not contain a non-empty 'choices' list.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("First API choice was not a JSON object.")

    message = first_choice.get("message")
    if isinstance(message, dict) and "content" in message:
        return flatten_content(message["content"])

    if "text" in first_choice:
        return flatten_content(first_choice["text"])

    raise ValueError("Could not extract response text from API payload.")


def call_chat_completion(
    url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float,
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(url, data=body, headers=headers, method="POST")

    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} while calling {url}: {details}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc

    payload = json.loads(response_body)
    return extract_response_text(payload)


def build_report(
    rows: list[dict[str, Any]],
    url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    timeout: float,
    temperature: float,
    max_tokens: int,
    verbose: bool,
) -> dict[str, Any]:
    per_category: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "evaluated": 0,
            "expected_personalization_1": 0,
            "expected_personalization_0": 0,
            "predicted_personalize_true": 0,
            "predicted_personalize_false": 0,
            "false_positives": [],
            "false_negatives": [],
        }
    )
    totals = Counter()
    errors: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        row_id = row.get("id")
        category = str(row.get("category", "unknown"))
        query_text = row.get("query")
        if not isinstance(query_text, str) or not query_text.strip():
            errors.append(
                {
                    "id": row_id,
                    "category": category,
                    "line_number": row.get("_line_number"),
                    "error": "Missing or empty 'query' field.",
                }
            )
            continue

        try:
            expected = coerce_personalization_flag(row)
            response_text = call_chat_completion(
                url=url,
                model=model,
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=query_text,
                timeout=timeout,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "id": row_id,
                    "category": category,
                    "line_number": row.get("_line_number"),
                    "error": str(exc),
                }
            )
            continue

        predicted = "personalize" in response_text.lower()
        category_report = per_category[category]
        category_report["evaluated"] += 1
        category_report[f"expected_personalization_{expected}"] += 1
        category_report[f"predicted_personalize_{str(predicted).lower()}"] += 1

        totals["evaluated"] += 1
        totals[f"expected_{expected}"] += 1
        totals[f"predicted_{str(predicted).lower()}"] += 1

        record = {
            "id": row_id,
            "line_number": row.get("_line_number"),
            "query": query_text,
            "response": response_text,
            "expected_personalization": expected,
            "predicted_personalize": predicted,
        }

        if predicted and expected == 0:
            category_report["false_positives"].append(record)
            totals["false_positives"] += 1
        elif not predicted and expected == 1:
            category_report["false_negatives"].append(record)
            totals["false_negatives"] += 1

        if verbose:
            print(
                (
                    f"[{index}/{len(rows)}] id={row_id} category={category} "
                    f"expected={expected} predicted={int(predicted)}"
                ),
                file=sys.stderr,
            )

    return {
        "meta": {
            "api_url": url,
            "model": model,
            "query_count": len(rows),
            "system_prompt_chars": len(system_prompt),
        },
        "totals": {
            "evaluated": totals["evaluated"],
            "expected_personalization_1": totals["expected_1"],
            "expected_personalization_0": totals["expected_0"],
            "predicted_personalize_true": totals["predicted_true"],
            "predicted_personalize_false": totals["predicted_false"],
            "false_positives": totals["false_positives"],
            "false_negatives": totals["false_negatives"],
            "errors": len(errors),
        },
        "by_category": dict(per_category),
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    system_prompt = read_system_prompt(args.prompt_file)
    queries = read_queries(args.queries_file)
    if args.max_queries is not None:
        queries = queries[: args.max_queries]

    report = build_report(
        rows=queries,
        url=resolve_url(args.base_url, args.endpoint),
        model=args.model,
        api_key=args.api_key,
        system_prompt=system_prompt,
        timeout=args.timeout,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
    )

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
