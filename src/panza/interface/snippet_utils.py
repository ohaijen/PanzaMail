from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ALLOWED_EXTS = {".txt", ".doc", ".docx"}

SKIP_PATH_KEYWORDS = {
    "/.git/",
    "/node_modules/",
    "/__pycache__/",
    "/.venv",
    "/venv/",
    "/build/",
    "/dist/",
    "/.idea/",
    "/.vscode/",
    "/library/application support/",
}

SKIP_FILE_KEYWORDS = {
    "readme",
    "todo",
    "changelog",
    "history",
    "license",
    "output",
    "run-",
    "dataset",
    "csv",
    "tsv",
    "bib",
    "references",
    "requirements",
}

SENSITIVE_PATH_KEYWORDS = {
    "/bank",
    "/banks",
    "/finance",
    "/financial",
    "/tax",
    "/taxes",
    "/medical",
    "/health",
    "/insurance",
    "/passport",
    "/visa",
    "/vize",
    "/immigration",
    "/residence",
    "/salary",
    "/payslip",
    "/invoice",
    "/receipts",
    "/statement",
    "/statements",
    "/billing",
}

SENSITIVE_TEXT_KEYWORDS = [
    "iban",
    "swift",
    "routing number",
    "account number",
    "sort code",
    "social security",
    "tax id",
    "passport number",
    "medical record",
    "diagnosis",
    "prescription",
    "credit card",
    "debit card",
    "cvv",
    "pin code",
    "password",
    "wire transfer",
]


def _keyword_regex(keyword: str) -> re.Pattern[str]:
    tokens = [re.escape(tok) for tok in keyword.split()]
    return re.compile(r"\b" + r"\s+".join(tokens) + r"\b", re.IGNORECASE)


SENSITIVE_TEXT_PATTERNS = [(kw, _keyword_regex(kw)) for kw in SENSITIVE_TEXT_KEYWORDS]

SENSITIVE_PATTERNS = {
    "iban_like": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card_like": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone_like": re.compile(r"\+?\d[\d\s().-]{7,}\d"),
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def iter_candidate_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dpath = dirpath.lower()
        if any(key in dpath for key in SKIP_PATH_KEYWORDS):
            dirnames[:] = []
            continue

        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for filename in filenames:
            if filename.startswith("~$"):
                continue
            path = Path(dirpath) / filename
            ext = path.suffix.lower()
            if ext not in ALLOWED_EXTS:
                continue
            files.append(path)
    return files


def should_skip_by_name_or_path(path: Path) -> Optional[str]:
    lower_path = str(path).lower()
    for key in SENSITIVE_PATH_KEYWORDS:
        if key in lower_path:
            return "sensitive_path"

    stem = path.stem.lower()
    if any(key in stem for key in SKIP_FILE_KEYWORDS):
        return "skip_filename_pattern"

    return None


def read_text_file(path: Path) -> str:
    encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    return path.read_bytes().decode("utf-8", errors="ignore")


def read_word_file(path: Path) -> str:
    try:
        proc = subprocess.run(
            ["/usr/bin/textutil", "-convert", "txt", "-stdout", str(path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
        )
    except Exception:
        return ""

    if proc.returncode != 0:
        return ""
    return proc.stdout.decode("utf-8", errors="ignore")


def normalize_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u0000", "")
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r"[ \xa0]+", " ", text)
    return text.strip()


def split_into_paragraphs(text: str) -> List[str]:
    lines = text.split("\n")
    paragraphs: List[str] = []
    current: List[str] = []

    for line in lines:
        stripped = re.sub(r"\s+", " ", line).strip()
        if not stripped:
            if current:
                para = " ".join(current)
                para = re.sub(r"(\w)-\s+(\w)", r"\1\2", para)
                para = re.sub(r"\s+", " ", para).strip()
                if para:
                    paragraphs.append(para)
                current = []
            continue

        current.append(stripped)

    if current:
        para = " ".join(current)
        para = re.sub(r"(\w)-\s+(\w)", r"\1\2", para)
        para = re.sub(r"\s+", " ", para).strip()
        if para:
            paragraphs.append(para)

    return [p for p in paragraphs if word_count(p) >= 25 and paragraph_quality_ok(p)]


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def looks_like_prose(text: str) -> bool:
    text = text.strip()
    if len(text) < 800:
        return False

    words = word_count(text)
    if words < 120:
        return False

    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    chars = len(text)
    if letters / max(chars, 1) < 0.45:
        return False
    if digits / max(letters, 1) > 0.18:
        return False

    non_empty_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not non_empty_lines:
        return False

    short_lines = sum(1 for ln in non_empty_lines if len(ln) < 20)
    if short_lines / max(len(non_empty_lines), 1) > 0.8:
        return False

    return True


def paragraph_quality_ok(paragraph: str) -> bool:
    if not paragraph:
        return False

    letters = sum(c.isalpha() for c in paragraph)
    chars = len(paragraph)
    if letters / max(chars, 1) < 0.5:
        return False

    weird = sum(
        1
        for c in paragraph
        if not (c.isalnum() or c.isspace() or c in ".,;:!?()[]{}'\"-_/&%")
    )
    if weird / max(chars, 1) > 0.12:
        return False

    return True


def snippet_quality_issue(snippet: str) -> Optional[str]:
    controls = sum(1 for c in snippet if ord(c) < 32 and c not in "\n\t")
    if controls > 0:
        return "control_chars"

    chars = len(snippet)
    letters = sum(c.isalpha() for c in snippet)
    if letters / max(chars, 1) < 0.55:
        return "low_letter_ratio"

    weird = sum(
        1
        for c in snippet
        if not (c.isalnum() or c.isspace() or c in ".,;:!?()[]{}'\"-_/&%")
    )
    if weird / max(chars, 1) > 0.08:
        return "high_symbol_ratio"

    sentence_marks = len(re.findall(r"[.!?]", snippet))
    if sentence_marks < 2:
        return "too_few_sentences"

    return None


def snippet_sensitive(snippet: str) -> Optional[str]:
    low = snippet.lower()
    for kw, pattern in SENSITIVE_TEXT_PATTERNS:
        if pattern.search(snippet):
            return f"keyword:{kw}"

    for label, regex in SENSITIVE_PATTERNS.items():
        if regex.search(snippet):
            if label in {"email", "phone_like"}:
                if any(k in low for k in ("contact", "address", "phone", "email", "fax")):
                    return f"pattern:{label}"
                continue
            return f"pattern:{label}"

    digits = sum(c.isdigit() for c in snippet)
    if digits > 70:
        return "too_many_digits"

    return None


def candidate_snippets(paragraphs: List[str], limit: int) -> List[str]:
    candidates: List[Tuple[float, str, Tuple[int, int]]] = []
    n = len(paragraphs)

    for i in range(n):
        for length in (1, 2, 3):
            j = i + length
            if j > n:
                break
            chunk = "\n\n".join(paragraphs[i:j])
            words = word_count(chunk)
            if words < 90 or words > 380:
                continue

            avg_sentence_len = words / max(len(re.findall(r"[.!?]", chunk)), 1)
            score = 0.0
            score += 1.0 if length == 2 else 0.8 if length == 3 else 0.6
            score += min(avg_sentence_len / 22.0, 1.2)
            score += min(words / 220.0, 1.0)
            candidates.append((score, chunk, (i, j)))

    candidates.sort(key=lambda x: x[0], reverse=True)

    chosen: List[str] = []
    used_ranges: List[Tuple[int, int]] = []

    for _, chunk, span in candidates:
        overlaps = False
        for s, e in used_ranges:
            if not (span[1] <= s or span[0] >= e):
                overlaps = True
                break
        if overlaps:
            continue

        chosen.append(chunk)
        used_ranges.append(span)
        if len(chosen) >= limit:
            break

    return chosen


def process_file(path: Path, snippets_per_file: int) -> Tuple[List[Dict[str, object]], Counter]:
    stats = Counter()
    skip_reason = should_skip_by_name_or_path(path)
    if skip_reason:
        stats[skip_reason] += 1
        return [], stats

    try:
        size = path.stat().st_size
    except OSError:
        stats["stat_error"] += 1
        return [], stats

    if size < 512:
        stats["too_small"] += 1
        return [], stats
    if size > 15 * 1024 * 1024:
        stats["too_large"] += 1
        return [], stats

    ext = path.suffix.lower()
    if ext == ".txt":
        raw = read_text_file(path)
    else:
        raw = read_word_file(path)

    if not raw.strip():
        stats["read_empty"] += 1
        return [], stats

    text = normalize_text(raw)
    if not looks_like_prose(text):
        stats["not_prose"] += 1
        return [], stats

    paragraphs = split_into_paragraphs(text)
    if len(paragraphs) < 1:
        stats["no_paragraphs"] += 1
        return [], stats

    snippets = candidate_snippets(paragraphs, snippets_per_file)
    if not snippets:
        stats["no_snippet_windows"] += 1
        return [], stats

    out: List[Dict[str, object]] = []
    for snippet in snippets:
        quality_issue = snippet_quality_issue(snippet)
        if quality_issue:
            stats[f"quality_{quality_issue}"] += 1
            continue

        sensitive_reason = snippet_sensitive(snippet)
        if sensitive_reason:
            stats[f"sensitive_{sensitive_reason}"] += 1
            continue

        out.append(
            {
                "source_path": str(path),
                "source_ext": ext,
                "snippet_text": snippet,
                "snippet_word_count": word_count(snippet),
                "paragraph_count": snippet.count("\n\n") + 1,
            }
        )

    if not out:
        stats["all_snippets_filtered"] += 1
    else:
        stats["files_with_snippets"] += 1

    return out, stats


def dedupe_snippets(snippets: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    seen: set[str] = set()
    unique: List[Dict[str, object]] = []

    for item in snippets:
        normalized = re.sub(r"\s+", " ", str(item["snippet_text"]).strip().lower())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        item["snippet_sha256"] = digest
        unique.append(item)

    return unique


def write_outputs(output_dir: Path, snippets: List[Dict[str, object]], report: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    snippets_path = output_dir / "snippets.json"
    report_path = output_dir / "scan_report.json"
    review_state_path = output_dir / "review_state.json"
    kept_jsonl_path = output_dir / "kept_snippets.jsonl"
    kept_json_path = output_dir / "kept_snippets.json"

    snippets_path.write_text(json.dumps(snippets, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    state = {
        "dataset_created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_sha256": hashlib.sha256(snippets_path.read_bytes()).hexdigest(),
        "decisions": {},
    }
    review_state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    kept_jsonl_path.write_text("", encoding="utf-8")
    kept_json_path.write_text("[]\n", encoding="utf-8")
