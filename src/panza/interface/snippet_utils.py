from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ALLOWED_EXTS = {".txt", ".doc", ".docx"}
FILE_SELECTOR_EXTS = {".txt"}

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
    """Build a whole-term, case-insensitive regex for a sensitive keyword phrase."""
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
    """Return text and Word files under root while pruning ignored folders."""
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


def iter_file_selector_documents(root: Path) -> List[Path]:
    """Return selectable documents under root while pruning ignored folders."""
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
            if path.suffix.lower() in FILE_SELECTOR_EXTS:
                files.append(path)
    return files


def scan_file_selector_root(root: Path) -> Tuple[List[Path], Dict[str, Dict[str, Any]]]:
    """Scan root once and return selectable documents plus all visited directories."""
    files: List[Path] = []
    directories: Dict[str, Dict[str, Any]] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        directory = Path(dirpath)
        dpath = str(directory).lower()
        if any(key in dpath for key in SKIP_PATH_KEYWORDS):
            dirnames[:] = []
            continue

        directories[str(directory)] = {
            "path": str(directory),
            "name": directory.name,
            "parent": str(directory.parent) if directory != root else None,
            "scanned_at": utc_now_text(),
        }

        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for filename in filenames:
            if filename.startswith("~$"):
                continue
            path = directory / filename
            if path.suffix.lower() in FILE_SELECTOR_EXTS:
                files.append(path)

    return files, directories


def should_skip_by_name_or_path(path: Path) -> Optional[str]:
    """Return a skip reason when a path looks sensitive or unsuitable."""
    lower_path = str(path).lower()
    for key in SENSITIVE_PATH_KEYWORDS:
        if key in lower_path:
            return "sensitive_path"

    stem = path.stem.lower()
    if any(key in stem for key in SKIP_FILE_KEYWORDS):
        return "skip_filename_pattern"

    return None


def read_text_file(path: Path) -> str:
    """Read a plain text file with a small set of common fallback encodings."""
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
    """Convert a Word document to text using macOS textutil."""
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
    """Normalize line endings, tabs, nulls, and repeated spaces in raw text."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u0000", "")
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r"[ \xa0]+", " ", text)
    return text.strip()


def split_into_paragraphs(text: str) -> List[str]:
    """Split normalized text into quality-filtered prose paragraphs."""
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
    """Count word-like alphabetic tokens in text."""
    return len(WORD_RE.findall(text))


def looks_like_prose(text: str) -> bool:
    """Return whether text is long and text-dense enough to be prose."""
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
    """Return whether a paragraph has enough letters and few odd symbols."""
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
    """Return a quality failure reason for a snippet, or None when it passes."""
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
    """Return a sensitivity failure reason for a snippet, or None when it passes."""
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
    """Choose high-scoring, non-overlapping snippet windows from paragraphs."""
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
    """Extract reviewable snippets from one file and return filter statistics."""
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
    """Remove duplicate snippets by normalized snippet text hash."""
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
    """Write snippets, scan report, empty kept files, and initial review state."""
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


def compute_dataset_hash(path: Path) -> str:
    """Return the SHA-256 hash for a dataset file, or an empty string on failure."""
    if not path.exists():
        return ""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def read_preview_file(path: Path) -> str:
    """Read a selected file for UI preview using forgiving text encodings."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="latin-1")
        except Exception as exc:
            return f"Could not read file: {exc}"


def persist_review_state(review_state_path: Path, review_state: Dict[str, Any]) -> None:
    """Write review state JSON, creating its parent directory if needed."""
    review_state_path.parent.mkdir(parents=True, exist_ok=True)
    review_state_path.write_text(
        json.dumps(review_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def group_snippets_by_source(
    snippets: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """Group snippets by source path while preserving first-seen source order."""
    docs_by_source: Dict[str, List[Dict[str, Any]]] = {}
    document_paths: List[str] = []
    for snippet in snippets:
        source_path = str(snippet.get("source_path", "Unknown"))
        if source_path not in docs_by_source:
            docs_by_source[source_path] = []
            document_paths.append(source_path)
        docs_by_source[source_path].append(snippet)
    return docs_by_source, document_paths


def load_snippet_tool_state(
    snippets_path: Path,
    review_state_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, List[Dict[str, Any]]], List[str]]:
    """Load snippets and review state, repairing missing defaults as needed."""
    snippets: List[Dict[str, Any]] = []
    review_state: Dict[str, Any] = {}

    if snippets_path.exists():
        try:
            loaded_snippets = json.loads(snippets_path.read_text(encoding="utf-8"))
            if isinstance(loaded_snippets, list):
                snippets = loaded_snippets
        except Exception:
            snippets = []

    if review_state_path.exists():
        try:
            loaded_state = json.loads(review_state_path.read_text(encoding="utf-8"))
            if isinstance(loaded_state, dict):
                review_state = loaded_state
        except Exception:
            review_state = {}

    if not isinstance(review_state.get("decisions"), dict):
        review_state["decisions"] = {}

    review_state.setdefault("updated_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    review_state.setdefault("dataset_sha256", compute_dataset_hash(snippets_path))
    persist_review_state(review_state_path, review_state)

    docs_by_source, document_paths = group_snippets_by_source(snippets)
    return snippets, review_state, docs_by_source, document_paths


def utc_now_text() -> str:
    """Return the current UTC time in the cache timestamp format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_document_cache(cache_path: Path) -> Dict[str, Any]:
    """Load the document snippet cache, returning an empty cache on failure."""
    if not cache_path.exists():
        return {"documents": {}, "directories": {}}
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {"documents": {}, "directories": {}}
    if not isinstance(cache, dict):
        return {"documents": {}, "directories": {}}
    if not isinstance(cache.get("documents"), dict):
        cache["documents"] = {}
    if not isinstance(cache.get("directories"), dict):
        cache["directories"] = {}
    return cache


def write_document_cache(cache_path: Path, cache: Dict[str, Any]) -> None:
    """Persist the document snippet cache to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def extracted_source_paths(snippets_path: Path) -> set[str]:
    """Return source paths that already have snippets in the review dataset."""
    if not snippets_path.exists():
        return set()
    try:
        snippets = json.loads(snippets_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if not isinstance(snippets, list):
        return set()
    return {str(snippet.get("source_path", "")) for snippet in snippets if snippet.get("source_path")}


def review_counts_by_source(
    snippets_path: Path,
    review_state_path: Optional[Path],
) -> Dict[str, Dict[str, int]]:
    """Return extracted, kept, and deleted snippet counts grouped by source path."""
    if not snippets_path.exists():
        return {}
    try:
        snippets = json.loads(snippets_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(snippets, list):
        return {}

    decisions: Dict[str, Any] = {}
    if review_state_path is not None and review_state_path.exists():
        try:
            review_state = json.loads(review_state_path.read_text(encoding="utf-8"))
            if isinstance(review_state, dict) and isinstance(review_state.get("decisions"), dict):
                decisions = review_state["decisions"]
        except Exception:
            decisions = {}

    counts_by_source: Dict[str, Dict[str, int]] = {}
    for snippet in snippets:
        if not isinstance(snippet, dict):
            continue
        source_path = str(snippet.get("source_path", ""))
        if not source_path:
            continue
        counts = counts_by_source.setdefault(
            source_path,
            {
                "snippets_extracted_count": 0,
                "snippets_kept_count": 0,
                "snippets_deleted_count": 0,
            },
        )
        counts["snippets_extracted_count"] += 1
        decision = decisions.get(str(snippet.get("id", "")))
        if decision == "keep":
            counts["snippets_kept_count"] += 1
        elif decision == "delete":
            counts["snippets_deleted_count"] += 1

    return counts_by_source


def empty_review_counts() -> Dict[str, int]:
    """Return an empty per-document review count record."""
    return {
        "snippets_extracted_count": 0,
        "snippets_kept_count": 0,
        "snippets_deleted_count": 0,
    }


def apply_document_review_counts(
    entry: Dict[str, Any],
    counts_by_source: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    """Copy per-document review counts into a cache entry."""
    counts = counts_by_source.get(str(entry.get("path", "")), empty_review_counts())
    entry["snippets_extracted"] = counts["snippets_extracted_count"] > 0
    entry["snippets_extracted_count"] = counts["snippets_extracted_count"]
    entry["snippets_kept_count"] = counts["snippets_kept_count"]
    entry["snippets_deleted_count"] = counts["snippets_deleted_count"]
    return entry


def document_metadata(path: Path) -> Optional[Dict[str, int]]:
    """Return cache metadata used to detect whether a document changed."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def document_cache_entry_is_fresh(entry: Dict[str, Any], metadata: Dict[str, int]) -> bool:
    """Return whether a cache entry matches the current document metadata."""
    return (
        entry.get("mtime_ns") == metadata["mtime_ns"]
        and entry.get("size") == metadata["size"]
    )


def analyze_document_for_cache(
    path: Path,
    counts_by_source: Dict[str, Dict[str, int]],
    snippets_per_file: int = 200,
) -> Optional[Dict[str, Any]]:
    """Analyze one document and return its cache entry."""
    metadata = document_metadata(path)
    if metadata is None:
        return None

    snippets, stats = process_file(path, snippets_per_file=snippets_per_file)
    entry = {
        "path": str(path),
        "mtime_ns": metadata["mtime_ns"],
        "size": metadata["size"],
        "scanned_at": utc_now_text(),
        "has_usable_snippet": bool(snippets),
        "snippet_count": len(snippets),
        "stats": dict(sorted(stats.items())),
    }
    return apply_document_review_counts(entry, counts_by_source)


def refresh_document_cache(
    root: Path,
    cache_path: Path,
    snippets_path: Path,
    review_state_path: Optional[Path] = None,
    snippets_per_file: int = 200,
) -> Dict[str, Any]:
    """Refresh stale document cache entries under root and persist the cache."""
    cache = load_document_cache(cache_path)
    documents = cache.setdefault("documents", {})
    selector_documents, directories = scan_file_selector_root(root)
    cache["directories"] = directories
    counts_by_source = review_counts_by_source(snippets_path, review_state_path)
    current_paths: set[str] = set()

    for path in sorted(selector_documents):
        path_key = str(path)
        current_paths.add(path_key)
        metadata = document_metadata(path)
        if metadata is None:
            documents.pop(path_key, None)
            continue

        entry = documents.get(path_key)
        if isinstance(entry, dict) and document_cache_entry_is_fresh(entry, metadata):
            entry["path"] = path_key
            apply_document_review_counts(entry, counts_by_source)
            documents[path_key] = entry
            continue

        analyzed_entry = analyze_document_for_cache(
            path,
            counts_by_source,
            snippets_per_file=snippets_per_file,
        )
        if analyzed_entry is not None:
            documents[path_key] = analyzed_entry

    for cached_path in list(documents):
        if cached_path not in current_paths:
            del documents[cached_path]

    cache["root"] = str(root)
    cache["refreshed_at"] = utc_now_text()
    write_document_cache(cache_path, cache)
    return cache


def build_cached_snippet_file_tree(root: Path, cache: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a useful tree from cached directories and usable cached documents."""
    documents = cache.get("documents", {})
    directories = cache.get("directories", {})
    if not isinstance(documents, dict) or not isinstance(directories, dict):
        return None

    usable_paths = [
        Path(path)
        for path, entry in documents.items()
        if isinstance(entry, dict) and entry.get("has_usable_snippet")
    ]
    usable_paths.sort(key=lambda p: str(p).lower())

    useful_directory_paths: set[str] = {str(root)}
    for path in usable_paths:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        for parent in [path.parent, *path.parents]:
            try:
                parent.relative_to(root)
            except ValueError:
                break
            useful_directory_paths.add(str(parent))
            if parent == root:
                break

    root_node: Dict[str, Any] = {"path": root, "type": "directory", "children": []}
    nodes_by_path: Dict[str, Dict[str, Any]] = {str(root): root_node}
    cached_directory_paths = [
        Path(path)
        for path in directories
        if path in useful_directory_paths and path != str(root)
    ]
    cached_directory_paths.sort(key=lambda p: (len(p.parts), str(p).lower()))

    for directory in cached_directory_paths:
        parent_node = nodes_by_path.get(str(directory.parent))
        if parent_node is None:
            continue
        node = {"path": directory, "type": "directory", "children": []}
        parent_node["children"].append(node)
        nodes_by_path[str(directory)] = node

    for path in usable_paths:
        parent_node = nodes_by_path.get(str(path.parent))
        if parent_node is not None:
            parent_node["children"].append(
                {
                    "path": path,
                    "type": "file",
                    "cache_entry": documents.get(str(path), {}),
                }
            )

    sort_cached_tree(root_node)

    if not root_node["children"]:
        return None
    return root_node


def sort_cached_tree(node: Dict[str, Any]) -> None:
    """Sort cached tree children by directory-first display order."""
    children = node.get("children", [])
    children.sort(key=lambda child: (child.get("type") != "directory", child["path"].name.lower()))
    for child in children:
        if child.get("type") == "directory":
            sort_cached_tree(child)


def build_txt_file_tree(path: Path) -> Optional[Dict[str, Any]]:
    """Build a pruned tree containing only directories with .txt descendants."""
    try:
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except Exception:
        return None

    children: List[Dict[str, Any]] = []
    for entry in entries:
        if entry.is_dir():
            subtree = build_txt_file_tree(entry)
            if subtree is not None:
                children.append(subtree)
        elif entry.is_file() and entry.suffix.lower() == ".txt":
            children.append({"path": entry, "type": "file"})

    if not children:
        return None
    return {"path": path, "type": "directory", "children": children}


def next_snippet_id_number(snippets: List[Dict[str, Any]]) -> int:
    """Return the next numeric suffix for snippet IDs of the form s000001."""
    max_id = 0
    for snippet in snippets:
        snippet_id = str(snippet.get("id", ""))
        match = re.fullmatch(r"s(\d+)", snippet_id)
        if match:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def assign_missing_snippet_ids(snippets: List[Dict[str, Any]]) -> None:
    """Assign stable sequential IDs to snippets that do not already have one."""
    next_id = next_snippet_id_number(snippets)
    for snippet in snippets:
        if snippet.get("id"):
            continue
        snippet["id"] = f"s{next_id:06d}"
        next_id += 1


def split_file_into_review_dataset(
    selected_file_path: Path,
    existing_snippets: List[Dict[str, Any]],
    review_state: Dict[str, Any],
    snippets_path: Path,
    review_state_path: Path,
    snippets_per_file: int = 200,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int, int, Counter]:
    """Split one selected file and merge new snippets into the review dataset."""
    new_snippets, stats = process_file(selected_file_path, snippets_per_file=snippets_per_file)
    if not new_snippets:
        return existing_snippets, review_state, 0, 0, stats

    before_count = len(existing_snippets)
    combined = dedupe_snippets([*existing_snippets, *new_snippets])
    assign_missing_snippet_ids(combined)

    snippets_path.parent.mkdir(parents=True, exist_ok=True)
    snippets_path.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    review_state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    review_state["dataset_sha256"] = compute_dataset_hash(snippets_path)
    persist_review_state(review_state_path, review_state)

    return combined, review_state, len(new_snippets), len(combined) - before_count, stats


def export_kept_snippets(
    review_snippets: List[Dict[str, Any]],
    review_state: Dict[str, Any],
    kept_jsonl_path: Path,
    kept_json_path: Path,
) -> None:
    """Write kept review snippets to JSONL and JSON export files."""
    decisions = review_state.get("decisions", {})
    kept = [
        snippet
        for snippet in review_snippets
        if decisions.get(str(snippet.get("id", ""))) == "keep"
    ]
    kept_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with kept_jsonl_path.open("w", encoding="utf-8") as fh:
        for row in kept:
            item = {
                "id": row.get("id"),
                "source_path": row.get("source_path"),
                "snippet_text": row.get("snippet_text"),
                "snippet_word_count": row.get("snippet_word_count"),
                "paragraph_count": row.get("paragraph_count"),
            }
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    kept_json_path.write_text(
        json.dumps(kept, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def apply_review_decision(
    review_state: Dict[str, Any],
    snippet_id: str,
    decision: str,
) -> None:
    """Apply one keep/delete decision to review state and update its timestamp."""
    review_state.setdefault("decisions", {})[snippet_id] = decision
    review_state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def apply_review_decision_bulk(
    review_state: Dict[str, Any],
    snippets: List[Dict[str, Any]],
    decision: str,
) -> None:
    """Apply one keep/delete decision to all snippets in a document group."""
    decisions = review_state.setdefault("decisions", {})
    for snippet in snippets:
        snippet_id = str(snippet.get("id", ""))
        if snippet_id:
            decisions[snippet_id] = decision
    review_state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def review_counts_text(review_snippets: List[Dict[str, Any]], review_state: Dict[str, Any]) -> str:
    """Return a compact review progress summary for the UI."""
    decisions = review_state.get("decisions", {})
    visible_decisions = [
        decisions.get(str(snippet.get("id", "")))
        for snippet in review_snippets
    ]
    kept = sum(1 for value in visible_decisions if value == "keep")
    deleted = sum(1 for value in visible_decisions if value == "delete")
    total = len(review_snippets)
    pending = max(total - kept - deleted, 0)
    return f"Total: {total} | Kept: {kept} | Deleted: {deleted} | Pending: {pending}"
