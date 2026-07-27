"""Small, dependency-light helpers shared by the tutorial scripts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
WORD_RE = re.compile(r"[\w'-]+", re.UNICODE)


def read_document(path: Path) -> str:
    """Extract searchable text from Markdown, plain text, or a text-based PDF."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            import pymupdf
        except ImportError as exc:
            raise RuntimeError(
                "PDF support needs PyMuPDF. Run: pip install -r requirements.txt"
            ) from exc

        sections: list[str] = []
        with pymupdf.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text", sort=True).strip()
                if text:
                    sections.append(f"[PDF page {page_number}]\n{text}")
        if not sections:
            raise ValueError(
                f"No selectable text found in {path}. It may need OCR before ingestion."
            )
        return "\n\n".join(sections)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_words: int = 260, overlap_words: int = 40) -> list[str]:
    """Split text into overlapping word windows that are small enough for a prompt."""
    if max_words < 20:
        raise ValueError("max_words must be at least 20")
    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("overlap_words must be between 0 and max_words - 1")

    words = normalize_text(text).split()
    if not words:
        return []

    step = max_words - overlap_words
    return [" ".join(words[start : start + max_words]) for start in range(0, len(words), step)]


def document_chunks(input_dir: Path, max_words: int, overlap_words: int) -> list[dict[str, Any]]:
    """Read every supported file recursively and return provenance-preserving chunks."""
    records: list[dict[str, Any]] = []
    paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    for path in paths:
        source = path.relative_to(input_dir).as_posix()
        for index, text in enumerate(chunk_text(read_document(path), max_words, overlap_words)):
            digest = hashlib.sha256(f"{source}\0{index}\0{text}".encode()).hexdigest()[:16]
            records.append(
                {
                    "chunk_id": digest,
                    "source": source,
                    "chunk_index": index,
                    "text": text,
                    "word_count": len(text.split()),
                }
            )
    return records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} on line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {path} on line {line_number}")
            records.append(value)
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def retrieve(query: str, chunks: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    """Rank a small local corpus with transparent TF-IDF-style lexical scoring."""
    if not chunks:
        return []
    query_terms = set(words(query))
    if not query_terms:
        return []

    tokenized = [words(str(chunk["text"])) for chunk in chunks]
    document_frequency = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    total = len(chunks)
    ranked: list[tuple[float, dict[str, Any]]] = []
    query_phrase = " ".join(words(query))
    for chunk, tokens in zip(chunks, tokenized):
        counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            if term in counts:
                inverse_document_frequency = math.log((total + 1) / (document_frequency[term] + 1)) + 1
                score += (1 + math.log(counts[term])) * inverse_document_frequency
        score /= math.sqrt(max(len(tokens), 1))
        if query_phrase and query_phrase in " ".join(tokens):
            score += 1.0
        if score > 0:
            ranked.append((score, chunk))

    ranked.sort(key=lambda item: (-item[0], str(item[1]["source"]), int(item[1]["chunk_index"])))
    return [chunk for _, chunk in ranked[:top_k]]


def grounded_prompt(question: str, matches: list[dict[str, Any]]) -> str:
    if matches:
        references = "\n\n".join(
            f"[{index}] Source: {match['source']} (chunk {match['chunk_index']})\n{match['text']}"
            for index, match in enumerate(matches, start=1)
        )
    else:
        references = "No relevant reference excerpt was found."
    return (
        "Answer the question using only the reference excerpts. "
        "If the answer is not in them, say you do not know. Be clear and concise.\n\n"
        f"{references}\n\nQuestion: {question.strip()}"
    )


def load_questions(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
