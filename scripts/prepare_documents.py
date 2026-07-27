#!/usr/bin/env python3
"""Extract and chunk PDF, TXT, and Markdown files into a local JSONL corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import document_chunks, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/source_documents"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--max-words", type=int, default=260)
    parser.add_argument("--overlap-words", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input}")
    records = document_chunks(args.input, args.max_words, args.overlap_words)
    if not records:
        raise SystemExit(f"No supported PDF, TXT, or Markdown files found in {args.input}")
    write_jsonl(args.output, records)
    source_count = len({record["source"] for record in records})
    print(f"Prepared {len(records)} chunks from {source_count} documents.")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
