#!/usr/bin/env python3
"""Interactively turn source excerpts into human-reviewed instruction examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import append_jsonl, grounded_prompt, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/sft.custom.jsonl"))
    parser.add_argument("--limit", type=int, default=0, help="Stop after this many chunks; 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = read_jsonl(args.chunks)
    if args.limit > 0:
        chunks = chunks[: args.limit]

    print("For each excerpt, write a realistic question and the best grounded answer.")
    print("Press Enter at the question to skip; enter q to stop.\n")
    written = 0
    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Excerpt {index}/{len(chunks)}: {chunk['source']} ---")
        print(str(chunk["text"])[:2400])
        question = input("\nQuestion (Enter=skip, q=quit): ").strip()
        if question.lower() == "q":
            break
        if not question:
            continue
        answer = input("Ideal answer: ").strip()
        if not answer:
            print("Skipped because the ideal answer was empty.")
            continue

        prompt = grounded_prompt(question, [chunk])
        append_jsonl(
            args.output,
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "source_paths": [chunk["source"]],
            },
        )
        written += 1
        print("Saved.")
    print(f"\nAdded {written} examples to {args.output}")


if __name__ == "__main__":
    main()
