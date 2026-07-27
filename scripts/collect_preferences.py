#!/usr/bin/env python3
"""Generate two grounded answers and ask a human which one is better."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from common import append_jsonl, grounded_prompt, load_questions, read_jsonl, retrieve
from modeling import generate_answer, load_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=Path("outputs/sft_adapter"))
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/questions.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/preferences.jsonl"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--use-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = read_jsonl(args.chunks)
    questions = load_questions(args.questions)
    existing = read_jsonl(args.output) if args.output.exists() else []
    completed = {
        record.get("metadata", {}).get("question")
        for record in existing
        if record.get("metadata", {}).get("question")
    }

    model, tokenizer = load_for_inference(adapter=args.adapter, use_4bit=args.use_4bit)
    print("Choose the answer that is more accurate, grounded, helpful, and concise.")
    print("Enter 1, 2, s to skip, or q to quit.\n")
    saved = 0
    for index, question in enumerate(questions, start=1):
        if question in completed:
            continue
        matches = retrieve(question, chunks, top_k=args.top_k)
        prompt_text = grounded_prompt(question, matches)
        prompt = [{"role": "user", "content": prompt_text}]
        first = generate_answer(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.65,
            seed=1000 + index * 2,
        )
        second = generate_answer(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.9,
            seed=1001 + index * 2,
        )
        if first == second:
            second = generate_answer(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=1.1,
                seed=2001 + index,
            )

        print(f"\n=== Question {index}/{len(questions)} ===\n{question}")
        print(f"\n--- Answer 1 ---\n{first}")
        print(f"\n--- Answer 2 ---\n{second}")
        while True:
            choice = input("\nBetter answer [1/2/s/q]: ").strip().lower()
            if choice in {"1", "2", "s", "q"}:
                break
        if choice == "q":
            break
        if choice == "s":
            continue
        chosen, rejected = (first, second) if choice == "1" else (second, first)
        append_jsonl(
            args.output,
            {
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "metadata": {
                    "question": question,
                    "source_paths": [match["source"] for match in matches],
                    "annotated_at": datetime.now(timezone.utc).isoformat(),
                    "annotator": "human",
                },
            },
        )
        saved += 1
        print("Preference saved.")
    print(f"\nAdded {saved} human preference pair(s) to {args.output}")


if __name__ == "__main__":
    main()
