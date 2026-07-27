#!/usr/bin/env python3
"""Ask the aligned model questions with fresh context retrieved from local documents."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import grounded_prompt, read_jsonl, retrieve
from modeling import DEFAULT_MODEL, generate_answer, load_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*")
    parser.add_argument("--adapter", type=Path, default=Path("outputs/dpo_adapter"))
    parser.add_argument("--base-only", action="store_true", help="Use the untouched base model.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--use-4bit", action="store_true")
    return parser.parse_args()


def answer_question(question: str, model, tokenizer, chunks, args) -> None:
    matches = retrieve(question, chunks, top_k=args.top_k)
    prompt = grounded_prompt(question, matches)
    answer = generate_answer(
        model,
        tokenizer,
        [{"role": "user", "content": prompt}],
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )
    print(f"\n{answer}\n")
    if matches:
        print("Sources used:")
        for match in matches:
            print(f"- {match['source']} (chunk {match['chunk_index']})")
    else:
        print("Sources used: none")


def main() -> None:
    args = parse_args()
    chunks = read_jsonl(args.chunks)
    adapter = None if args.base_only else args.adapter
    model, tokenizer = load_for_inference(
        model_id=args.model,
        adapter=adapter,
        use_4bit=args.use_4bit,
    )

    initial_question = " ".join(args.question).strip()
    if initial_question:
        answer_question(initial_question, model, tokenizer, chunks, args)
        return

    print("Ask a question about your documents. Enter q to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if question:
            answer_question(question, model, tokenizer, chunks, args)


if __name__ == "__main__":
    main()
