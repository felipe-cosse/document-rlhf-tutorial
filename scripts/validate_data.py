#!/usr/bin/env python3
"""Validate the tutorial's SFT and DPO JSONL files before expensive training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import read_jsonl


def message_list(value: Any, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    for message in value:
        if not isinstance(message, dict):
            raise ValueError(f"Each {label} message must be an object")
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"Invalid role in {label}: {message.get('role')!r}")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"Each {label} message needs non-empty content")


def validate_sft(path: Path) -> int:
    records = read_jsonl(path)
    for number, record in enumerate(records, start=1):
        try:
            message_list(record.get("messages"), "messages")
            roles = [message["role"] for message in record["messages"]]
            if "user" not in roles or roles[-1] != "assistant":
                raise ValueError("messages must include a user and end with an assistant")
        except ValueError as exc:
            raise ValueError(f"{path}, record {number}: {exc}") from exc
    if not records:
        raise ValueError(f"{path} is empty")
    return len(records)


def validate_preferences(path: Path) -> tuple[int, int]:
    records = read_jsonl(path)
    demo_count = 0
    for number, record in enumerate(records, start=1):
        try:
            message_list(record.get("prompt"), "prompt")
            message_list(record.get("chosen"), "chosen")
            message_list(record.get("rejected"), "rejected")
            if record["chosen"] == record["rejected"]:
                raise ValueError("chosen and rejected answers must differ")
            if record["chosen"][-1]["role"] != "assistant":
                raise ValueError("chosen must end with an assistant answer")
            if record["rejected"][-1]["role"] != "assistant":
                raise ValueError("rejected must end with an assistant answer")
            if record.get("metadata", {}).get("demo"):
                demo_count += 1
        except ValueError as exc:
            raise ValueError(f"{path}, record {number}: {exc}") from exc
    if not records:
        raise ValueError(f"{path} is empty")
    return len(records), demo_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path)
    parser.add_argument("--preferences", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sft and not args.preferences:
        raise SystemExit("Provide --sft, --preferences, or both.")
    if args.sft:
        print(f"SFT data valid: {validate_sft(args.sft)} examples in {args.sft}")
    if args.preferences:
        count, demos = validate_preferences(args.preferences)
        print(f"Preference data valid: {count} pairs in {args.preferences}")
        if demos:
            print(f"Warning: {demos} pair(s) are marked as demonstrations, not human feedback.")


if __name__ == "__main__":
    main()
