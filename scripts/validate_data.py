#!/usr/bin/env python3
"""Validate the tutorial's SFT and DPO JSONL files before expensive training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import read_jsonl


def message_list(value: Any, label: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    for message in value:
        if not isinstance(message, dict):
            raise ValueError(f"Each {label} message must be an object")
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"Invalid role in {label}: {message.get('role')!r}")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"Each {label} message needs non-empty content")
    return value


def conversation(value: Any, label: str, *, final_role: str) -> list[dict[str, str]]:
    messages = message_list(value, label)
    roles = [message["role"] for message in messages]
    if "system" in roles[1:] or roles.count("system") > 1:
        raise ValueError(f"{label} may contain one system message, and only at the beginning")

    dialogue_roles = roles[1:] if roles[0] == "system" else roles
    if not dialogue_roles or dialogue_roles[0] != "user":
        raise ValueError(f"{label} dialogue must begin with a user message")
    expected = ["user" if index % 2 == 0 else "assistant" for index in range(len(dialogue_roles))]
    if dialogue_roles != expected:
        raise ValueError(f"{label} roles must alternate user and assistant")
    if dialogue_roles[-1] != final_role:
        raise ValueError(f"{label} must end with {final_role}")
    return messages


def assistant_completion(value: Any, label: str) -> list[dict[str, str]]:
    messages = message_list(value, label)
    if len(messages) != 1 or messages[0]["role"] != "assistant":
        raise ValueError(f"{label} must contain exactly one assistant response")
    return messages


def validate_sft(path: Path) -> int:
    records = read_jsonl(path)
    for number, record in enumerate(records, start=1):
        try:
            conversation(record.get("messages"), "messages", final_role="assistant")
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
            conversation(record.get("prompt"), "prompt", final_role="user")
            assistant_completion(record.get("chosen"), "chosen")
            assistant_completion(record.get("rejected"), "rejected")
            if record["chosen"] == record["rejected"]:
                raise ValueError("chosen and rejected answers must differ")
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object when provided")
            if metadata.get("demo"):
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
