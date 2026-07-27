from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_data import validate_preferences, validate_sft


def write_jsonl(directory: Path, name: str, record: dict[str, object]) -> Path:
    path = directory / name
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


class ValidateDataTests(unittest.TestCase):
    def test_sft_rejects_assistant_first_dialogue(self) -> None:
        record = {
            "messages": [
                {"role": "assistant", "content": "First"},
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Last"},
            ]
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = write_jsonl(Path(temporary_directory), "sft.jsonl", record)

            with self.assertRaisesRegex(ValueError, "begin with a user"):
                validate_sft(path)

    def test_preferences_require_user_ended_prompt(self) -> None:
        record = {
            "prompt": [{"role": "assistant", "content": "Not a user prompt"}],
            "chosen": [{"role": "assistant", "content": "A"}],
            "rejected": [{"role": "assistant", "content": "B"}],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = write_jsonl(Path(temporary_directory), "preferences.jsonl", record)

            with self.assertRaisesRegex(ValueError, "begin with a user"):
                validate_preferences(path)

    def test_preferences_require_single_assistant_completion(self) -> None:
        record = {
            "prompt": [{"role": "user", "content": "Question"}],
            "chosen": [
                {"role": "assistant", "content": "A"},
                {"role": "assistant", "content": "Extra"},
            ],
            "rejected": [{"role": "assistant", "content": "B"}],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = write_jsonl(Path(temporary_directory), "preferences.jsonl", record)

            with self.assertRaisesRegex(ValueError, "exactly one assistant response"):
                validate_preferences(path)

    def test_preferences_reject_non_object_metadata(self) -> None:
        record = {
            "prompt": [{"role": "user", "content": "Question"}],
            "chosen": [{"role": "assistant", "content": "A"}],
            "rejected": [{"role": "assistant", "content": "B"}],
            "metadata": "not-an-object",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = write_jsonl(Path(temporary_directory), "preferences.jsonl", record)

            with self.assertRaisesRegex(ValueError, "metadata must be an object"):
                validate_preferences(path)


if __name__ == "__main__":
    unittest.main()
