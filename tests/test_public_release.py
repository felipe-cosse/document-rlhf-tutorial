from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseTests(unittest.TestCase):
    def is_ignored(self, relative_path: str) -> bool:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", relative_path],
            cwd=ROOT,
            check=False,
        )
        return result.returncode == 0

    def test_new_source_documents_are_ignored(self) -> None:
        self.assertTrue(self.is_ignored("data/source_documents/private-document.pdf"))

    def test_fictional_examples_remain_trackable(self) -> None:
        self.assertFalse(self.is_ignored("data/source_documents/refund_policy.md"))
        self.assertFalse(self.is_ignored("data/source_documents/store_information.txt"))


if __name__ == "__main__":
    unittest.main()
