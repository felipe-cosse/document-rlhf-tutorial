from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from collect_preferences import randomized_candidates


class ReverseRandom:
    def shuffle(self, values: list[tuple[str, str]]) -> None:
        values.reverse()


class RandomizedCandidatesTests(unittest.TestCase):
    def test_display_order_changes_without_losing_candidate_identity(self) -> None:
        displayed = randomized_candidates("low temperature", "high temperature", rng=ReverseRandom())

        self.assertEqual(
            displayed,
            [("second", "high temperature"), ("first", "low temperature")],
        )


if __name__ == "__main__":
    unittest.main()
