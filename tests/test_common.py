from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from common import chunk_text, retrieve


class ChunkTextTests(unittest.TestCase):
    def test_exact_window_does_not_create_overlap_only_chunk(self) -> None:
        text = " ".join(f"word{index}" for index in range(260))

        chunks = chunk_text(text, max_words=260, overlap_words=40)

        self.assertEqual([len(chunk.split()) for chunk in chunks], [260])

    def test_final_chunk_contains_new_content(self) -> None:
        text = " ".join(f"word{index}" for index in range(261))

        chunks = chunk_text(text, max_words=260, overlap_words=40)

        self.assertEqual([len(chunk.split()) for chunk in chunks], [260, 41])
        self.assertEqual(chunks[0].split()[-40:], chunks[1].split()[:40])
        self.assertEqual(chunks[1].split()[-1], "word260")


class RetrieveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            {
                "source": "refund_policy.md",
                "chunk_index": 0,
                "text": "Refunds require a receipt within fourteen days.",
            },
            {
                "source": "store_information.txt",
                "chunk_index": 0,
                "text": (
                    "The bakery is open Sunday. The delivery fee is five dollars "
                    "for addresses within eight miles."
                ),
            },
        ]

    def test_relevant_query_returns_matching_document_only(self) -> None:
        matches = retrieve("When is the bakery open Sunday?", self.chunks)

        self.assertEqual([match["source"] for match in matches], ["store_information.txt"])

    def test_partial_incidental_overlap_is_not_reported_as_a_source(self) -> None:
        chunks = [
            {
                "source": "refund_policy.md",
                "chunk_index": 0,
                "text": "Contact us after delivery. Refunds arrive in five business days.",
            },
            self.chunks[1],
        ]

        matches = retrieve("What is the delivery fee for five miles?", chunks)

        self.assertEqual([match["source"] for match in matches], ["store_information.txt"])

    def test_unsupported_query_does_not_return_incidental_match(self) -> None:
        matches = retrieve("Does the bakery offer catering?", self.chunks)

        self.assertEqual(matches, [])

    def test_single_content_word_can_match(self) -> None:
        matches = retrieve("Refund?", self.chunks)

        self.assertEqual([match["source"] for match in matches], ["refund_policy.md"])


if __name__ == "__main__":
    unittest.main()
