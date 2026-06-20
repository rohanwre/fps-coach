from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coaching.retrieval import LocalKnowledgeRetriever, tokenize


class LocalKnowledgeRetrieverTests(unittest.TestCase):
    def test_tokenize_lowercases_words_and_numbers(self) -> None:
        self.assertEqual(tokenize("First-shot accuracy 101!"), ["first", "shot", "accuracy", "101"])

    def test_retrieve_ranks_matching_passage_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            knowledge_path = Path(tmpdir) / "tips.md"
            knowledge_path.write_text(
                "Aim\n\n"
                "Keep crosshair placement steady before peeking.\n\n"
                "Positioning\n\n"
                "Use cover and reposition after contact.\n",
                encoding="utf-8",
            )
            retriever = LocalKnowledgeRetriever(str(knowledge_path))

            results = retriever.retrieve("crosshair peeking", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertGreater(results[0][0], 0.0)
        self.assertIn("crosshair", results[0][1])


if __name__ == "__main__":
    unittest.main()
