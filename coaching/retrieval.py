"""Lightweight retrieval module for coaching advice.

This is a simple local lexical retriever to support RAG-style grounding without
external services. It can be swapped with vector retrieval later.
"""

from __future__ import annotations

import math
import pathlib
import re
from collections import Counter


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class LocalKnowledgeRetriever:
    def __init__(self, knowledge_path: str) -> None:
        self.knowledge_path = pathlib.Path(knowledge_path)
        self.passages = self._load_passages()
        self.doc_freq = self._compute_doc_freq()
        self.num_docs = len(self.passages)

    def _load_passages(self) -> list[str]:
        raw = self.knowledge_path.read_text(encoding="utf-8")
        chunks = [chunk.strip() for chunk in raw.split("\n\n") if chunk.strip()]
        return chunks

    def _compute_doc_freq(self) -> Counter[str]:
        counter: Counter[str] = Counter()
        for passage in self.passages:
            for token in set(tokenize(passage)):
                counter[token] += 1
        return counter

    def _idf(self, token: str) -> float:
        # Smoothed IDF to avoid divide-by-zero and reduce rare-token spikes.
        return math.log((1 + self.num_docs) / (1 + self.doc_freq.get(token, 0))) + 1.0

    def score(self, query: str, passage: str) -> float:
        query_tokens = tokenize(query)
        if not query_tokens:
            return 0.0
        passage_tf = Counter(tokenize(passage))
        score_value = 0.0
        for token in query_tokens:
            score_value += passage_tf[token] * self._idf(token)
        return score_value

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[float, str]]:
        scored = [(self.score(query, passage), passage) for passage in self.passages]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]


if __name__ == "__main__":
    kb_path = pathlib.Path(__file__).resolve().parent / "knowledge" / "tips.md"
    retriever = LocalKnowledgeRetriever(str(kb_path))
    demo_query = "I keep dying while peeking and missing first shots"
    print(f"Query: {demo_query}")
    for score_value, text in retriever.retrieve(demo_query, top_k=3):
        print(f"\n[score={score_value:.2f}] {text}")
