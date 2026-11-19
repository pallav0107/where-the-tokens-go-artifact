"""Deterministic BM25 retrieval over the chunked knowledge base (PREREG section 2)."""
from __future__ import annotations

import math
import re

K1 = 1.5
B = 0.75
_TOKEN_RE = re.compile(r"[a-z0-9]+")

UNKNOWN_DOC_MESSAGE = "Unknown doc_id '{doc_id}'. Use search to discover valid doc_ids."


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Retriever:
    """BM25 (k1=1.5, b=0.75, lowercase alphanumeric tokens) over chunks.

    kb is the object produced by agent.data.build_kb (attrs docs, chunks).
    Ties in score are broken by chunk order.
    """

    def __init__(self, kb):
        self.docs = {d["doc_id"]: d for d in kb.docs}
        self.chunks = list(kb.chunks)
        self.chunk_ids = []  # stable "<doc_id>#<chunk_index>" per chunk (v2 arms cite these)
        seen: dict[str, int] = {}
        for c in self.chunks:
            i = seen.get(c["doc_id"], 0)
            seen[c["doc_id"]] = i + 1
            self.chunk_ids.append(f'{c["doc_id"]}#{i}')
        self._toks = [tokenize(c["text"]) for c in self.chunks]
        self._df: dict[str, int] = {}
        for toks in self._toks:
            for t in set(toks):
                self._df[t] = self._df.get(t, 0) + 1
        self._n = len(self.chunks)
        self._avgdl = (sum(len(t) for t in self._toks) / self._n) if self._n else 0.0

    def _score(self, query_tokens: set[str], idx: int) -> float:
        toks = self._toks[idx]
        freq: dict[str, int] = {}
        for t in toks:
            freq[t] = freq.get(t, 0) + 1
        dl = len(toks)
        score = 0.0
        for t in query_tokens:
            tf = freq.get(t, 0)
            if tf == 0:
                continue
            df = self._df.get(t, 0)
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
            score += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / self._avgdl))
        return score

    def _top(self, query: str, k: int) -> list[int]:
        qt = set(tokenize(query))
        scored = sorted(
            ((self._score(qt, i), i) for i in range(self._n)),
            key=lambda x: (-x[0], x[1]),
        )
        return [i for _, i in scored[:k]]

    def search(self, query: str, k: int) -> list[dict]:
        """Top-k chunks for the query; ties broken by chunk order."""
        return [self.chunks[i] for i in self._top(query, k)]

    def search_chunks(self, query: str, k: int) -> list[dict]:
        """Like search, but each hit is {"id": "<doc_id>#<i>", "text": ...} (v2 arms)."""
        return [{"id": self.chunk_ids[i], "text": self.chunks[i]["text"]}
                for i in self._top(query, k)]

    def read_chunks(self, doc_id: str) -> list[dict]:
        """Every chunk of one document with its id ([] when the doc_id is unknown).
        A chunk id ("<doc_id>#<i>") is accepted and reduced to its doc_id."""
        doc_id = doc_id.split("#", 1)[0]
        return [{"id": cid, "text": c["text"]}
                for cid, c in zip(self.chunk_ids, self.chunks) if c["doc_id"] == doc_id]

    def search_observation(self, query: str, k: int) -> str:
        """One line per hit: "[doc_id] chunk text"."""
        return "\n".join(f'[{c["doc_id"]}] {c["text"]}' for c in self.search(query, k))

    def read_observation(self, doc_id: str) -> str:
        """The full paragraph: "[doc_id] Title: full paragraph"."""
        doc = self.docs.get(doc_id)
        if doc is None:
            return UNKNOWN_DOC_MESSAGE.format(doc_id=doc_id)
        return f'[{doc_id}] {doc["title"]}: {doc["paragraph"]}'
