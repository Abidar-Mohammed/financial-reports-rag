"""Hybrid retrieval: BM25 + dense embeddings, fused with Reciprocal Rank Fusion,
optionally re-ranked with a cross-encoder.

BM25 is strong on exact tokens (company names, "Q3 2025", "EBITDA"), dense
retrieval on paraphrases ("turnover" vs "revenue"). RRF combines the two rank
lists without having to calibrate their scores against each other.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings
from .ingest import Chunk

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray | None = None, rrf_k: int = 60):
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.rrf_k = rrf_k
        self.bm25 = BM25Okapi([tokenize(c.full_text) for c in chunks])
        self.encoder = SentenceTransformer(settings.embed_model, device="cpu")
        if embeddings is None:
            embeddings = self.encoder.encode([c.full_text for c in chunks], normalize_embeddings=True,
                                             batch_size=32, show_progress_bar=False)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self._reranker = None

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "chunks.json").write_text(json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False),
                                               encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "HybridRetriever":
        data = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**{**d, "bbox": tuple(d["bbox"]) if d["bbox"] else None}) for d in data]
        return cls(chunks, np.load(directory / "embeddings.npy"))

    # ------------------------------------------------------------------ search
    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(settings.rerank_model, device="cpu")
        return self._reranker

    def _candidates(self, doc_id: str | None) -> np.ndarray:
        if doc_id is None:
            return np.arange(len(self.chunks))
        return np.array([i for i, c in enumerate(self.chunks) if c.doc_id == doc_id])

    def _bm25_rank(self, query: str, idx: np.ndarray) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))[idx]
        return [int(idx[i]) for i in np.argsort(-scores)]

    def _dense_rank(self, query: str, idx: np.ndarray) -> list[int]:
        q = self.encoder.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.embeddings[idx] @ q
        return [int(idx[i]) for i in np.argsort(-scores)]

    def _rrf(self, rankings: list[list[int]]) -> list[int]:
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return sorted(fused, key=fused.get, reverse=True)

    def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank", doc_id: str | None = None,
               rerank_depth: int = 20) -> list[Chunk]:
        """Return the ``k`` most relevant chunks, optionally restricted to one document."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        idx = self._candidates(doc_id)
        if mode == "bm25":
            ranking = self._bm25_rank(query, idx)
        elif mode == "dense":
            ranking = self._dense_rank(query, idx)
        else:
            ranking = self._rrf([self._bm25_rank(query, idx), self._dense_rank(query, idx)])
            if mode == "hybrid_rerank":
                pool = ranking[:rerank_depth]
                scores = self.reranker.predict([(query, self.chunks[i].full_text) for i in pool], show_progress_bar=False)
                ranking = [pool[i] for i in np.argsort(-np.asarray(scores))]
        return [self.chunks[i] for i in ranking[:k]]
