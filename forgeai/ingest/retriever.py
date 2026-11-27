"""Hybrid BM25 + dense vector retrieval with Reciprocal Rank Fusion and optional reranking."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

from forgeai.ingest.chunker import Chunk
from forgeai.schemas.agent import RetrievalConfig

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Lightweight Okapi BM25 lexical index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: list[Counter[str]] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def fit(self, documents: list[str]) -> None:
        """Build BM25 statistics over ``documents``."""
        self._doc_tokens = [_tokenize(doc) for doc in documents]
        self._doc_freqs = [Counter(tokens) for tokens in self._doc_tokens]
        n_docs = len(self._doc_tokens)
        if n_docs == 0:
            self._avgdl = 0.0
            self._idf = {}
            return

        self._avgdl = sum(len(t) for t in self._doc_tokens) / n_docs
        df: Counter[str] = Counter()
        for freq in self._doc_freqs:
            df.update(freq.keys())
        self._idf = {
            term: math.log(1.0 + (n_docs - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def score(self, query: str) -> list[float]:
        """Score every indexed document against ``query``."""
        if not self._doc_tokens:
            return []
        q_tokens = _tokenize(query)
        scores = [0.0] * len(self._doc_tokens)
        avgdl = self._avgdl or 1.0
        for idx, freq in enumerate(self._doc_freqs):
            dl = len(self._doc_tokens[idx])
            s = 0.0
            for term in q_tokens:
                if term not in freq:
                    continue
                tf = freq[term]
                idf = self._idf.get(term, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                s += idf * ((tf * (self.k1 + 1.0)) / (denom or 1.0))
            scores[idx] = s
        return scores


class Embedder(Protocol):
    """Protocol for dense text embedders."""

    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, query: str) -> np.ndarray: ...


class TfidfEmbedder:
    """Fast offline L2-normalized TF-IDF dense embedder (zero external downloads)."""

    name: str = "tfidf"

    def __init__(self, max_features: int = 2048) -> None:
        self.max_features = max_features
        self._vectorizer = TfidfVectorizer(max_features=max_features, norm="l2")
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        if texts:
            self._vectorizer.fit(texts)
            self._fitted = True

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        matrix = self._vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix.toarray().astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        if not self._fitted:
            return np.zeros((1,), dtype=np.float32)
        vec = self._vectorizer.transform([query]).toarray().astype(np.float32)
        return vec[0]


class SentenceTransformerEmbedder:
    """Lazy-loaded SentenceTransformer embedder (default: BAAI/bge-small-en-v1.5)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.name = model_name
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        emb = self._get_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(emb, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        emb = self._get_model().encode([query], normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(emb[0], dtype=np.float32)


class Reranker(Protocol):
    """Protocol for cross-encoder rerankers."""

    def rerank(self, query: str, texts: list[str]) -> list[float]: ...


class NoOpReranker:
    """Pass-through reranker returning uniform scores (used when reranking is disabled)."""

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [1.0 for _ in texts]


class CrossEncoderReranker:
    """Lazy-loaded cross-encoder reranker (default: ms-marco-MiniLM-L-6-v2)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        scores = self._get_model().predict(pairs)
        return [float(1.0 / (1.0 + math.exp(-float(s)))) for s in scores]


class RetrievedChunk(BaseModel):
    """A retrieved chunk paired with its fused/reranked relevance score and ranks."""

    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


class HybridRetriever:
    """Hybrid BM25 + dense retriever using weighted Reciprocal Rank Fusion and optional reranker."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.embedder: Embedder = embedder or TfidfEmbedder()
        self.reranker: Reranker = reranker or NoOpReranker()
        self.chunks: list[Chunk] = []
        self._bm25 = BM25Index()
        self._embeddings: np.ndarray = np.zeros((0, 1), dtype=np.float32)

    def index(self, chunks: list[Chunk]) -> None:
        """Index ``chunks`` into both BM25 and dense vector spaces."""
        self.chunks = list(chunks)
        texts = [c.contextual_text for c in self.chunks]
        self._bm25.fit(texts)
        self._embeddings = self.embedder.embed_documents(texts)

    def search(self, query: str, config: RetrievalConfig | None = None) -> list[RetrievedChunk]:
        """Retrieve top chunks using BM25 + dense RRF fusion + optional reranking."""
        if not self.chunks:
            return []
        cfg = config or RetrievalConfig()

        bm25_scores = self._bm25.score(query)
        q_vec = self.embedder.embed_query(query)
        if self._embeddings.shape[0] > 0 and q_vec.shape[0] == self._embeddings.shape[1]:
            dense_scores = (self._embeddings @ q_vec).tolist()
        else:
            dense_scores = [0.0] * len(self.chunks)

        bm25_order = sorted(range(len(self.chunks)), key=lambda i: bm25_scores[i], reverse=True)
        dense_order = sorted(range(len(self.chunks)), key=lambda i: dense_scores[i], reverse=True)

        bm25_rank_map = {doc_i: rank + 1 for rank, doc_i in enumerate(bm25_order)}
        dense_rank_map = {doc_i: rank + 1 for rank, doc_i in enumerate(dense_order)}

        rrf_k = 60.0
        alpha = max(0.0, min(1.0, cfg.hybrid_alpha))
        rrf_scores: dict[int, float] = {}
        for idx in range(len(self.chunks)):
            b_rank = bm25_rank_map[idx]
            d_rank = dense_rank_map[idx]
            rrf_scores[idx] = (1.0 - alpha) / (rrf_k + b_rank) + alpha / (rrf_k + d_rank)

        top_indices = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[: cfg.top_k]

        candidate_texts = [self.chunks[i].contextual_text for i in top_indices]
        if isinstance(self.reranker, NoOpReranker):
            # Combine normalized lexical + dense signal when no cross-encoder is attached
            max_bm25 = max((bm25_scores[i] for i in top_indices), default=1.0) or 1.0
            final_scores = [
                max(0.0, min(1.0, (1.0 - alpha) * (bm25_scores[i] / max_bm25) + alpha * max(0.0, dense_scores[i])))
                for i in top_indices
            ]
        else:
            final_scores = self.reranker.rerank(query, candidate_texts)

        results: list[RetrievedChunk] = []
        for idx, score in zip(top_indices, final_scores):
            if score < cfg.min_relevance_score:
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=float(score),
                    bm25_rank=bm25_rank_map[idx],
                    dense_rank=dense_rank_map[idx],
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[: cfg.rerank_top_n]

    def save(self, index_dir: Path) -> None:
        """Persist chunks, embeddings, and index metadata to ``index_dir``."""
        target = Path(index_dir)
        target.mkdir(parents=True, exist_ok=True)
        chunks_payload = [c.model_dump(mode="json") for c in self.chunks]
        (target / "chunks.json").write_text(json.dumps(chunks_payload, indent=2), encoding="utf-8")
        np.savez_compressed(target / "embeddings.npz", embeddings=self._embeddings)
        meta = {"embedder_name": getattr(self.embedder, "name", "custom"), "chunk_count": len(self.chunks)}
        (target / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        index_dir: Path,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> HybridRetriever:
        """Load a persisted HybridRetriever from ``index_dir``."""
        target = Path(index_dir)
        raw_chunks = json.loads((target / "chunks.json").read_text(encoding="utf-8"))
        chunks = [Chunk.model_validate(item) for item in raw_chunks]
        meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))

        chosen_embedder = embedder
        if chosen_embedder is None:
            if meta.get("embedder_name") == "tfidf":
                tfidf = TfidfEmbedder()
                tfidf.fit([c.contextual_text for c in chunks])
                chosen_embedder = tfidf
            else:
                chosen_embedder = SentenceTransformerEmbedder(meta["embedder_name"])

        retriever = cls(embedder=chosen_embedder, reranker=reranker)
        retriever.chunks = chunks
        retriever._bm25.fit([c.contextual_text for c in chunks])
        npz = np.load(target / "embeddings.npz")
        retriever._embeddings = npz["embeddings"]
        return retriever

