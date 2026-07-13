"""design_review_miner.hybrid_index — v2: hybrid lexical + semantic retrieval.

v1 (index.py) matches words. This module adds a second, parallel channel that
matches *meaning*, then fuses the two rankings. The design goals, in order:

1. Exact-term wins survive  — part numbers, spec callouts, "salt spray" still
   match literally (BM25 channel).
2. Vocabulary mismatch is bridged — "joint loosening under vibration" should
   retrieve "torque relaxation" records (embedding channel).
3. Auditability is preserved — every match reports *which channel(s)* found
   it and at what rank, so an engineer can still answer "why did this match?"

Architecture:

    new issue ──┬─► BM25 over token corpus        ─► lexical ranking ─┐
                │                                                     ├─► RRF fusion ─► top-k Matches
                └─► embed + cosine over vectors    ─► semantic ranking ┘

Fusion is Reciprocal Rank Fusion (RRF): score = Σ 1/(k + rank_channel).
RRF is deliberately rank-based, not score-based — BM25 scores and cosine
similarities live on incomparable scales, and RRF sidesteps calibration
entirely. It is the boring, robust default used across production search.

Embedding backends (pluggable, chosen at construction):
  * "sentence-transformers" — best quality; needs the optional dependency
    and a one-time model download.
  * "lsa" — TruncatedSVD over TF-IDF (latent semantic analysis). Zero new
    heavyweight deps, captures co-occurrence structure ("loosening" and
    "torque" appearing in similar records land near each other). A genuine
    if modest semantic signal, and a sane offline/air-gapped fallback for
    factory environments.
  * any callable — pass `embed_fn=lambda texts: np.ndarray` to plug in an
    API embedder (Voyage, OpenAI, etc.) without touching this file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .index import KnowledgeIndex, Match

# --------------------------------------------------------------------------
# Tokenization (shared by BM25 and LSA fallback)
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\.]*")


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens; keeps hyphens/dots so 'DR-007' and 'M8.1' survive."""
    return _TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------
# Minimal BM25 (Okapi). ~30 lines, no dependency; swap in `rank_bm25` if
# preferred — the interface is identical (fit on corpus, score a query).
# --------------------------------------------------------------------------

class _BM25:
    def __init__(self, corpus_tokens: Sequence[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=float)
        self.avgdl = self.doc_len.mean() if len(corpus_tokens) else 0.0
        self.n_docs = len(corpus_tokens)
        # term -> {doc_index: term_frequency}
        self.tf: dict[str, dict[int, int]] = {}
        for i, doc in enumerate(corpus_tokens):
            for tok in doc:
                self.tf.setdefault(tok, {})
                self.tf[tok][i] = self.tf[tok].get(i, 0) + 1
        self.idf = {
            term: np.log(1 + (self.n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self.tf.items()
        }

    def scores(self, query_tokens: List[str]) -> np.ndarray:
        out = np.zeros(self.n_docs)
        for tok in query_tokens:
            postings = self.tf.get(tok)
            if not postings:
                continue
            idf = self.idf[tok]
            for i, f in postings.items():
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


# --------------------------------------------------------------------------
# Embedding backends
# --------------------------------------------------------------------------

def _make_sbert_embedder(model_name: str) -> Callable[[Sequence[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer  # optional dep
    model = SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(model.encode(list(texts), normalize_embeddings=True))

    return embed


class _LSAEmbedder:
    """TF-IDF -> TruncatedSVD. Fit on the corpus, reused for queries."""

    def __init__(self, corpus: Sequence[str], n_components: int = 64):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        self._normalize = normalize
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf = self.vectorizer.fit_transform(corpus)
        k = max(2, min(n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0)
        self.corpus_vectors = self._normalize(self.svd.fit_transform(tfidf))

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return self._normalize(self.svd.transform(self.vectorizer.transform(texts)))


# --------------------------------------------------------------------------
# Hybrid match: a Match that can explain which channel found it
# --------------------------------------------------------------------------

@dataclass
class HybridMatch(Match):
    lexical_rank: Optional[int] = None    # 1-based rank in BM25 channel, None if unranked
    semantic_rank: Optional[int] = None   # 1-based rank in embedding channel
    channels: List[str] = field(default_factory=list)

    def summary(self) -> str:
        via = ", ".join(
            f"{name} #{rank}"
            for name, rank in (("lexical", self.lexical_rank), ("semantic", self.semantic_rank))
            if rank is not None
        )
        base = super().summary().replace("similarity=", "rrf=")
        return base + f"\n    Matched via: {via}"


# --------------------------------------------------------------------------
# The hybrid index
# --------------------------------------------------------------------------

class HybridIndex(KnowledgeIndex):
    """BM25 + embedding retrieval fused with Reciprocal Rank Fusion.

    Drop-in replacement for KnowledgeIndex:

        index = HybridIndex.from_csv("data/design_review_records.csv")
        print(index.review("Bolts backing out during shaker table runs"))
    """

    def __init__(
        self,
        records: pd.DataFrame,
        embedder: str | Callable[[Sequence[str]], np.ndarray] = "auto",
        sbert_model: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
        channel_depth: int = 10,
    ):
        super().__init__(records)  # keeps v1 TF-IDF around for A/B comparison
        self.rrf_k = rrf_k
        self.channel_depth = channel_depth

        corpus = (
            self.records[self.TEXT_COLUMNS].fillna("").agg(" ".join, axis=1).tolist()
        )

        # Channel 1: lexical
        self._corpus_tokens = [_tokenize(t) for t in corpus]
        self.bm25 = _BM25(self._corpus_tokens)

        # Channel 2: semantic
        if callable(embedder):
            self.embed = embedder
            self.corpus_embeddings = np.asarray(self.embed(corpus))
        elif embedder in ("auto", "sentence-transformers"):
            try:
                self.embed = _make_sbert_embedder(sbert_model)
                self.corpus_embeddings = self.embed(corpus)
            except Exception:
                if embedder == "sentence-transformers":
                    raise
                lsa = _LSAEmbedder(corpus)
                self.embed, self.corpus_embeddings = lsa, lsa.corpus_vectors
        elif embedder == "lsa":
            lsa = _LSAEmbedder(corpus)
            self.embed, self.corpus_embeddings = lsa, lsa.corpus_vectors
        else:
            raise ValueError(f"Unknown embedder: {embedder!r}")

    # -- retrieval ----------------------------------------------------------

    def _channel_rankings(self, new_issue: str) -> dict[str, List[int]]:
        depth = min(self.channel_depth, len(self.records))

        bm25_scores = self.bm25.scores(_tokenize(new_issue))
        lexical = [i for i in np.argsort(bm25_scores)[::-1][:depth] if bm25_scores[i] > 0]

        q = np.asarray(self.embed([new_issue]))[0]
        sims = self.corpus_embeddings @ q
        semantic = list(np.argsort(sims)[::-1][:depth])

        return {"lexical": lexical, "semantic": semantic}

    def query(self, new_issue: str, top_k: int = 3, min_score: float = 0.0) -> List[HybridMatch]:
        rankings = self._channel_rankings(new_issue)

        fused: dict[int, float] = {}
        ranks: dict[int, dict[str, int]] = {}
        for channel, order in rankings.items():
            for rank, doc_idx in enumerate(order, start=1):
                fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (self.rrf_k + rank)
                ranks.setdefault(doc_idx, {})[channel] = rank

        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        matches: List[HybridMatch] = []
        for doc_idx, score in top:
            if score < min_score:
                continue
            row = self.records.iloc[doc_idx]
            r = ranks[doc_idx]
            matches.append(HybridMatch(
                record_id=row["record_id"], score=float(score),
                component=row["component"], issue_type=row["issue_type"],
                description=row["description"], root_cause=row["root_cause"],
                resolution=row["resolution"], severity=row["severity"],
                lexical_rank=r.get("lexical"), semantic_rank=r.get("semantic"),
                channels=sorted(r),
            ))
        return matches

    # -- evaluation hook ------------------------------------------------------

    def compare_to_v1(self, new_issue: str, top_k: int = 3) -> str:
        """Side-by-side v1 vs v2 for the same query — for building trust."""
        v1 = super().query(new_issue, top_k=top_k)
        v2 = self.query(new_issue, top_k=top_k)
        lines = [f"QUERY: {new_issue}", "-" * 72]
        lines.append("v1 TF-IDF : " + (", ".join(m.record_id for m in v1) or "(none)"))
        lines.append("v2 hybrid : " + (", ".join(
            f"{m.record_id}[{'+'.join(c[0].upper() for c in m.channels)}]" for m in v2) or "(none)"))
        return "\n".join(lines)
