"""design_review_miner — surface past engineering lessons for new design issues.

The core idea: design review knowledge usually dies in folders. This module
indexes historical design review / failure records with TF-IDF vectorization
and retrieves the most similar past issues for any new problem description —
so the lesson someone learned last year gets surfaced automatically today.
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Match:
    """One retrieved historical record with its similarity score."""

    record_id: str
    score: float
    component: str
    issue_type: str
    description: str
    root_cause: str
    resolution: str
    severity: str

    def summary(self) -> str:
        return (
            f"[{self.record_id}] similarity={self.score:.2f} | "
            f"{self.component} | {self.issue_type} ({self.severity})\n"
            f"    Issue      : {self.description}\n"
            f"    Root cause : {self.root_cause}\n"
            f"    Resolution : {self.resolution}"
        )


class KnowledgeIndex:
    """TF-IDF index over historical design review records."""

    #: Columns combined into the searchable text for each record.
    TEXT_COLUMNS = ["component", "issue_type", "description", "root_cause"]

    def __init__(self, records: pd.DataFrame):
        required = set(self.TEXT_COLUMNS + ["record_id", "resolution", "severity"])
        missing = required - set(records.columns)
        if missing:
            raise ValueError(f"Records missing required columns: {sorted(missing)}")

        self.records = records.reset_index(drop=True)
        corpus = (
            self.records[self.TEXT_COLUMNS]
            .fillna("")
            .agg(" ".join, axis=1)
            .str.lower()
        )
        # Unigrams + bigrams so phrases like "weld porosity" match as a unit.
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(corpus)

    @classmethod
    def from_csv(cls, path: str) -> "KnowledgeIndex":
        return cls(pd.read_csv(path))

    def query(self, new_issue: str, top_k: int = 3,
              min_score: float = 0.05) -> List[Match]:
        """Return the *top_k* most similar past records for *new_issue*."""
        vector = self.vectorizer.transform([new_issue.lower()])
        scores = cosine_similarity(vector, self.matrix).ravel()

        order = scores.argsort()[::-1][:top_k]
        matches: List[Match] = []
        for idx in order:
            if scores[idx] < min_score:
                continue
            row = self.records.iloc[idx]
            matches.append(Match(
                record_id=row["record_id"],
                score=float(scores[idx]),
                component=row["component"],
                issue_type=row["issue_type"],
                description=row["description"],
                root_cause=row["root_cause"],
                resolution=row["resolution"],
                severity=row["severity"],
            ))
        return matches

    def review(self, new_issue: str, top_k: int = 3) -> str:
        """Human-readable report: has this issue (or a cousin) happened before?"""
        matches = self.query(new_issue, top_k=top_k)
        lines = [f"NEW ISSUE: {new_issue}", "-" * 72]
        if not matches:
            lines.append("No similar historical records found. "
                         "This may be a genuinely new failure mode — "
                         "consider adding it to the knowledge base after review.")
        else:
            lines.append(f"Found {len(matches)} similar past record(s) — "
                         "review these lessons before proceeding:\n")
            lines.extend(m.summary() + "\n" for m in matches)
        return "\n".join(lines)


def keyword_profile(records: pd.DataFrame, top_n: int = 15) -> pd.Series:
    """Most distinctive terms across the knowledge base (quick corpus insight)."""
    corpus = (
        records[KnowledgeIndex.TEXT_COLUMNS].fillna("").agg(" ".join, axis=1)
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vec.fit_transform(corpus)
    weights = pd.Series(
        matrix.sum(axis=0).A1, index=vec.get_feature_names_out()
    )
    return weights.sort_values(ascending=False).head(top_n)
