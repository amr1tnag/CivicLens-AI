"""Duplicate / repeat-complaint detection via TF-IDF cosine similarity.

Many real 311-style systems see the same underlying issue (e.g. "the
streetlight on 5th Cross Road is out") reported by several different
residents. Recognizing that is useful on its own (route them as one ticket)
and is also a strong prioritization signal: five independent reports about
the same pothole is a stronger case for bumping priority than the text of
any single report.

This module keeps a single fitted TfidfVectorizer + a growing sparse matrix
of every complaint seen so far (starting from the training corpus) and
answers "have we seen something like this before?" via cosine similarity,
restricted to the same category for both relevance and speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from civiclens.config import DUPLICATE_INDEX_PATH
from civiclens.nlp.preprocessing import vectorizer_tokenizer

DEFAULT_SIMILARITY_THRESHOLD = 0.55


@dataclass
class SimilarComplaint:
    complaint_id: str
    similarity: float
    category: str
    text: str


class DuplicateIndex:
    def __init__(self, vectorizer: TfidfVectorizer, matrix: sparse.csr_matrix, metadata: pd.DataFrame):
        self.vectorizer = vectorizer
        self.matrix = matrix
        # metadata columns: complaint_id, category, text
        self.metadata = metadata.reset_index(drop=True)

    @classmethod
    def fit(cls, texts: list[str], complaint_ids: list[str], categories: list[str]) -> "DuplicateIndex":
        vectorizer = TfidfVectorizer(
            tokenizer=vectorizer_tokenizer,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        metadata = pd.DataFrame({"complaint_id": complaint_ids, "category": categories, "text": texts})
        return cls(vectorizer, matrix, metadata)

    def find_similar(
        self,
        text: str,
        category: str | None = None,
        top_k: int = 5,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[SimilarComplaint]:
        query_vec = self.vectorizer.transform([text])
        if query_vec.nnz == 0:
            return []  # every token in `text` is out-of-vocabulary; nothing to compare

        if category is not None:
            mask = (self.metadata["category"] == category).to_numpy()
        else:
            mask = np.ones(len(self.metadata), dtype=bool)
        if not mask.any():
            return []

        candidate_matrix = self.matrix[mask]
        sims = cosine_similarity(query_vec, candidate_matrix)[0]
        candidate_meta = self.metadata[mask].reset_index(drop=True)

        order = np.argsort(-sims)
        results: list[SimilarComplaint] = []
        for i in order[:top_k]:
            if sims[i] < threshold:
                break
            row = candidate_meta.iloc[i]
            results.append(SimilarComplaint(
                complaint_id=row["complaint_id"],
                similarity=round(float(sims[i]), 4),
                category=row["category"],
                text=row["text"],
            ))
        return results

    def add(self, text: str, complaint_id: str, category: str) -> None:
        """Append a newly submitted complaint to the index for future lookups."""
        new_vec = self.vectorizer.transform([text])
        self.matrix = sparse.vstack([self.matrix, new_vec], format="csr")
        new_row = pd.DataFrame([{"complaint_id": complaint_id, "category": category, "text": text}])
        self.metadata = pd.concat([self.metadata, new_row], ignore_index=True)

    def save(self, path=DUPLICATE_INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "matrix": self.matrix, "metadata": self.metadata}, path)

    @classmethod
    def load(cls, path=DUPLICATE_INDEX_PATH) -> "DuplicateIndex":
        obj = joblib.load(path)
        return cls(obj["vectorizer"], obj["matrix"], obj["metadata"])
