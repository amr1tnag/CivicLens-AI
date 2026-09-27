"""Engineered numeric features derived from the lexicon module.

Exposed as a scikit-learn-compatible transformer so it can sit inside a
`FeatureUnion` next to a `TfidfVectorizer` (see urgency_model.py).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from civiclens.nlp import lexicons

FEATURE_NAMES = [
    "urgency_lexicon_score",
    "sentiment_score",
    "num_urgent_terms",
    "num_dampener_terms",
    "num_frustration_terms",
    "has_repeat_mention",
    "num_duration_mentions",
    "exclamation_count",
    "capitalized_word_count",
    "text_length_words",
    "text_length_chars",
]


class TextStatsTransformer(BaseEstimator, TransformerMixin):
    """Turns raw complaint strings into a dense engineered-feature matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        rows = []
        for text in X:
            r = lexicons.analyze(text)
            rows.append([
                r.urgency_score_normalized,
                r.sentiment_score,
                len(r.matched_urgent_terms),
                len(r.matched_dampener_terms),
                len(r.matched_frustration_terms),
                float(r.has_repeat_mention),
                len(r.duration_mentions),
                r.exclamation_count,
                r.capitalized_word_count,
                len(text.split()),
                len(text),
            ])
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES)
