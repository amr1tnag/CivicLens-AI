"""Complaint category classifier: TF-IDF features + a linear classifier.

Model selection compares Multinomial Naive Bayes, Logistic Regression and
Linear SVM via stratified cross-validation on macro-F1, and keeps whichever
generalizes best rather than assuming one algorithm in advance.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from civiclens.config import CATEGORY_MODEL_PATH, RANDOM_SEED
from civiclens.nlp.preprocessing import vectorizer_tokenizer


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=vectorizer_tokenizer,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
    )


def _candidate_classifiers() -> dict:
    return {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "LinearSVC": LinearSVC(random_state=RANDOM_SEED, max_iter=10000),
    }


@dataclass
class ModelSelectionResult:
    scores: dict[str, list[float]]
    best_name: str

    def summary(self) -> str:
        lines = ["Category model comparison (5-fold stratified CV, macro-F1):"]
        for name, scores in self.scores.items():
            lines.append(f"  {name:20s} mean={np.mean(scores):.4f}  std={np.std(scores):.4f}")
        lines.append(f"Selected: {self.best_name}")
        return "\n".join(lines)


def select_best_classifier(X_text: list[str], y: list[str], cv_splits: int = 5) -> ModelSelectionResult:
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
    scores: dict[str, list[float]] = {}
    for name, clf in _candidate_classifiers().items():
        pipe = Pipeline([("tfidf", build_vectorizer()), ("clf", clf)])
        fold_scores = cross_val_score(pipe, X_text, y, cv=skf, scoring="f1_macro")
        scores[name] = fold_scores.tolist()
    best_name = max(scores, key=lambda n: np.mean(scores[n]))
    return ModelSelectionResult(scores=scores, best_name=best_name)


class CategoryModel:
    """Thin, serializable wrapper around a fitted (vectorizer + classifier) pipeline."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    @classmethod
    def build(cls, classifier_name: str) -> "CategoryModel":
        if classifier_name == "LinearSVC":
            # LinearSVC has no predict_proba; calibrate so the API can
            # still return a confidence score alongside the label.
            clf = CalibratedClassifierCV(LinearSVC(random_state=RANDOM_SEED, max_iter=10000), cv=3)
        elif classifier_name == "LogisticRegression":
            clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
        else:
            clf = MultinomialNB()
        pipeline = Pipeline([("tfidf", build_vectorizer()), ("clf", clf)])
        return cls(pipeline)

    def fit(self, X_text: list[str], y: list[str]) -> "CategoryModel":
        self.pipeline.fit(X_text, y)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)

    @property
    def classes_(self) -> np.ndarray:
        return self.pipeline.named_steps["clf"].classes_

    def predict_one(self, text: str) -> tuple[str, float]:
        proba = self.predict_proba([text])[0]
        idx = int(np.argmax(proba))
        return str(self.classes_[idx]), float(proba[idx])

    def save(self, path=CATEGORY_MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path=CATEGORY_MODEL_PATH) -> "CategoryModel":
        return cls(joblib.load(path))
