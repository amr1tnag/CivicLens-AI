"""Urgency classifier: TF-IDF text features + hand-engineered lexicon
features (see features.py / lexicons.py), combined via FeatureUnion.

Predicts one of Low / Medium / High / Critical. This is treated as a plain
multiclass problem for simplicity; evaluate.py additionally reports a
mean-absolute-error over the ordinal encoding (0..3) since, for this task,
confusing "Critical" with "Low" is a worse mistake than confusing it with
"High" and plain accuracy/F1 don't capture that.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from civiclens.config import RANDOM_SEED, URGENCY_MODEL_PATH
from civiclens.nlp.category_model import build_vectorizer
from civiclens.nlp.features import TextStatsTransformer


def build_feature_union() -> FeatureUnion:
    return FeatureUnion([
        ("tfidf", build_vectorizer()),
        ("stats", Pipeline([
            ("extract", TextStatsTransformer()),
            ("scale", StandardScaler(with_mean=False)),
        ])),
    ])


def _candidate_classifiers() -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        "LinearSVC": LinearSVC(random_state=RANDOM_SEED, max_iter=10000),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1
        ),
    }


@dataclass
class ModelSelectionResult:
    scores: dict[str, list[float]]
    best_name: str

    def summary(self) -> str:
        lines = ["Urgency model comparison (5-fold stratified CV, macro-F1):"]
        for name, scores in self.scores.items():
            lines.append(f"  {name:20s} mean={np.mean(scores):.4f}  std={np.std(scores):.4f}")
        lines.append(f"Selected: {self.best_name}")
        return "\n".join(lines)


def select_best_classifier(X_text: list[str], y: list[str], cv_splits: int = 5) -> ModelSelectionResult:
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)
    scores: dict[str, list[float]] = {}
    for name, clf in _candidate_classifiers().items():
        pipe = Pipeline([("features", build_feature_union()), ("clf", clf)])
        fold_scores = cross_val_score(pipe, X_text, y, cv=skf, scoring="f1_macro")
        scores[name] = fold_scores.tolist()
    best_name = max(scores, key=lambda n: np.mean(scores[n]))
    return ModelSelectionResult(scores=scores, best_name=best_name)


class UrgencyModel:
    """Thin, serializable wrapper around a fitted (features + classifier) pipeline."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    @classmethod
    def build(cls, classifier_name: str) -> "UrgencyModel":
        if classifier_name == "LinearSVC":
            clf = CalibratedClassifierCV(LinearSVC(random_state=RANDOM_SEED, max_iter=10000), cv=3)
        elif classifier_name == "RandomForest":
            clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
        else:
            clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
        pipeline = Pipeline([("features", build_feature_union()), ("clf", clf)])
        return cls(pipeline)

    def fit(self, X_text: list[str], y: list[str]) -> "UrgencyModel":
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

    def save(self, path=URGENCY_MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path=URGENCY_MODEL_PATH) -> "UrgencyModel":
        return cls(joblib.load(path))
