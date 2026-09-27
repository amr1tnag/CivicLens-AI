"""Train the category classifier, urgency classifier, and duplicate index.

Usage:
    python -m civiclens.train
"""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from civiclens.config import (
    CATEGORY_MODEL_PATH,
    DATASET_PATH,
    DUPLICATE_INDEX_PATH,
    MODELS_DIR,
    RANDOM_SEED,
    URGENCY_MODEL_PATH,
)
from civiclens.nlp.category_model import CategoryModel
from civiclens.nlp.category_model import select_best_classifier as select_best_category_classifier
from civiclens.nlp.duplicates import DuplicateIndex
from civiclens.nlp.urgency_model import UrgencyModel
from civiclens.nlp.urgency_model import select_best_classifier as select_best_urgency_classifier

TEST_SPLIT_PATH = MODELS_DIR / "test_split.joblib"


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run `python -m civiclens.data.generate_dataset` first."
        )
    return pd.read_csv(DATASET_PATH)


def main() -> None:
    df = load_dataset()
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["category"]
    )
    print(f"Train size: {len(train_df)}  Test size: {len(test_df)}")

    print("\n=== Category model selection ===")
    cat_selection = select_best_category_classifier(train_df["text"].tolist(), train_df["category"].tolist())
    print(cat_selection.summary())
    category_model = CategoryModel.build(cat_selection.best_name)
    category_model.fit(train_df["text"].tolist(), train_df["category"].tolist())
    category_model.save()
    print(f"Saved category model -> {CATEGORY_MODEL_PATH}")

    print("\n=== Urgency model selection ===")
    urg_selection = select_best_urgency_classifier(train_df["text"].tolist(), train_df["urgency"].tolist())
    print(urg_selection.summary())
    urgency_model = UrgencyModel.build(urg_selection.best_name)
    urgency_model.fit(train_df["text"].tolist(), train_df["urgency"].tolist())
    urgency_model.save()
    print(f"Saved urgency model -> {URGENCY_MODEL_PATH}")

    # Duplicate index is seeded with the *full* corpus (train + test): its
    # job is to answer "have we seen something like this before?" against
    # all historical complaints, it is not a predictive model being
    # evaluated for generalization.
    dup_index = DuplicateIndex.fit(df["text"].tolist(), df["complaint_id"].tolist(), df["category"].tolist())
    dup_index.save()
    print(f"Saved duplicate index -> {DUPLICATE_INDEX_PATH}")

    TEST_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "test_df": test_df,
        "train_size": len(train_df),
        "category_selection": cat_selection,
        "urgency_selection": urg_selection,
        "category_model_name": cat_selection.best_name,
        "urgency_model_name": urg_selection.best_name,
    }, TEST_SPLIT_PATH)
    print(f"Saved test split + selection metadata -> {TEST_SPLIT_PATH}")


if __name__ == "__main__":
    main()
