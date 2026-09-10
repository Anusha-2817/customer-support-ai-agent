"""TF-IDF + logistic regression intent classifier: the intent model of baseline B1.

    from tfidf_intent import train_on_silver
    clf = train_on_silver()
    clf.predict(["my bag never arrived"])        # [("baggage", 0.83)]

Trained on the retrieval pool's silver labels (nearest centroid + the curated cluster
mapping), so it learns to reproduce the clusters, noise included. It is only ever
evaluated against the human golden labels. Word unigrams and bigrams, sublinear TF, and
class-balanced weights so a 2% intent (Executive Club & Avios) isn't drowned out.
Training takes seconds and is deterministic, so the model is retrained on demand rather
than shipped as a pickle.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "processed" / "pool.jsonl"
SILVER_POOL = ROOT / "artifacts" / "silver" / "pool.jsonl"


def make_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                  token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")),
        ("lr", LogisticRegression(C=C, max_iter=3000, class_weight="balanced")),
    ])


class TfidfIntent:
    def __init__(self, C: float = 1.0):
        self.pipe = make_pipeline(C)

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfIntent":
        self.pipe.fit(texts, labels)
        return self

    @property
    def classes(self) -> list[str]:
        return list(self.pipe.classes_)

    def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        """(intent, probability of that intent) per text."""
        P = self.pipe.predict_proba(texts)
        best = P.argmax(axis=1)
        return [(self.pipe.classes_[j], round(float(P[i, j]), 4)) for i, j in enumerate(best)]


def silver_training_data() -> pd.DataFrame:
    pool = pd.read_json(POOL, lines=True, convert_dates=False)[["case_id", "customer_msg"]]
    silver = pd.read_json(SILVER_POOL, lines=True)[["case_id", "silver_intent"]]
    return pool.merge(silver, on="case_id", validate="one_to_one")


def train_on_silver(C: float = 1.0) -> TfidfIntent:
    d = silver_training_data()
    return TfidfIntent(C).fit(d.customer_msg.tolist(), d.silver_intent.tolist())


def silver_agreement_cv(folds: int = 5) -> dict:
    """Cross-validated agreement with the silver labels. A sanity check only: it measures
    how well TF-IDF reproduces the embedding clusters, not whether intents are right."""
    d = silver_training_data()
    pred = cross_val_predict(make_pipeline(), d.customer_msg, d.silver_intent,
                             cv=StratifiedKFold(folds, shuffle=True, random_state=0))
    agree = float(np.mean(pred == d.silver_intent.to_numpy()))
    majority = float(d.silver_intent.value_counts(normalize=True).iloc[0])
    return {"n": len(d), "agreement_with_silver": round(agree, 4), "majority_share": round(majority, 4)}
