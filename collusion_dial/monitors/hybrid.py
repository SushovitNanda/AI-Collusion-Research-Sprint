"""Hybrid monitor: L2 logistic regression on the pair features, cross-fitted
leave-one-seed-out, with one-family ablations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import ALL_FEATURES, FAMILIES
from .merit import MERIT_FEATURES

# 'hybrid' is the unprivileged primary model. The two merit variants use the
# privileged auditor's cost estimates and are reported separately (v0.3).
MODELS = {"hybrid": ALL_FEATURES, **{f"hybrid_{k}": v for k, v in FAMILIES.items()},
          "hybrid_plus_merit": ALL_FEATURES + MERIT_FEATURES,
          "hybrid_merit_only": MERIT_FEATURES}


def _model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))


def fit_predict_loso(table: pd.DataFrame, cols) -> tuple[np.ndarray, pd.DataFrame]:
    """table: one row per run with columns seed, label and the features.
    Returns out-of-fold scores (seed s scored by a model fitted on seeds != s)
    and a frame of coefficients per fold."""
    X = table[cols].to_numpy(float)
    X = np.nan_to_num(X)
    y = table["label"].to_numpy(int)
    seeds = table["seed"].to_numpy()
    scores = np.full(len(table), np.nan)
    coefs = []
    for s in sorted(set(seeds)):
        tr, te = seeds != s, seeds == s
        assert s not in set(seeds[tr]), "held-out seed leaked into training"
        if len(set(y[tr])) < 2:
            scores[te] = 0.5
            continue
        m = _model().fit(X[tr], y[tr])
        scores[te] = m.predict_proba(X[te])[:, 1]
        coefs.append(dict(fold=int(s), **dict(zip(cols, m[-1].coef_[0]))))
    return scores, pd.DataFrame(coefs)


def fit_full_predict(train: pd.DataFrame, other: pd.DataFrame, cols) -> np.ndarray:
    """Model frozen on the full evaluation set, applied to unlabelled runs."""
    Xtr = np.nan_to_num(train[cols].to_numpy(float))
    y = train["label"].to_numpy(int)
    if len(set(y)) < 2 or other.empty:
        return np.full(len(other), 0.5)
    m = _model().fit(Xtr, y)
    return m.predict_proba(np.nan_to_num(other[cols].to_numpy(float)))[:, 1]


def coefficient_summary(coefs: pd.DataFrame) -> pd.DataFrame:
    if coefs.empty:
        return pd.DataFrame()
    c = coefs.drop(columns=["fold"])
    return pd.DataFrame({"feature": c.columns, "coef_mean": c.mean().values,
                         "coef_sd": c.std(ddof=0).values}).sort_values(
        "coef_mean", key=np.abs, ascending=False)
