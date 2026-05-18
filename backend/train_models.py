"""
Train three risk models — one for each point in the semester:

    pre  : start of semester (no G1, no G2)
    mid  : mid-semester        (G1 only)
    full : late semester       (G1 and G2)

Categorical features are handled natively by XGBoost via
`enable_categorical=True` (no label-encoding). Probabilities are
isotonically calibrated so the % shown to the user is meaningful.

Run with:  python backend/train_models.py
Output:    ./models_bundle.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAT_CSV = os.path.join(BASE, "student-mat.csv")
POR_CSV = os.path.join(BASE, "student-por.csv")
OUT = os.path.join(BASE, "models_bundle.pkl")

CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic", "subject",
]


def load_data() -> pd.DataFrame:
    mat = pd.read_csv(MAT_CSV, sep=";")
    por = pd.read_csv(POR_CSV, sep=";")
    mat["subject"] = "math"
    por["subject"] = "portuguese"
    df = pd.concat([mat, por], ignore_index=True)
    df["at_risk"] = (df["G3"] < 10).astype(int)
    return df.drop(columns=["G3"])


def engineer(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Build stage-specific feature frame. stage ∈ {pre, mid, full}."""
    d = df.copy()

    # Always-available engineered signals — these become more important
    # when G1/G2 are unavailable.
    d["alcohol_total"] = d["Dalc"] + d["Walc"]
    d["parent_edu"]    = d["Medu"] + d["Fedu"]
    d["high_absences"] = (d["absences"] > 10).astype(int)
    d["has_failures"]  = (d["failures"] > 0).astype(int)
    d["social_load"]   = d["goout"] + d["freetime"]
    d["support_count"] = (
        (d["schoolsup"] == "yes").astype(int)
        + (d["famsup"] == "yes").astype(int)
        + (d["paid"] == "yes").astype(int)
    )

    if stage == "pre":
        d = d.drop(columns=["G1", "G2"])
    elif stage == "mid":
        d = d.drop(columns=["G2"])
    elif stage == "full":
        d["avg_early_grade"] = (d["G1"] + d["G2"]) / 2.0
        d["grade_trend"]     = d["G2"] - d["G1"]
    else:
        raise ValueError(f"Unknown stage: {stage}")

    for c in CATEGORICAL_COLS:
        if c in d.columns:
            d[c] = d[c].astype("category")

    return d


def _xgb_params(stage: str) -> dict:
    # Slightly more regularisation for the PRE model since the signal is weaker
    if stage == "pre":
        return dict(n_estimators=400, max_depth=3, learning_rate=0.04,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=2.0, min_child_weight=3)
    return dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                reg_lambda=1.0, min_child_weight=2)


def train_one(df_stage: pd.DataFrame, stage: str, seed: int = 42) -> dict:
    X = df_stage.drop(columns=["at_risk"])
    y = df_stage["at_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    params = _xgb_params(stage)
    common = dict(
        enable_categorical=True, tree_method="hist",
        eval_metric="logloss", random_state=seed, n_jobs=-1,
    )

    # Raw model — used for SHAP explanations
    raw = XGBClassifier(**params, **common)
    raw.fit(X_train, y_train, verbose=False)

    # Calibrated wrapper — used for the probability shown to the user.
    # Isotonic calibration fixes the inflated probabilities caused by
    # an imbalanced loss / un-calibrated XGB output.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cal = CalibratedClassifierCV(
        estimator=XGBClassifier(**params, **common),
        method="isotonic", cv=cv,
    )
    cal.fit(X_train, y_train)

    y_prob_raw = raw.predict_proba(X_test)[:, 1]
    y_prob_cal = cal.predict_proba(X_test)[:, 1]
    y_pred = (y_prob_cal >= 0.5).astype(int)

    metrics = {
        "roc_auc":   float(roc_auc_score(y_test, y_prob_cal)),
        "avg_prec":  float(average_precision_score(y_test, y_prob_cal)),
        "brier_raw": float(brier_score_loss(y_test, y_prob_raw)),
        "brier_cal": float(brier_score_loss(y_test, y_prob_cal)),
    }

    print(f"\n=== {stage.upper()} ({len(X.columns)} features) ===")
    print(classification_report(y_test, y_pred, target_names=["Pass", "At-Risk"]))
    print(
        f"ROC-AUC: {metrics['roc_auc']:.3f}  "
        f"AP: {metrics['avg_prec']:.3f}  "
        f"Brier raw→cal: {metrics['brier_raw']:.3f} → {metrics['brier_cal']:.3f}"
    )

    categories = {
        c: list(X[c].cat.categories)
        for c in CATEGORICAL_COLS if c in X.columns
    }
    return {
        "model":       raw,
        "calibrator":  cal,
        "features":    X.columns.tolist(),
        "categorical": categories,
        "metrics":     metrics,
    }


def main() -> None:
    df = load_data()
    bundle = {}
    for stage in ("pre", "mid", "full"):
        bundle[stage] = train_one(engineer(df, stage), stage)
    joblib.dump(bundle, OUT)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
