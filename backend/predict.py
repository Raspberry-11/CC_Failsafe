"""Inference for the three-stage risk models (pre / mid / full)."""

import os
import joblib
import numpy as np
import pandas as pd
import shap

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(BASE_DIR, "models_bundle.pkl")

_bundle = None
_explainers: dict = {}

STAGES = ("pre", "mid", "full")

STAGE_LABEL = {
    "pre":  "Start of Semester",
    "mid":  "After G1",
    "full": "After G1 + G2",
}

INTERVENTIONS = [
    ("avg_early_grade", lambda v: v < 10, "Enrol in extra tutoring; weekly grade check-ins with faculty."),
    ("grade_trend",     lambda v: v < 0,  "Grade declining — urgent counselling to identify blockers."),
    ("G1",              lambda v: v < 10, "Low G1 grade — schedule mid-semester tutoring."),
    ("absences",        lambda v: v > 10, "High absenteeism — attendance monitoring + guardian contact."),
    ("failures",        lambda v: v > 0,  "Past failures — assign peer mentor + remedial materials."),
    ("studytime",       lambda v: v < 2,  "Low study time — structured plan recommended (2–4 hrs/day)."),
    ("alcohol_total",   lambda v: v > 4,  "High alcohol use — refer to student wellness services."),
    ("internet",        lambda v: str(v) == "no", "No internet access — provide library access + offline resources."),
]


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(BUNDLE_PATH)
    return _bundle


def _explainer(stage: str):
    if stage not in _explainers:
        _explainers[stage] = shap.TreeExplainer(_bundle[stage]["model"])
    return _explainers[stage]


def _prepare_row(raw: dict, stage: str) -> pd.DataFrame:
    """Engineer + dtype-cast a single raw input dict for the given stage."""
    b = _bundle[stage]
    feats: list = b["features"]
    cats: dict  = b["categorical"]

    d = dict(raw)

    # Always-available engineered signals (must mirror train_models.py)
    d["alcohol_total"] = float(d.get("Dalc", 0)) + float(d.get("Walc", 0))
    d["parent_edu"]    = float(d.get("Medu", 0)) + float(d.get("Fedu", 0))
    d["high_absences"] = int(float(d.get("absences", 0)) > 10)
    d["has_failures"]  = int(float(d.get("failures", 0)) > 0)
    d["social_load"]   = float(d.get("goout", 0)) + float(d.get("freetime", 0))
    d["support_count"] = (
        int(str(d.get("schoolsup", "no")) == "yes")
        + int(str(d.get("famsup", "no")) == "yes")
        + int(str(d.get("paid", "no")) == "yes")
    )

    if stage == "full":
        g1, g2 = d.get("G1"), d.get("G2")
        if g1 is not None and g2 is not None:
            d["avg_early_grade"] = (float(g1) + float(g2)) / 2.0
            d["grade_trend"]     = float(g2) - float(g1)

    row = {f: d.get(f, None) for f in feats}
    df = pd.DataFrame([row])

    for col, categories in cats.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=categories)
    for col in df.columns:
        if col not in cats:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _json_safe(v):
    """Convert numpy / pandas scalars + categoricals to plain JSON types."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (int, float, str, bool)):
        return v
    return str(v)


def predict_student(raw: dict, stage: str = "full") -> dict:
    _load()
    if stage not in _bundle:
        raise ValueError(f"Unknown stage: {stage}")
    b = _bundle[stage]
    row = _prepare_row(raw, stage)

    prob = float(b["calibrator"].predict_proba(row)[0, 1])

    # SHAP from the raw XGB model
    sv = _explainer(stage)(row)
    shap_s = pd.Series(sv[0].values, index=sv[0].feature_names)
    top5 = shap_s.abs().nlargest(5).index
    shap_top5 = {
        f: {"value": _json_safe(row.iloc[0][f]), "shap": float(shap_s[f])}
        for f in top5
    }

    plans = []
    for feat, check, msg in INTERVENTIONS:
        if feat not in row.columns:
            continue
        v = row.iloc[0][feat]
        if pd.isna(v):
            continue
        try:
            if check(v):
                plans.append(msg)
        except Exception:
            pass
    if not plans:
        plans = ["No immediate intervention required."]

    level = "High" if prob >= 0.6 else ("Medium" if prob >= 0.4 else "Low")
    return {
        "stage":         stage,
        "stage_label":   STAGE_LABEL[stage],
        "risk_prob":     prob,
        "risk_level":    level,
        "shap_top5":     shap_top5,
        "interventions": plans,
    }


def _stages_available(raw: dict) -> list:
    """Pick the stages that can run given which grades are present."""
    has_g1 = raw.get("G1") is not None
    has_g2 = raw.get("G2") is not None
    stages = ["pre"]
    if has_g1:
        stages.append("mid")
    if has_g1 and has_g2:
        stages.append("full")
    return stages


def predict_stages(raw: dict, stages: list = None) -> list:
    _load()
    if not stages:
        stages = _stages_available(raw)
    # filter to stages whose inputs are actually present
    available = set(_stages_available(raw))
    stages = [s for s in stages if s in available]
    return [predict_student(raw, s) for s in stages]
