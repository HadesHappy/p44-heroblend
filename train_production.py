"""Fit the production model on ALL benchmark releases.

Same protocol as train.py (whose temporal holdout validated it), but uses every
release: OOF predictions via GroupKFold by date fit the calibrator and the
operating threshold, then all models refit on the full dataset.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from train import META_COLS, FEATURES_PATH, ARTIFACT_DIR, make_models, pick_operating_threshold, remap_scores
from poker44.score.scoring import reward

N_FOLDS = 10


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=int)
    groups = df["source_date"].to_numpy()
    print(f"production fit: {len(df)} groups, {len(feature_cols)} features, {len(set(groups))} dates")

    oof = {name: np.zeros(len(df)) for name in make_models()}
    cv = GroupKFold(n_splits=N_FOLDS)
    for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
        for name, model in make_models().items():
            model.fit(X[tr], y[tr])
            oof[name][va] = model.predict_proba(X[va])[:, 1]
        print(f"fold {fold + 1}/{N_FOLDS} done", flush=True)

    blend_oof = np.mean([oof[name] for name in oof], axis=0)
    print(f"OOF blend AUC={roc_auc_score(y, blend_oof):.4f} AP={average_precision_score(y, blend_oof):.4f}")

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    calibrator.fit(blend_oof, y)
    oof_cal = calibrator.predict(blend_oof)
    op_threshold = pick_operating_threshold(oof_cal, y, groups)

    remapped = remap_scores(oof_cal, op_threshold)
    per_date = []
    for date in sorted(set(groups)):
        mask = groups == date
        value, _ = reward(remapped[mask], y[mask])
        per_date.append(value)
    print(f"operating threshold: {op_threshold:.4f}")
    print(f"OOF per-date composite: mean={np.mean(per_date):.4f} min={np.min(per_date):.4f}")

    fitted = {}
    for name, model in make_models().items():
        model.fit(X, y)
        fitted[name] = model

    ARTIFACT_DIR.mkdir(exist_ok=True)
    out = ARTIFACT_DIR / "production_model.pkl"
    with open(out, "wb") as f:
        pickle.dump({
            "models": fitted,
            "calibrator": calibrator,
            "feature_cols": feature_cols,
            "operating_threshold": op_threshold,
        }, f)
    (ARTIFACT_DIR / "production_meta.json").write_text(json.dumps({
        "n_groups": len(df),
        "n_dates": len(set(groups)),
        "operating_threshold": op_threshold,
        "oof_mean_composite": float(np.mean(per_date)),
        "oof_min_composite": float(np.min(per_date)),
    }, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
