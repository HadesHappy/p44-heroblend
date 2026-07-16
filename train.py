"""Train and evaluate the Poker44 chunk classifier.

Protocol:
- strict temporal holdout: the most recent HOLDOUT_DAYS release dates are never
  used for training or calibration;
- GroupKFold by release date on the dev set for out-of-fold predictions;
- isotonic calibration fitted on OOF predictions;
- final metric is the subnet's own reward() composite, simulated per holdout
  release date (each date ~= one evaluation window).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from poker44.score.scoring import reward

FEATURES_PATH = Path("/home/sn126/data/features.parquet")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
META_COLS = {"label", "source_date", "split", "chunk_id", "group_index", "n_hands", "kind"}
HOLDOUT_DAYS = 10
N_FOLDS = 5
SEED = 7


def make_models():
    return {
        "lgbm": lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=SEED,
            verbose=-1,
        ),
        "histgb": HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=2.0,
            random_state=SEED,
        ),
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.2, max_iter=2000, random_state=SEED),
        ),
    }


def remap_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Monotone piecewise-linear remap sending `threshold` -> 0.5.

    Preserves ranking (AP, recall@FPR) while placing the subnet's hard 0.5
    threshold at a chosen operating point.
    """
    s = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    t = float(min(max(threshold, 1e-6), 1 - 1e-6))
    return np.where(s < t, 0.5 * s / t, 0.5 + 0.5 * (s - t) / (1.0 - t))


def pick_operating_threshold(oof_scores: np.ndarray, labels: np.ndarray, dates: np.ndarray) -> float:
    """Choose the remap threshold maximizing mean per-date reward on OOF preds."""
    candidates = np.quantile(oof_scores, np.linspace(0.30, 0.95, 40))
    best_t, best_val = 0.5, -1.0
    unique_dates = np.unique(dates)
    for t in candidates:
        remapped = remap_scores(oof_scores, t)
        vals = []
        for date in unique_dates:
            mask = dates == date
            value, _ = reward(remapped[mask], labels[mask])
            vals.append(value)
        # mean with a worst-case tiebreak: stability across windows matters
        score = float(np.mean(vals)) + 0.25 * float(np.min(vals))
        if score > best_val:
            best_val, best_t = score, float(t)
    return best_t


def composite_for_window(scores: np.ndarray, labels: np.ndarray) -> dict:
    value, metrics = reward(scores, labels)
    return {
        "reward": value,
        "ap": metrics["ap_score"],
        "recall_fpr5": metrics["bot_recall"],
        "sanity": metrics["threshold_sanity_quality"],
        "hard_fpr": metrics["hard_fpr"],
    }


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    dates = sorted(df["source_date"].unique())
    holdout_dates = set(dates[-HOLDOUT_DAYS:])
    dev = df[~df["source_date"].isin(holdout_dates)].reset_index(drop=True)
    hold = df[df["source_date"].isin(holdout_dates)].reset_index(drop=True)
    print(f"features: {len(feature_cols)} | dev: {len(dev)} groups over {len(dates) - HOLDOUT_DAYS} dates "
          f"| holdout: {len(hold)} groups over {HOLDOUT_DAYS} dates ({min(holdout_dates)}..{max(holdout_dates)})")

    X_dev = dev[feature_cols].to_numpy(dtype=np.float64)
    y_dev = dev["label"].to_numpy(dtype=int)
    X_hold = hold[feature_cols].to_numpy(dtype=np.float64)
    y_hold = hold["label"].to_numpy(dtype=int)
    groups = dev["source_date"].to_numpy()

    oof = {name: np.zeros(len(dev)) for name in make_models()}
    cv = GroupKFold(n_splits=N_FOLDS)
    for fold, (tr, va) in enumerate(cv.split(X_dev, y_dev, groups)):
        for name, model in make_models().items():
            model.fit(X_dev[tr], y_dev[tr])
            oof[name][va] = model.predict_proba(X_dev[va])[:, 1]
        print(f"fold {fold + 1}/{N_FOLDS} done", flush=True)

    print("\n=== OOF metrics (dev) ===")
    for name, preds in oof.items():
        print(f"{name:8s} AUC={roc_auc_score(y_dev, preds):.4f} AP={average_precision_score(y_dev, preds):.4f}")
    blend_oof = np.mean([oof["lgbm"], oof["histgb"], oof["logreg"]], axis=0)
    print(f"{'blend':8s} AUC={roc_auc_score(y_dev, blend_oof):.4f} AP={average_precision_score(y_dev, blend_oof):.4f}")

    # Refit every model on the full dev set, calibrate the blend on OOF preds.
    fitted = {}
    for name, model in make_models().items():
        model.fit(X_dev, y_dev)
        fitted[name] = model
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    calibrator.fit(blend_oof, y_dev)

    oof_cal = calibrator.predict(blend_oof)
    op_threshold = pick_operating_threshold(oof_cal, y_dev, groups)
    print(f"\noperating threshold (calibrated-score value remapped to 0.5): {op_threshold:.4f}")

    hold_raw = np.mean(
        [fitted[name].predict_proba(X_hold)[:, 1] for name in fitted], axis=0
    )
    hold_cal = remap_scores(calibrator.predict(hold_raw), op_threshold)

    print("\n=== Holdout (pooled) ===")
    print(f"AUC={roc_auc_score(y_hold, hold_cal):.4f} AP={average_precision_score(y_hold, hold_cal):.4f}")

    print("\n=== Holdout per-date composite (subnet reward fn) ===")
    kinds = sorted(hold["kind"].unique()) if "kind" in hold.columns else ["group"]
    per_date = []
    for kind in kinds:
        kind_vals = []
        for date in sorted(hold["source_date"].unique()):
            mask = ((hold["source_date"] == date) & (hold["kind"] == kind)).to_numpy()
            if mask.sum() < 4:
                continue
            res = composite_for_window(hold_cal[mask], y_hold[mask])
            kind_vals.append(res["reward"])
            print(f"{kind:8s} {date}: reward={res['reward']:.4f} ap={res['ap']:.4f} "
                  f"recall@fpr5={res['recall_fpr5']:.4f} sanity={res['sanity']:.2f} hard_fpr={res['hard_fpr']:.3f}")
        print(f"-> {kind}: mean={np.mean(kind_vals):.4f} min={np.min(kind_vals):.4f}\n")
        per_date.extend(kind_vals)
    print(f"mean composite over holdout windows (all kinds): {np.mean(per_date):.4f} "
          f"(min={np.min(per_date):.4f})")

    # feature importances from the lgbm member
    imp = pd.Series(fitted["lgbm"].feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\ntop 15 features (lgbm gain):")
    print(imp.head(15).to_string())

    ARTIFACT_DIR.mkdir(exist_ok=True)
    with open(ARTIFACT_DIR / "model.pkl", "wb") as f:
        pickle.dump({
            "models": fitted,
            "calibrator": calibrator,
            "feature_cols": feature_cols,
            "operating_threshold": op_threshold,
        }, f)
    (ARTIFACT_DIR / "train_meta.json").write_text(json.dumps({
        "holdout_dates": sorted(holdout_dates),
        "n_dev": len(dev),
        "n_holdout": len(hold),
        "operating_threshold": op_threshold,
        "mean_holdout_composite": float(np.mean(per_date)),
    }, indent=2))
    print(f"\nsaved artifacts to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
