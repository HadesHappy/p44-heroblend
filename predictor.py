"""Chunk scorer serving the trained Poker44 bot-detection ensemble."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from features import add_batch_rank_features, extract_chunk_features

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "production_model.pkl"
NEUTRAL_SCORE = 0.45  # returned for empty/unparseable chunks: below the 0.5 gate

# The subnet reward uses 0.5-crossings only through the threshold-sanity gate
# (needs >=1 true positive and FPR <= 10%); hard recall is NOT rewarded, so
# extra flags are pure downside risk. Flag exactly TOP_K chunks per request by
# rank: immune to calibration drift (live flag counts previously swung 88 ->
# 55 -> 7 across retrains), >=1 TP virtually certain, and worst-case FPR stays
# under the cliff whenever the window holds >= 10*TOP_K humans.
TOP_K_FLAGS = 5


def _looks_unprojected(hand: Dict[str, Any]) -> bool:
    """Detect hands that did not pass through the validator's payload view.

    Every live validator projects hands via prepare_hand_for_miner before
    sending (blinds stripped, actions subsampled to a 5-8 window). If a hand
    still carries blind posts or an oversized action list, it came from some
    other path and must be projected locally to match the training
    distribution.
    """
    actions = hand.get("actions") or []
    if len(actions) > 12:
        return True
    for action in actions:
        if str(action.get("action_type", "")) in {"small_blind", "big_blind", "ante"}:
            return True
    return False


# Weight of the supervised ranking vs the human-anomaly ranking when ordering
# chunks. Benchmark bots are synthetic, live bots are real: the anomaly model
# (fitted on real human examples only) is bot-type independent, so it carries
# substantial weight even though it scores lower on the synthetic benchmark.
SUPERVISED_RANK_WEIGHT = 0.6

class ChunkPredictor:
    def __init__(self, artifact_path: Path = ARTIFACT_PATH):
        with open(artifact_path, "rb") as f:
            bundle = pickle.load(f)
        self.models = bundle["models"]
        self.calibrator = bundle["calibrator"]
        self.feature_cols: List[str] = bundle["feature_cols"]
        self.operating_threshold: float = float(bundle["operating_threshold"])
        self.anomaly_scaler = bundle.get("anomaly_scaler")
        self.anomaly_iso = bundle.get("anomaly_iso")
        self.anomaly_knn = bundle.get("anomaly_knn")

    def _remap(self, scores: np.ndarray) -> np.ndarray:
        t = min(max(self.operating_threshold, 1e-6), 1 - 1e-6)
        s = np.clip(scores, 0.0, 1.0)
        return np.where(s < t, 0.5 * s / t, 0.5 + 0.5 * (s - t) / (1.0 - t))

    def _normalize_chunk(self, chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        hands = [h for h in chunk if isinstance(h, dict)]
        if any(_looks_unprojected(h) for h in hands):
            from poker44.validator.payload_view import prepare_hand_for_miner

            hands = [prepare_hand_for_miner(h) for h in hands]
        return hands

    def score_chunks(self, chunks: List[List[Dict[str, Any]]]) -> List[float]:
        """Return one calibrated bot-risk score per chunk, in input order."""
        feature_rows: List[Dict[str, float]] = []
        valid_index: List[int] = []
        for i, chunk in enumerate(chunks or []):
            try:
                feats = extract_chunk_features(self._normalize_chunk(chunk or []))
            except Exception:
                feats = {}
            if feats:
                feature_rows.append(feats)
                valid_index.append(i)

        scores = [NEUTRAL_SCORE] * len(chunks or [])
        if not feature_rows:
            return scores

        add_batch_rank_features(feature_rows)
        X = np.array(
            [[row.get(col, 0.0) for col in self.feature_cols] for row in feature_rows],
            dtype=np.float64,
        )
        raw = np.mean([m.predict_proba(X)[:, 1] for m in self.models.values()], axis=0)
        # Isotonic calibration is a step function: chunks collapse into a
        # handful of tied score values, which destroys rank metrics (validators
        # sort our scores) and defeats the flag cap. Blend in a sliver of the
        # continuous raw ensemble score so ordering is strict while calibration
        # stays essentially intact.
        cal = 0.98 * self.calibrator.predict(raw) + 0.02 * raw
        final = self._reorder_by_blended_rank(self._remap(cal), raw, X)
        final = self._flag_top_k(final)
        for i, s in zip(valid_index, final):
            scores[i] = float(np.round(np.clip(s, 0.0, 1.0), 6))
        return scores

    @staticmethod
    def _flag_top_k(scores: np.ndarray, k: int = TOP_K_FLAGS) -> np.ndarray:
        """Monotone remap so exactly min(k, max(1, n//8)) chunks score >= 0.5.

        Rank metrics are unaffected; only the 0.5-crossing set changes.
        """
        s = np.asarray(scores, dtype=float)
        n = s.size
        if n < 2:
            return s
        k_eff = int(min(k, max(1, n // 8)))
        order = np.argsort(-s, kind="stable")
        flip = np.empty(n, dtype=float)
        top = order[:k_eff]
        rest = order[k_eff:]
        # spread top set over (0.5, 1.0] and the rest over [0.0, 0.5),
        # preserving relative order within each band
        for band, lo, hi in ((top, 0.5 + 1e-6, 1.0), (rest, 0.0, 0.5 - 1e-6)):
            m = band.size
            if m == 0:
                continue
            vals = s[band]
            span = vals.max() - vals.min()
            frac = (vals - vals.min()) / span if span > 0 else np.linspace(1, 0, m)
            flip[band] = lo + frac * (hi - lo)
        return flip

    def _anomaly_scores(self, X: np.ndarray) -> "np.ndarray | None":
        """Bot-risk as distance from real human play (bot-type independent)."""
        if self.anomaly_scaler is None or self.anomaly_iso is None:
            return None
        Xs = self.anomaly_scaler.transform(X)
        iso = -self.anomaly_iso.score_samples(Xs)
        knn_dist, _ = self.anomaly_knn.kneighbors(Xs)
        knn = knn_dist.mean(axis=1)
        # combine as within-request ranks so scales don't matter
        n = len(Xs)
        if n < 2:
            return None
        rank = lambda v: np.argsort(np.argsort(v)) / (n - 1)
        return 0.5 * rank(iso) + 0.5 * rank(knn)

    def _reorder_by_blended_rank(
        self, final: np.ndarray, raw: np.ndarray, X: np.ndarray
    ) -> np.ndarray:
        """Keep the calibrated score distribution (flag counts, cap and sanity
        behavior unchanged) but reassign scores by a blended ranking of the
        supervised score and the human-anomaly score."""
        n = len(final)
        anomaly = self._anomaly_scores(X) if n >= 4 else None
        if anomaly is None:
            return final
        rank = lambda v: np.argsort(np.argsort(v)) / (n - 1)
        blended = SUPERVISED_RANK_WEIGHT * rank(raw) + (
            1.0 - SUPERVISED_RANK_WEIGHT
        ) * anomaly
        order = np.argsort(blended)  # ascending blended risk
        reordered = np.empty(n, dtype=float)
        reordered[order] = np.sort(final)  # assign sorted scores by blended order
        return reordered
