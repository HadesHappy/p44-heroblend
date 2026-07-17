"""Chunk scorer serving the trained Poker44 bot-detection ensemble."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from features import extract_chunk_features

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "production_model.pkl"
NEUTRAL_SCORE = 0.45  # returned for empty/unparseable chunks: below the 0.5 gate

# Live payloads are distribution-shifted vs the public benchmark: scores
# compress upward and the 0.5 threshold over-flags (observed 84-89% of live
# chunks vs ~45-73% on benchmark data at any plausible class mix). The subnet
# reward's threshold-sanity term punishes human false positives at 0.5, so cap
# the flagged fraction per request with an order-preserving squeeze: rank
# metrics (AP, recall@FPR) are unaffected, false positives can only decrease.
FLAG_CAP_FRACTION = 0.65


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


class ChunkPredictor:
    def __init__(self, artifact_path: Path = ARTIFACT_PATH):
        with open(artifact_path, "rb") as f:
            bundle = pickle.load(f)
        self.models = bundle["models"]
        self.calibrator = bundle["calibrator"]
        self.feature_cols: List[str] = bundle["feature_cols"]
        self.operating_threshold: float = float(bundle["operating_threshold"])

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

        X = np.array(
            [[row.get(col, 0.0) for col in self.feature_cols] for row in feature_rows],
            dtype=np.float64,
        )
        raw = np.mean([m.predict_proba(X)[:, 1] for m in self.models.values()], axis=0)
        final = self._cap_flag_fraction(self._remap(self.calibrator.predict(raw)))
        for i, s in zip(valid_index, final):
            scores[i] = float(np.round(np.clip(s, 0.0, 1.0), 6))
        return scores

    @staticmethod
    def _cap_flag_fraction(scores: np.ndarray, cap: float = FLAG_CAP_FRACTION) -> np.ndarray:
        """If more than `cap` of scores cross 0.5, squeeze so only the top
        `cap` fraction does. Monotone (rank-preserving); no-op otherwise."""
        s = np.asarray(scores, dtype=float)
        if s.size < 4 or float(np.mean(s >= 0.5)) <= cap:
            return s
        cutoff = float(np.quantile(s, 1.0 - cap))
        cutoff = min(max(cutoff, 1e-6), 1 - 1e-6)
        return np.where(
            s < cutoff, 0.5 * s / cutoff, 0.5 + 0.5 * (s - cutoff) / (1.0 - cutoff)
        )
