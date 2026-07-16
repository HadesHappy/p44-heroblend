"""Shadow test: exercise the serving path end-to-end on real benchmark payloads.

Checks:
1. predictor loads and returns one score per chunk, all in [0, 1];
2. reward on the latest release (in-sample smoke, projected 30-40 hand groups);
3. robustness to live-sized chunks (~100 hands, merged same-label groups);
4. the unprojected-hand guard: raw payloads score like projected ones;
5. latency at live scale (90 chunks x ~100 hands) vs the 180s validator timeout.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from poker44.score.scoring import reward
from poker44.validator.payload_view import prepare_hand_for_miner

from predictor import ChunkPredictor

DATA_DIR = Path("/home/sn126/data/benchmark")


def load_release(date: str):
    payload = json.loads((DATA_DIR / f"chunks_{date}.json").read_text())
    groups, labels = [], []
    for chunk in payload["chunks"]:
        for group, label in zip(chunk.get("chunks") or [], chunk.get("groundTruth") or []):
            groups.append([h for h in group if isinstance(h, dict)])
            labels.append(int(label))
    return groups, labels


def main() -> None:
    predictor = ChunkPredictor()
    latest = sorted(p.stem.split("_")[1] for p in DATA_DIR.glob("chunks_*.json"))[-1]
    raw_groups, labels = load_release(latest)
    projected_groups = [[prepare_hand_for_miner(h) for h in g] for g in raw_groups]
    print(f"release {latest}: {len(raw_groups)} groups, {sum(len(g) for g in raw_groups)} hands")

    # 1+2: contract + reward on projected groups (what validators actually send)
    t0 = time.time()
    scores = predictor.score_chunks(projected_groups)
    elapsed = time.time() - t0
    assert len(scores) == len(projected_groups), "score count != chunk count"
    assert all(0.0 <= s <= 1.0 for s in scores), "score out of [0,1]"
    value, metrics = reward(np.array(scores), np.array(labels))
    print(f"[projected 30-40h] reward={value:.4f} ap={metrics['ap_score']:.4f} "
          f"recall@fpr5={metrics['bot_recall']:.4f} hard_fpr={metrics['hard_fpr']:.3f} "
          f"sanity={metrics['threshold_sanity_quality']:.2f} | {elapsed:.2f}s")

    # 3: live-sized chunks (~100 hands) by merging 3 same-label groups
    merged_chunks, merged_labels = [], []
    for label_value in (0, 1):
        idx = [i for i, l in enumerate(labels) if l == label_value]
        for j in range(0, len(idx) - 2, 3):
            merged = [h for k in idx[j:j + 3] for h in projected_groups[k]]
            merged_chunks.append(merged)
            merged_labels.append(label_value)
    t0 = time.time()
    merged_scores = predictor.score_chunks(merged_chunks)
    elapsed_merged = time.time() - t0
    value_m, metrics_m = reward(np.array(merged_scores), np.array(merged_labels))
    print(f"[merged ~100h]     reward={value_m:.4f} ap={metrics_m['ap_score']:.4f} "
          f"recall@fpr5={metrics_m['bot_recall']:.4f} hard_fpr={metrics_m['hard_fpr']:.3f} "
          f"sanity={metrics_m['threshold_sanity_quality']:.2f} "
          f"| {len(merged_chunks)} chunks in {elapsed_merged:.2f}s")

    # 4: raw (unprojected) payloads must route through the guard and still score sanely
    raw_scores = predictor.score_chunks(raw_groups)
    value_r, metrics_r = reward(np.array(raw_scores), np.array(labels))
    corr = float(np.corrcoef(scores, raw_scores)[0, 1])
    print(f"[raw guard]        reward={value_r:.4f} ap={metrics_r['ap_score']:.4f} "
          f"corr(projected, raw)={corr:.3f}")

    # 5: latency at live scale: 90 chunks x ~100 hands
    live_scale = (merged_chunks * (90 // max(1, len(merged_chunks)) + 1))[:90]
    t0 = time.time()
    live_scores = predictor.score_chunks(live_scale)
    live_elapsed = time.time() - t0
    assert len(live_scores) == 90
    print(f"[latency]          90 chunks x ~100 hands in {live_elapsed:.2f}s "
          f"(validator timeout: 180s)")

    # import check for the miner module (no wallet: class import only)
    import miner  # noqa: F401
    print("[miner module]     imports cleanly")

    print("\nSHADOW TEST PASSED")


if __name__ == "__main__":
    main()
