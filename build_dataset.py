"""Build the training feature table from downloaded benchmark releases.

Projects every hand through the validator's own prepare_hand_for_miner so the
training distribution matches what miners see live, then extracts chunk-level
features and writes a single parquet file.
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import add_batch_rank_features, extract_chunk_features  # noqa: E402

from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402

DATA_DIR = Path("/home/sn126/data/benchmark")
OUT_PATH = Path("/home/sn126/data/features.parquet")


MERGE_SIZE = 3  # same-label groups merged to simulate live ~100-hand chunks


def main() -> None:
    rows = []
    files = sorted(glob.glob(str(DATA_DIR / "chunks_*.json")))
    started = time.time()
    for file_index, path in enumerate(files, start=1):
        payload = json.loads(Path(path).read_text())
        date_groups = []  # (projected_hands, label, split, chunk_id, group_index)
        for chunk in payload["chunks"]:
            groups = chunk.get("chunks") or []
            labels = chunk.get("groundTruth") or []
            split = str(chunk.get("split") or "")
            for group_index, (group, label) in enumerate(zip(groups, labels)):
                projected = [prepare_hand_for_miner(h) for h in group if isinstance(h, dict)]
                date_groups.append(
                    (projected, int(label), split, str(chunk.get("chunkId") or ""), group_index)
                )

        source_date = str(payload["chunks"][0].get("sourceDate") or "") if payload["chunks"] else ""

        # original 30-40 hand groups; batch-rank features computed over the
        # date's full group set (mirrors one live validator request)
        group_feats, group_meta = [], []
        for projected, label, split, chunk_id, group_index in date_groups:
            feats = extract_chunk_features(projected)
            if not feats:
                continue
            group_feats.append(feats)
            group_meta.append(dict(
                label=label, source_date=source_date, split=split, chunk_id=chunk_id,
                group_index=group_index, n_hands=len(projected), kind="group",
            ))
        add_batch_rank_features(group_feats)
        for feats, meta in zip(group_feats, group_meta):
            feats.update(meta)
            rows.append(feats)

        # merged same-label triples: live-sized ~100 hand chunks
        merged_feats, merged_meta = [], []
        for label_value in (0, 1):
            same = [g for g in date_groups if g[1] == label_value]
            for j in range(0, len(same) - MERGE_SIZE + 1, MERGE_SIZE):
                merged_hands = [h for g in same[j:j + MERGE_SIZE] for h in g[0]]
                feats = extract_chunk_features(merged_hands)
                if not feats:
                    continue
                merged_feats.append(feats)
                merged_meta.append(dict(
                    label=label_value, source_date=source_date, split="merged",
                    chunk_id=same[j][3], group_index=j, n_hands=len(merged_hands),
                    kind="merged3",
                ))
        add_batch_rank_features(merged_feats)
        for feats, meta in zip(merged_feats, merged_meta):
            feats.update(meta)
            rows.append(feats)

        print(
            f"[{file_index}/{len(files)}] {Path(path).name}: {len(rows)} total rows "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} | shape={df.shape} | bots={int(df.label.sum())} humans={int((1 - df.label).sum())}")


if __name__ == "__main__":
    main()
