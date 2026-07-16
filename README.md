# p44-heroblend

Poker bot-detection model for Bittensor subnet 126 (**Poker44**).

Scores chunks of anonymized poker hands with one bot-risk probability per chunk,
as required by the Poker44 `DetectionSynapse` contract.

## Approach

1. **Payload-view projection.** Every training hand is projected through the
   validator's own `prepare_hand_for_miner` (from the Poker44-subnet repo)
   before feature extraction, so the training distribution matches exactly what
   miners receive live (5-8 action windows, bucketed amounts, stripped blinds).
2. **Chunk-level features** (`features.py`). 102 size-invariant behavioral
   aggregates per chunk: hero-vs-table action-type shares, per-street action
   distribution, bet-size bucket histograms (snapping the payload noise back to
   the canonical buckets), pot trajectories, stack statistics, and
   hand-to-hand consistency measures.
3. **Ensemble** (`train_production.py`). LightGBM + HistGradientBoosting +
   logistic regression blend, out-of-fold isotonic calibration (GroupKFold by
   release date), and a monotone operating-point remap that places the subnet's
   hard 0.5 threshold at a ~5% false-positive operating point.
4. **Size augmentation.** Trained on both native 30-40 hand benchmark groups
   and merged ~100-hand same-label chunks to stay robust to live chunk sizes.

## Training data

Trained **exclusively on the public Poker44 training benchmark**
(`https://api.poker44.net/api/v1/benchmark`, daily releases). No
validator-only evaluation data is used. See `download_benchmark.py`.

## Reproduce

```bash
python download_benchmark.py      # fetch all public benchmark releases
python build_dataset.py           # project + extract features -> parquet
python train.py                   # temporal-holdout validation report
python train_production.py        # fit production artifact on all releases
python shadow_test.py             # end-to-end serving-path checks
```

Requires the [Poker44-subnet](https://github.com/Poker44/Poker44-subnet)
package on `PYTHONPATH` (for `payload_view` and the scoring reference) plus
`requirements.txt`.

## Serving

`miner.py` is the served neuron (see header for the pm2 invocation);
`predictor.py` loads `artifacts/production_model.pkl` and scores chunks.
`implementation_files` in the published model manifest cover `miner.py`,
`predictor.py`, and `features.py`.

## License

MIT
