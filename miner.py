"""Poker44 (SN126) miner serving the trained chunk-level bot-detection ensemble.

Run with PYTHONPATH including both this directory and the Poker44-subnet repo:

    pm2 start .venv/bin/python --name p44_miner -- /home/sn126/p44model/miner.py \
        --netuid 126 --wallet.name <cold> --wallet.hotkey <hot> \
        --subtensor.network finney --axon.port <port> \
        --blacklist.force_validator_permit
"""

# NOTE: no `from __future__ import annotations` here — bittensor's axon.attach
# introspects the forward() signature at runtime and needs a real class, not a
# string annotation.

import sys
import time
from pathlib import Path
from typing import Tuple

MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIR))

import bittensor as bt
from bittensor.utils import axon_utils

# Validators sign a synapse once, then fan the large chunk payload out to many
# miners; slow fan-out lands fresh requests outside the default 4s+timeout
# nonce window ("Nonce is too old" rejections). Widen the window — the per-
# hotkey monotonic nonce check still blocks actual replays.
axon_utils.ALLOWED_DELTA = 300 * 1_000_000_000  # 300s in nanoseconds

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

from capture import capture_chunks
from predictor import ChunkPredictor


class ModelMiner(BaseMinerNeuron):
    """Serves calibrated bot-risk scores from the trained ensemble."""

    def __init__(self, config=None):
        super().__init__(config=config)
        self.predictor = ChunkPredictor()
        bt.logging.info(
            f"ChunkPredictor loaded | features={len(self.predictor.feature_cols)} "
            f"threshold={self.predictor.operating_threshold:.4f}"
        )

        self.model_manifest = build_local_model_manifest(
            repo_root=MODEL_DIR,
            implementation_files=[
                MODEL_DIR / "miner.py",
                MODEL_DIR / "predictor.py",
                MODEL_DIR / "features.py",
            ],
            defaults={
                "model_name": "p44-heroblend",
                "model_version": "1.0.0",
                "framework": "lightgbm+sklearn",
                "license": "MIT",
                # repo_url / repo_commit supplied via POKER44_MODEL_REPO_URL /
                # POKER44_MODEL_REPO_COMMIT env vars once the public repo exists.
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained exclusively on public Poker44 benchmark releases "
                    "(api.poker44.net/api/v1/benchmark), projected through the "
                    "validator payload view before feature extraction."
                ),
                "training_data_sources": ["poker44-public-benchmark"],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data."
                ),
            },
        )
        compliance = evaluate_manifest_compliance(self.model_manifest)
        bt.logging.info(
            f"Manifest status={compliance['status']} missing={compliance['missing_fields']} "
            f"violations={compliance['policy_violations']} digest={manifest_digest(self.model_manifest)}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        started = time.time()
        chunks = synapse.chunks or []
        scores = self.predictor.score_chunks(chunks)
        capture_chunks(chunks, scores)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(
            f"Scored {len(chunks)} chunks in {time.time() - started:.2f}s | "
            f"flagged={sum(synapse.predictions)}"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with ModelMiner() as miner:
        bt.logging.info("p44-heroblend miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
