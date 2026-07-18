"""Fail-safe capture of live validator payloads (inputs + our own scores).

Live queries carry no labels, so nothing here can be a supervised training
label. Captures are for domain-shift diagnosis of the benchmark->live gap.
Validators resend the same daily snapshot repeatedly, so chunks are deduped by
content hash. Every path is wrapped: a capture failure can never affect
serving.

Enabled via P44_CAPTURE=1. Output dir (P44_CAPTURE_DIR, default
/home/sn126/data/live_capture) lives outside the repo and is never pushed.

NOTE: these captures are validator-side evaluation inputs. Using them for
anything beyond diagnosis (e.g. training or calibration) requires truthfully
updating the model manifest's private-data attestation first.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, List

_LOCK = threading.Lock()
_DIR = Path(os.getenv("P44_CAPTURE_DIR", "/home/sn126/data/live_capture"))
_MAX_BYTES = int(os.getenv("P44_CAPTURE_MAX_BYTES", str(200 * 1024 * 1024)))
_ENABLED = os.getenv("P44_CAPTURE", "0").strip() in {"1", "true", "yes"}
_seen: set = set()
_loaded = False


def _chunk_key(chunk: List[dict]) -> str:
    blob = json.dumps(chunk, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _load_seen(path: Path) -> None:
    global _loaded
    try:
        if path.exists():
            with path.open() as f:
                for line in f:
                    try:
                        _seen.add(json.loads(line)["chunk_key"])
                    except Exception:
                        continue
    except Exception:
        pass
    _loaded = True


def capture_chunks(chunks: List[List[dict]], scores: List[float]) -> None:
    """Persist new unique chunks with our scores. Never raises."""
    if not _ENABLED:
        return
    try:
        with _LOCK:
            _DIR.mkdir(parents=True, exist_ok=True)
            path = _DIR / "live_chunks.jsonl"
            if not _loaded:
                _load_seen(path)
            if path.exists() and path.stat().st_size > _MAX_BYTES:
                return
            now = time.time()
            with path.open("a") as f:
                for chunk, score in zip(chunks or [], scores or []):
                    key = _chunk_key(chunk)
                    if key in _seen:
                        continue
                    _seen.add(key)
                    f.write(json.dumps({
                        "ts": now,
                        "chunk_key": key,
                        "our_score": float(score),
                        "n_hands": len(chunk),
                        "hands": chunk,
                    }, default=str) + "\n")
    except Exception:
        pass
