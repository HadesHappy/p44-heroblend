#!/bin/bash
# Daily Poker44 retrain: fetch new benchmark data, retrain, validate, swap.
#
# Runs shortly before the 12:00 UTC round flip so each new round's first
# evaluation (which locks the round's model per v2.1 rules) sees a model
# trained on that morning's 00:05 UTC benchmark release.
#
# The swap is gated on shadow_test.py: if training or validation fails, the
# previous artifact is restored and the running miner is left untouched.

set -euo pipefail

cd /home/sn126/p44model
PY=/home/sn126/Poker44-subnet/.venv/bin/python
ARTIFACT=artifacts/production_model.pkl

echo "=== retrain start $(date -u +%FT%TZ) ==="

$PY download_benchmark.py
$PY build_dataset.py

cp "$ARTIFACT" "$ARTIFACT.bak"

if $PY train_production.py && $PY shadow_test.py; then
    rm -f "$ARTIFACT.bak"

    git add artifacts/production_model.pkl artifacts/production_meta.json
    if git -c user.name="HadesHappy" -c user.email="happyhades123@gmail.com" \
         commit -m "daily retrain $(date -u +%F)"; then
        git push origin main
        NEW_COMMIT=$(git rev-parse HEAD)
        sed -i "s/POKER44_MODEL_REPO_COMMIT: \"[0-9a-f]*\"/POKER44_MODEL_REPO_COMMIT: \"$NEW_COMMIT\"/" \
            ecosystem.config.js
        echo "pushed retrained artifact, manifest commit -> $NEW_COMMIT"
    else
        echo "no artifact changes to commit"
    fi

    pm2 restart ecosystem.config.js --update-env
    echo "=== retrain OK $(date -u +%FT%TZ) ==="
else
    echo "!!! RETRAIN OR VALIDATION FAILED — restoring previous artifact, miner untouched"
    mv "$ARTIFACT.bak" "$ARTIFACT"
    git checkout -- artifacts/ 2>/dev/null || true
    exit 1
fi
