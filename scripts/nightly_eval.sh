#!/usr/bin/env bash
# Nightly eval architecture run — idempotent wrapper around the four
# operational steps from the build-eval-architecture plan.
#
# Invoked by ~/.config/systemd/user/gutenborg-nightly-eval.timer at 04:00
# local time. Safe to re-run by hand at any point.

set -euo pipefail

PROJECT_DIR="${GUTENBORG_DIR:-/home/revelri/Dev/Pipeline/Building/gutenborg}"
COLLECTION="${GUTENBORG_COLLECTION:-gutenberg-deleuze-corpus}"
NOCTX_COLLECTION="${COLLECTION}-noctx"
LOG_DIR="${PROJECT_DIR}/data/eval/nightly"
STAMP="$(date +%Y-%m-%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${STAMP}.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== gutenborg nightly eval @ $(date -Iseconds) ==="
echo "project=$PROJECT_DIR  collection=$COLLECTION"

# ── 1. Bring up API + worker (no-op if already running).
echo "[1/4] docker compose up -d api worker"
docker compose up -d api worker

# Wait for API health before any exec calls.
for i in {1..60}; do
    if docker compose exec -T api python -c "import sys; sys.exit(0)" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

# ── 2. Mine gold datasets + augment precis.
echo "[2/4] mine_eval_golds + augment-precis"
docker compose exec -T api python scripts/mine_eval_golds.py \
    --collection "$COLLECTION" \
    --augment-precis data/eval/deleuze_precis_evolution.json

# ── 3. Build the sibling noctx collection once. Skip if already present.
echo "[3/4] reindex_noctx (idempotent; skips if dest already has chunks)"
NOCTX_COUNT="$(docker compose exec -T api python -c "
from core.chroma import get_collection
try:
    print(get_collection('${NOCTX_COLLECTION}').count())
except Exception:
    print(0)
" 2>/dev/null | tr -d '[:space:]')"
NOCTX_COUNT="${NOCTX_COUNT:-0}"
if [[ "$NOCTX_COUNT" -gt 0 ]]; then
    echo "  sibling '$NOCTX_COLLECTION' already has $NOCTX_COUNT chunks — skipping"
else
    docker compose exec -T api python scripts/reindex_noctx.py --source "$COLLECTION"
fi

# ── 4. Roll up the feature matrix + ablations.
echo "[4/4] eval_feature_matrix --ablate P0,P5,P7"
docker compose exec -T api python scripts/eval_feature_matrix.py \
    --ablate P0,P5,P7 \
    --collection "$COLLECTION" \
    --ctx-collection "$COLLECTION" \
    --output-md "data/eval/nightly/${STAMP}.md" \
    --output-json "data/eval/nightly/${STAMP}.json"

# Keep a rolling "latest" pointer for humans.
ln -sf "${STAMP}.md"   "${LOG_DIR}/latest.md"
ln -sf "${STAMP}.json" "${LOG_DIR}/latest.json"
ln -sf "${STAMP}.log"  "${LOG_DIR}/latest.log"

# Prune runs older than 30 days.
find "$LOG_DIR" -maxdepth 1 -type f -name '20*' -mtime +30 -delete 2>/dev/null || true

echo "=== done @ $(date -Iseconds) ==="
