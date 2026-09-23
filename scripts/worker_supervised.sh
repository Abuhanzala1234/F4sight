#!/usr/bin/env bash
# Run the analytics worker and restart it whenever it exits.
#
# A native crash inside ONNX Runtime (seen live: CUDA failure 716 on the GPU
# profile) calls std::terminate, which Python cannot catch -- the worker just
# dies, and without this every camera silently stops being analysed until a
# human notices. Video keeps playing either way (MediaMTX, P8); this only
# bounds how long analytics is gone to one model reload (~20 s).
#
#   scripts/worker_supervised.sh [profile] [site]
set -u
PROFILE="${1:-laptop}"
SITE="${2:-BOP-03}"
cd "$(dirname "$0")/.."

# The worker reads its secrets (PLATE_HMAC_KEY above all) from the process
# environment, not from .env the way the API does -- started without this,
# ANPR still reads plates but can never match one against the watchlist.
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi
export PYTHONPATH="worker/src:api/src"

while true; do
    echo "[supervisor] $(date -u +%FT%TZ) starting worker profile=$PROFILE site=$SITE"
    .venv/bin/python -m ibvap_worker --profile "$PROFILE" --site "$SITE"
    code=$?
    # 0 or SIGTERM/SIGINT means someone stopped it on purpose -- stop too.
    if [ "$code" -eq 0 ] || [ "$code" -eq 143 ] || [ "$code" -eq 130 ]; then
        echo "[supervisor] worker exited $code; not restarting"
        exit "$code"
    fi
    echo "[supervisor] $(date -u +%FT%TZ) worker died with exit $code; restarting in 3s"
    sleep 3
done
