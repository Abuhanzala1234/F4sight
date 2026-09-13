#!/usr/bin/env bash
set -euo pipefail
FABRIC_DIR="${FABRIC_DIR:-$HOME/fabric-samples}"
[ -d "$FABRIC_DIR/test-network" ] || { echo "nothing to tear down"; exit 0; }
cd "$FABRIC_DIR/test-network" && ./network.sh down
echo "✓ Fabric network down. Set ledger.backend=mock in config/ledger.yaml."
