#!/usr/bin/env bash
# Bring up a Hyperledger Fabric test network and deploy evidencecc.
#
# Blocker #6: Fabric is the highest-variance task in this project. Nothing here
# is on the critical path — the mock ledger backend is the default and `make
# demo` uses it. If this script fails, the demo is unaffected.
set -euo pipefail

CHANNEL="${CHANNEL:-ibvap}"
CC_NAME="${CC_NAME:-evidencecc}"
CC_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/fabric/chaincode/evidencecc"
FABRIC_DIR="${FABRIC_DIR:-$HOME/fabric-samples}"

say() { printf '\033[36m▸\033[0m %s\n' "$1"; }
die() { printf '\033[31m✗\033[0m %s\n' "$1" >&2; exit 1; }

command -v docker >/dev/null || die "docker is required"
command -v go >/dev/null || die "go is required to build the chaincode"

if [ ! -d "$FABRIC_DIR/test-network" ]; then
  die "fabric-samples not found at $FABRIC_DIR

Install it once:
  curl -sSLO https://raw.githubusercontent.com/hyperledger/fabric/main/scripts/install-fabric.sh
  chmod +x install-fabric.sh && ./install-fabric.sh docker samples binary

Then re-run: make fabric-up
(Everything Fabric is Apache-2.0 and free. See docs/MODELS.md.)"
fi

say "starting the Fabric test network with channel '$CHANNEL'"
cd "$FABRIC_DIR/test-network"
./network.sh down || true
./network.sh up createChannel -c "$CHANNEL" -ca

say "vendoring chaincode dependencies"
(cd "$CC_PATH" && go mod tidy && GOFLAGS=-mod=mod go mod vendor)

say "deploying $CC_NAME"
./network.sh deployCC -c "$CHANNEL" -ccn "$CC_NAME" -ccp "$CC_PATH" -ccl go

cat <<MSG

✓ Fabric is up.

  Switch the worker over by editing config/ledger.yaml:

      ledger:
        backend: fabric

  Then restart the anchor service. Alerts already anchored to the mock ledger
  stay verifiable against the mock; new batches go to the chain.

  To go back: set backend to mock. That is the whole rollback (§7.12).
MSG
