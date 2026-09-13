# DRISHTI-BOP — every command a human needs. See CLAUDE.md "Common tasks".
.DEFAULT_GOAL := help
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

PY        := python3
VENV      := .venv
BIN       := $(VENV)/bin
COMPOSE   := docker compose
PROFILE   ?= laptop
SITE      ?= BOP-03

export PYTHONPATH := worker/src:api/src

# ---------------------------------------------------------------- help
.PHONY: help
help:  ## Show this help
	@echo "DRISHTI-BOP — SIH 2026 · PS 26187 · Team SW-73"; echo
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup
.PHONY: venv
venv: $(BIN)/activate  ## Create the virtualenv
$(BIN)/activate:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip wheel

.PHONY: install
install: venv  ## Install worker + api in editable mode with dev extras
	$(BIN)/pip install -e "worker[dev]" -e "api[dev]"
	@cd dashboard && npm install --silent
	@echo "✓ dependencies installed"

.PHONY: env
env: .env  ## Generate .env with random secrets if absent
.env:
	@cp .env.example .env
	@for k in JWT_SECRET PLATE_HMAC_KEY EVIDENCE_ENC_KEY; do \
	  v=$$(openssl rand -hex 32); \
	  sed -i.bak "s|^$$k=.*|$$k=$$v|" .env; \
	done; rm -f .env.bak
	@echo "✓ .env created with fresh secrets"

# ---------------------------------------------------------------- infra
.PHONY: up
up: env  ## Start infra: postgres, minio, redis, mediamtx, fixture streams
	$(COMPOSE) up -d postgres minio redis mediamtx minio-init fixture-streamer
	@echo "waiting for health…"
	@$(COMPOSE) ps
	@echo "✓ infra up  ·  minio console http://localhost:9001  ·  hls http://localhost:8888"

.PHONY: down
down:  ## Stop infra (keeps volumes)
	$(COMPOSE) down

.PHONY: nuke
nuke:  ## Stop infra and DELETE all data. Asks first.
	@read -p "Delete all postgres + minio data? [y/N] " a; [ "$$a" = "y" ] || exit 1
	$(COMPOSE) down -v

.PHONY: logs
logs:  ## Tail infra logs
	$(COMPOSE) logs -f --tail=80

# ---------------------------------------------------------------- data
.PHONY: migrate
migrate:  ## alembic upgrade head
	$(BIN)/alembic -c api/alembic.ini upgrade head

.PHONY: seed
seed:  ## Demo sites, cameras, zones, users, watchlist
	$(BIN)/python -m drishti_api.seed --profile $(PROFILE) --site $(SITE)

.PHONY: models
models:  ## Download + export every free model into models/ (one time, ~166 MB)
	$(BIN)/python scripts/fetch_models.py $(if $(FORCE),--force,)

.PHONY: fixtures
fixtures:  ## Fetch/generate sample MP4s for the fixture RTSP streams
	$(BIN)/python scripts/make_fixtures.py

# ---------------------------------------------------------------- run
.PHONY: worker
worker:  ## Run the analytics worker against fixture streams
	$(BIN)/python -m drishti_worker --profile $(PROFILE) --site $(SITE)

.PHONY: api
api:  ## Run FastAPI with reload
	$(BIN)/uvicorn drishti_api.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: anchor
anchor:  ## Run the ledger anchoring service
	$(BIN)/python -m drishti_worker.anchor_service

.PHONY: dash
dash:  ## Vite dev server
	cd dashboard && npm run dev

# ---------------------------------------------------------------- quality
.PHONY: test
test: test-py test-ts  ## pytest + vitest

.PHONY: test-py
test-py:
	$(BIN)/pytest worker/tests api/tests -q

.PHONY: test-ts
test-ts:
	cd dashboard && npm run test -- --run

.PHONY: lint
lint:  ## ruff + black --check + mypy + tsc
	$(BIN)/ruff check worker api scripts
	$(BIN)/black --check worker api scripts
	$(BIN)/mypy worker/src api/src
	cd dashboard && npx tsc --noEmit

.PHONY: fmt
fmt:  ## Autoformat everything
	$(BIN)/ruff check --fix worker api scripts
	$(BIN)/black worker api scripts
	cd dashboard && npm run format

# ---------------------------------------------------------------- measure
.PHONY: bench
bench:  ## Throughput/latency benchmark → docs/BENCH.md
	$(BIN)/python scripts/bench.py --profile $(PROFILE) --out docs/BENCH.md

.PHONY: eval
eval:  ## Run the evaluation set → metrics table
	$(BIN)/python scripts/evaluate.py --clips eval/clips --labels eval/labels.jsonl

# ---------------------------------------------------------------- demo
.PHONY: demo
demo: env up  ## THE DEMO-DAY COMMAND. Cold start, offline, everything.
	@echo "── 1/6 migrations ──"      && $(MAKE) migrate
	@echo "── 2/6 seed ──"            && $(MAKE) seed
	@echo "── 3/6 verify models ──"   && $(BIN)/python scripts/fetch_models.py --verify-only
	@echo "── 4/6 fixtures ──"        && $(MAKE) fixtures
	@echo "── 5/6 dashboard build ──" && cd dashboard && npm run build
	@echo "── 6/6 starting worker + api + anchor ──"
	@$(BIN)/python scripts/demo.py --profile $(PROFILE) --site $(SITE)

# ---------------------------------------------------------------- ledger
.PHONY: fabric-up
fabric-up:  ## Hyperledger Fabric test network + chaincode deploy
	./scripts/fabric_up.sh

.PHONY: fabric-down
fabric-down:
	./scripts/fabric_down.sh
