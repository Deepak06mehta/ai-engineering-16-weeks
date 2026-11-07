.DEFAULT_GOAL := help
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := uv pip

.PHONY: help setup setup-models setup-apps check lint fmt test exercises up down logs clean

help:                     ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:                    ## Create the venv and install Weeks 1-6 dependencies
	uv venv --python 3.12
	$(PIP) install -r requirements/core.txt
	@echo "\nDone. Activate with:  source $(VENV)/bin/activate"

setup-models:             ## Add Weeks 7-8 dependencies (transformers, peft, trl)
	$(PIP) install -r requirements/models.txt

setup-apps:               ## Add Weeks 9-16 dependencies (fastapi, pgvector, langgraph)
	$(PIP) install -r requirements/apps.txt

check:                    ## Print the environment report (run this first, Week 1 Day 1)
	@$(PY) scripts/env_check.py 2>/dev/null || python3 scripts/env_check.py

lint:                     ## Lint and format
	$(VENV)/bin/ruff check . --fix && $(VENV)/bin/ruff format .

fmt: lint

test:                     ## Run the scaffold tests (should always be green)
	$(PY) -m pytest

exercises:                ## Run the exercise tests (RED until you implement them)
	$(PY) -m pytest -m exercise

up:                       ## Start Postgres+pgvector, Qdrant, Redis, Langfuse
	cd infra && docker compose up -d && docker compose ps

down:                     ## Stop the data services
	cd infra && docker compose down

logs:                     ## Tail the data service logs
	cd infra && docker compose logs -f

clean:                    ## Remove caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
