.PHONY: help install lint typecheck test ingest-all train-all eval-all eval-baselines

AIRPORT_YAMLS := $(wildcard config/airports/*.yaml)
AIRPORTS := $(basename $(notdir $(AIRPORT_YAMLS)))

help:
	@echo "Targets:"
	@echo "  install         uv sync with dev extras"
	@echo "  lint            ruff check"
	@echo "  typecheck       mypy --strict"
	@echo "  test            pytest"
	@echo "  ingest-all      ingest METAR + HRRR for every configured airport"
	@echo "  train-all       train a model per configured airport"
	@echo "  eval-all        evaluate every configured airport"
	@echo "  eval-baselines  run baseline-only evaluation across airports"
	@echo ""
	@echo "Configured airports: $(AIRPORTS)"

install:
	uv sync --extra dev

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy --strict src

test:
	uv run pytest

ingest-all:
	@for a in $(AIRPORTS); do \
		echo "==> $$a"; \
		uv run wind-forecast ingest-metar --airport $$a || exit 1; \
		uv run wind-forecast ingest-hrrr  --airport $$a || exit 1; \
	done

train-all:
	@for a in $(AIRPORTS); do \
		echo "==> training $$a"; \
		uv run wind-forecast train --airport $$a || exit 1; \
	done

eval-all:
	@for a in $(AIRPORTS); do \
		echo "==> evaluating $$a"; \
		uv run wind-forecast eval --airport $$a || exit 1; \
	done

eval-baselines:
	@for a in $(AIRPORTS); do \
		echo "==> baselines $$a"; \
		uv run wind-forecast eval --airport $$a --baseline all || exit 1; \
	done
