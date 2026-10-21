.PHONY: install test test-fast lint legacy-pilot

install:
	pip install -e ".[dev]"

test:
	python3 -m pytest -q

test-fast:
	python3 -m pytest -q -m "not legacy"

lint:
	ruff check src tests

# Archived SAE phase-segmentation pilot (see src/rfd/legacy/README.md).
legacy-pilot:
	bash scripts/legacy/run_pilot.sh
