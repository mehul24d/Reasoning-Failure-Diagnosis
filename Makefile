.PHONY: install test pilot lint

install:
	pip install -e ".[dev]"

test:
	python3 -m pytest -q

pilot:
	bash scripts/run_pilot.sh

lint:
	ruff check src tests
