.PHONY: setup lint fmt test all

setup:
	uv sync --extra dev
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push

lint:
	uv run ruff format --check src/ tests/
	uv run ruff check src/ tests/
	uv run mypy src/

fmt:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

test:
	uv run pytest tests/unit/ -v

all: fmt lint test
