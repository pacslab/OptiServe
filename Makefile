# OptiServe developer entry points.
#
# Everything here has a one-to-one counterpart in .github/workflows/ci.yml, so
# "green locally" and "green in CI" mean the same thing.

PYTHON      ?= python3
VENV        ?= .venv
BIN         := $(VENV)/bin
COMPOSE     ?= docker compose
IMAGE       ?= optiserve
PYTHON_VERSION ?= 3.11

.DEFAULT_GOAL := help
.PHONY: help venv install lint format typecheck test test-integration coverage \
        golden check deps-sync docker docker-dev docker-test compose-test \
        compose-integration example clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# Local environment
# --------------------------------------------------------------------------- #
venv:  ## Create the local virtualenv
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip

install: venv  ## Install the package with dev extras (editable)
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install --no-deps -e .

# --------------------------------------------------------------------------- #
# Static analysis
# --------------------------------------------------------------------------- #
lint:  ## ruff check + format check
	$(BIN)/ruff check optiserve tests examples scripts
	$(BIN)/ruff format --check optiserve tests examples scripts

format:  ## Apply ruff's formatter and autofixes
	$(BIN)/ruff check --fix optiserve tests examples scripts
	$(BIN)/ruff format optiserve tests examples scripts

typecheck:  ## mypy over the strictly-typed layers
	$(BIN)/mypy optiserve

deps-sync:  ## Assert requirements*.txt still mirror pyproject.toml
	$(BIN)/python scripts/check_requirements_sync.py

# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
test:  ## Offline suite (unit + golden); no AWS, no network
	$(BIN)/pytest -q -m "not integration"

golden:  ## Only the golden-master regressions
	$(BIN)/pytest -q -m golden

test-integration:  ## AWS adapter tests against in-process moto
	$(BIN)/pytest -q -m integration

coverage:  ## Offline suite with a coverage report
	$(BIN)/pytest -q -m "not integration" --cov=optiserve --cov-report=term-missing --cov-report=html
	@echo "HTML report: htmlcov/index.html"

check: lint typecheck deps-sync test  ## Everything CI runs on a pull request

# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
docker:  ## Build the production runtime image
	docker build --target runtime --build-arg PYTHON_VERSION=$(PYTHON_VERSION) -t $(IMAGE):latest .

docker-dev:  ## Build the dev/test image
	docker build --target dev --build-arg PYTHON_VERSION=$(PYTHON_VERSION) -t $(IMAGE):dev .

docker-test: docker-dev  ## Run the offline suite inside the dev image
	docker run --rm $(IMAGE):dev pytest -q -m "not integration"

compose-test:  ## Offline suite via docker compose
	$(COMPOSE) run --rm tests

compose-integration:  ## Integration suite against the moto server container
	$(COMPOSE) run --rm integration
	$(COMPOSE) down -v

# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
example:  ## Run the offline optimization example
	$(BIN)/python examples/optimize_workflow.py

clean:  ## Remove build/test caches
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
