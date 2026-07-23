.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@uv sync
	@uv run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running ty"
	@uv run ty check .

.PHONY: test
test: ## Test the code with pytest (fast unit tests, no model loaded)
	@echo "🚀 Testing code: Running pytest"
	@uv run --env-file .env python -m pytest -m "not integration and not model and not performance"

.PHONY: test-cov
test-cov: ## Run the fast unit tests with coverage (writes coverage.xml + junit.xml for CI/Codecov)
	@echo "🚀 Testing code: Running pytest with coverage"
	@uv run --env-file .env python -m pytest -m "not integration and not model and not performance" \
		--cov --cov-report=term-missing --cov-report=xml --junitxml=junit.xml

.PHONY: test-model
test-model: ## Run the unit tests that load a real Whisper model (slow, needs a GPU)
	@echo "🚀 Testing code: Running model-backed unit tests"
	@uv run --env-file .env python -m pytest -m "model and not integration and not performance"

.PHONY: test-all
test-all: ## Run every unit test, model-backed ones included
	@echo "🚀 Testing code: Running all unit tests"
	@uv run --env-file .env python -m pytest -m "not integration and not performance"


.PHONY: integration
integration: ## Run integration tests (excluding performance load tests)
	@echo "🚀 Testing code: Running integration tests"
	@uv run --env-file .env python -m pytest -m "integration and not performance"

.PHONY: performance
performance: ## Run E2E load test benchmark tool (records tracking metrics to eval_results/load_test_results.json)
	@echo "🚀 Running E2E load test benchmark tool"
	@uv run --env-file .env python tools/load_test.py

.PHONY: test-performance
test-performance: ## Run performance pytest suite
	@echo "🚀 Running performance pytest suite"
	@uv run --env-file .env python -m pytest -m performance

.PHONY: eval-quality
eval-quality: ## Run quality evaluation (WER, CER, BLEU) against the curated test suite (set WHISPER_EVAL_REPO to your whisper-evaluation checkout)
	@echo "🚀 Running quality evaluation against curated test suite"
	@uv run --env-file .env python tools/eval_quality.py





.PHONY: docker-build
docker-build: ## Build and tag the Docker image (set HF_TOKEN to bake the gated pyannote weights)
	$(eval VERSION := $(shell uv version --short))
	@echo "🐳 Building Docker image faster-whisper:v$(VERSION)"
	@HF_TOKEN="$(HF_TOKEN)" DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_TOKEN \
		-t quay.io/ktbs/fd-itbs/faster-whisper:v$(VERSION) .

.PHONY: docker-push
docker-push: ## Push the Docker image to quay.io/ktbs/fd-itbs/faster-whisper
	$(eval VERSION := $(shell uv version --short))
	@echo "🐳 Pushing Docker image faster-whisper:v$(VERSION)"
	@docker push quay.io/ktbs/fd-itbs/faster-whisper:v$(VERSION)

.PHONY: docker-up
docker-up: ## Build and run the Docker container
	@echo "🐳 Running docker compose"
	@docker compose up -d

.PHONY: docker-down
docker-down: ## Stop and remove the Docker container
	@echo "🐳 Stopping docker compose"
	@docker compose down

.PHONY: run
run: ## Run the BentoML service
	@echo "🚀 Running the BentoML service"
	@./scripts/run_varlock.sh uv run --env-file .env bentoml serve bentoml_faster_whisper.service:FasterWhisper -p 50001

.PHONY: diagnose-ui
diagnose-ui: ## Launch the Gradio UI comparing raw pyannote turns to the full pipeline output
	@echo "🚀 Launching the diagnosis UI"
	@uv run --env-file .env python tools/diagnose_ui.py

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
