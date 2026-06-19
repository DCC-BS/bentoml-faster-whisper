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
test: ## Test the code with pytest (excludes integration tests)
	@echo "🚀 Testing code: Running pytest"
	@uv run --env-file .env python -m pytest -m "not integration"

.PHONY: integration
integration: ## Run integration tests (requires a running service)
	@echo "🚀 Testing code: Running integration tests"
	@uv run --env-file .env python -m pytest -m integration

.PHONY: docker-build
docker-build: ## Build and tag the Docker image for quay.io/ktbs/fd-itbs/faster-whisper
	$(eval VERSION := $(shell uv version --short))
	@echo "🐳 Building Docker image faster-whisper:v$(VERSION)"
	@docker build -t quay.io/ktbs/fd-itbs/faster-whisper:v$(VERSION) .

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
	@uv run --env-file .env bentoml serve service:FasterWhisper -p 50001

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
