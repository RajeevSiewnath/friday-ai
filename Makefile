.PHONY: dev
dev:
	uv run watchfiles "python src/main.py" src

.PHONY: lint
lint:
	flake8 .

.PHONY: test
test:
	pytest tests/
