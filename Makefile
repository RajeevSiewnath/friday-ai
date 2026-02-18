.PHONY: dev
dev:
	uv run watchfiles "python main.py" .

.PHONY: lint
lint:
	flake8 .

.PHONY: test
test:
	pytest tests/
