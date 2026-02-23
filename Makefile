.PHONY: dev
dev:
	uv run watchfiles "python main.py" data main.py

.PHONY: lint
lint:
	flake8 .

.PHONY: test
test:
	pytest tests/
