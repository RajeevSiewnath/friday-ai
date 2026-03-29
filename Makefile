.PHONY: dev-legacy
dev-legacy:
	uv run watchfiles "python legacy/src/main.py" legacy/src

.PHONY: eval-one
eval-one:
	uv run evaluation/evaluator.py ${N}

.PHONY: eval
eval:
	uv run evaluation/browser_evaluator.py

.PHONY: lint
lint:
	flake8 .

.PHONY: test
test:
	@echo "Running pytest with extra args: $(ARGS)"
	PYTHONPATH=src uv run pytest tests $(ARGS)
