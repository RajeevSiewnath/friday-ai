.PHONY: dev
dev:
	uv run watchfiles "python src/main.py" src

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
	PYTHONPATH=src uv run pytest $(ARGS)
