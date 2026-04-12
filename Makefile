# Makefile

# ========================================
# Config
# ========================================
PACKAGE := friday
SRC_DIR := src
TEST_DIR := tests
PYTHON := python
UV := uv

# ========================================
# Dev: Run your main app with file watcher
# ========================================
.PHONY: dev
dev:
	$(UV) run watchfiles "$(PYTHON) -m $(PACKAGE).main" src

# ========================================
# Build: Install the package (editable)
# ========================================
.PHONY: build
build:
	$(UV) sync

# ========================================
# Lint: Check code style with flake8
# ========================================
.PHONY: lint
lint:
	flake8 $(SRC_DIR) $(TEST_DIR)

# ========================================
# Test: Run pytest (optional ARGS)
# ========================================
.PHONY: test
test:
	@echo "Running pytest with extra args: $(ARGS)"
	$(UV) run pytest $(TEST_DIR) $(ARGS)