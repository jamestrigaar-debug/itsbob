# Developer shortcuts. `./install.sh` is the path for using itsbob;
# this is the path for changing it.
VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help install test lint check run gui clean
.DEFAULT_GOAL := help

help:            ## Show this help
	@grep -hE '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-10s\033[0m %s\n", $$1, $$2}'

$(PY):
	python3 -m venv $(VENV)

install: $(PY)   ## Install itsbob and dev dependencies
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[all,dev]"

test: install    ## Run the test suite (offline; no network)
	$(PY) -m pytest -q

check: test      ## Everything CI would run
	$(PY) -m itsbob doctor

run: install     ## Start an interactive chat
	$(PY) -m itsbob chat

gui: install     ## Start the browser interface
	$(PY) -m itsbob gui

clean:           ## Remove build artefacts and caches
	rm -rf $(VENV) build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
