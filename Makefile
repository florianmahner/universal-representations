PYTHON ?= python3
EXTRAS ?= test

.PHONY: install check

install: .venv/bin/python
	.venv/bin/python -m pip install -e ".[${EXTRAS}]"

.venv/bin/python:
	$(PYTHON) -m venv .venv

check:
	.venv/bin/python -m pytest -q
