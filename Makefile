PYTHON?=python

.venv:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade setuptools pip
	.venv/bin/python -m pip install pip-tools
	touch .venv

requirements/base.txt: requirements/base.in .venv
	.venv/bin/python -m piptools compile requirements/base.in -o requirements/base.txt
	touch requirements/dev.txt

requirements/dev.txt: requirements/dev.in .venv
	.venv/bin/python -m piptools compile requirements/dev.in -o requirements/dev.txt
	touch requirements/dev.txt

compile: requirements/base.txt requirements/dev.txt

.install_requires:
	.venv/bin/python -m pip install -r requirements/base.txt
	.venv/bin/python -m pip install -r requirements/dev.txt
	.venv/bin/python -m pip install -e .
	pre-commit install

test: 
	.venv/bin/python -m pytest -m pytest tests --cov=rl_cbf --cov-report=xml

install: .venv compile .install_requires

all: install test

.PHONY: .install_requires compile install test all