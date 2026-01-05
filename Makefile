# Makefile
.PHONY: install test lint format docs clean

# Installation
install:
	pip install -r requirements.txt
	pip install -r requirements-docs.txt
	pip install -e .

# Testing
test:
	pytest tests/ -v --cov=.

# Linting
lint:
	flake8 .
	black --check .
	isort --check-only .

# Formatting
format:
	black .
	isort .

# Documentation
docs:
	cd docs && make html

# Clean up
clean:
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .mypy_cache/

# Run the application
run:
	python main_v15_2.py

# Run with Docker
docker-run:
	docker-compose up --build

# Run tests in Docker
docker-test:
	docker-compose -f docker-compose.test.yml up --build --exit-code-from test

# Run linter in Docker
docker-lint:
	docker-compose -f docker-compose.lint.yml up --build