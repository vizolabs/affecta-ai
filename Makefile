# AFFECTA AI Makefile

.PHONY: help install train evaluate test lint clean

# Default target
help:
	@echo "AFFECTA AI - Available commands:"
	@echo "  install    - Install dependencies"
	@echo "  train      - Train models"
	@echo "  evaluate   - Evaluate models"
	@echo "  test       - Run tests"
	@echo "  lint       - Run linter"
	@echo "  clean      - Clean generated files"

# Install dependencies
install:
	pip install -r requirements.txt

# Train expression classifier
train-expression:
	python -m ml.training.train_expression --backbone efficientnet_b3 --experiment-id EXP-007 --data-dir data/processed/rafdb

# Evaluate a trained model / experiment
evaluate:
	python ml/evaluate.py --experiment-dir experiments/EXP-005-RESNET50 --split test

# Run tests
test:
	python -m pytest tests/ -v

# Run linting
lint:
	flake8 src/ ml/ tests/
	mypy src/ ml/

# Format code
format:
	black src/ ml/ tests/

# Clean generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf coverage
