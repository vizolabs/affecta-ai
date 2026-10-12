# Human Facial Emotion Detection Makefile
# API and frontend both expect port 8765 (see frontend/.env.local).

.PHONY: help install setup train train-expression evaluate test lint format clean webcam ensemble serve serve-web run report

# Default target
help:
	@echo "Human Facial Emotion Detection - Available commands:"
	@echo "  setup      - Python deps + frontend deps + production build"
	@echo "  install    - Install Python dependencies"
	@echo "  run        - Start API (8765) + web app (3000) together"
	@echo "  serve      - Local API server only (FastAPI + SQLite, port 8765)"
	@echo "  serve-web  - Next.js frontend only (port 3000)"
	@echo "  train      - Alias for train-expression"
	@echo "  evaluate   - Evaluate models"
	@echo "  webcam     - Live webcam emotion detection (DAN, 89.70%)"
	@echo "  ensemble   - Ensemble/TTA/calibration leaderboard"
	@echo "  report     - Regenerate the project report (.docx)"
	@echo "  test       - Run tests"
	@echo "  lint       - Run linter"
	@echo "  clean      - Clean generated files"

# Local API server (FastAPI + SQLite + WebSocket stream) — port 8765
serve:
	uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8765

# API server without auto-reload
serve-prod:
	uvicorn src.api.main:app --host 127.0.0.1 --port 8765

# Next.js frontend (dev; set NEXT_PUBLIC_API_URL in frontend/.env.local)
serve-web:
	cd frontend && npm run dev

# Start API + web together (Ctrl-C stops both)
run:
	@trap 'kill 0' EXIT; \
	uvicorn src.api.main:app --host 127.0.0.1 --port 8765 & \
	cd frontend && npm run dev

# One-shot environment setup: Python + frontend deps + production build
setup:
	pip install -r requirements.txt
	cd frontend && npm install && npm run build

# Install dependencies
install:
	pip install -r requirements.txt

# Live webcam emotion detection
webcam:
	python -m src.pipeline.webcam $(ARGS)

# Ensemble / TTA / calibration leaderboard
ensemble:
	python scripts/ensemble_production.py $(ARGS)

# Train expression classifier
train: train-expression

train-expression:
	python -m ml.training.train_expression --backbone efficientnet_b3 --experiment-id EXP-007 --data-dir data/processed/rafdb

# Evaluate a trained model / experiment
evaluate:
	python ml/evaluate.py --experiment-dir experiments/EXP-008-VIT-SMALL --split test

# Regenerate the project report
report:
	python scripts/generate_project_report.py

# Run tests
test:
	python -m pytest tests/ -v

# Run linting
lint:
	flake8 src/ ml/ tests/ scripts/
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
