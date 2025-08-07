#!/bin/bash
set -e

# Start Redis server in the background
redis-server --daemonize yes

# Start Celery worker in the background
celery -A src.worker.celery_app worker --loglevel=info --concurrency=1 &

# Start FastAPI server
exec uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# This script starts Redis, Celery worker, and the FastAPI server
# The FastAPI server runs in the foreground to keep the container alive
