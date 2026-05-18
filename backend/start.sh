#!/bin/bash
# Run from the failsafe/ root directory:  bash backend/start.sh
cd "$(dirname "$0")/.."
source .venv/bin/activate
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
