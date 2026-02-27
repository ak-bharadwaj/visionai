#!/bin/bash
set -e
[ ! -d venv ] && python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --quiet
[ ! -f .env ] && cp .env.example .env
uvicorn backend.main:app --host 0.0.0.0 --port 8000
