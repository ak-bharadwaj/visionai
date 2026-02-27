@echo off
echo [VisionTalk] Starting...

if not exist venv (
    echo [VisionTalk] Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo [VisionTalk] Installing dependencies...
pip install -r requirements.txt --quiet

if not exist .env (
    copy .env.example .env
    echo [VisionTalk] Created .env — edit CAMERA_SOURCE with your phone's IP
)

echo [VisionTalk] Starting server at http://0.0.0.0:8000
uvicorn backend.main:app --host 0.0.0.0 --port 8000
