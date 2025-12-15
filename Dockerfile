FROM python:3.11-slim

# System deps:
# - build-essential: compile webrtcvad C-extension
# - ffmpeg: required by openai-whisper at runtime
# - libsndfile1: common runtime dep for soundfile/librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8080"]
