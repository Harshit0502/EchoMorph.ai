FROM python:3.11-slim

# Whisper needs ffmpeg installed as a system binary at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*
# (ffmpeg requirement is documented for Whisper) 
# (libsndfile helps soundfile/librosa on slim images)
# 

WORKDIR /app
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir webrtcvad-wheels==2.0.14 \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir --no-deps resemblyzer==0.1.4

COPY . .

CMD ["sh","-c","streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true"]
