# Chatterbox TTS Local Server
# Serves a FastAPI endpoint for text-to-speech

# Use clean CUDA base image (avoids conda/pip conflicts from pytorch image)
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Upgrade pip first
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Configure pip for better reliability (retries, timeout)
RUN pip config set global.retries 10 && \
    pip config set global.timeout 60

# Install PyTorch stack with CUDA 12.4 support
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install FastAPI deps
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Install chatterbox-tts (has many dependencies, may take time)
RUN pip install --no-cache-dir chatterbox-tts

# Copy server code
COPY server.py /app/server.py

# Pre-download model during build (skip if it fails - will download on first use)
RUN python -c "from chatterbox.tts import ChatterboxTTS; print('Downloading model...'); ChatterboxTTS.from_pretrained(device='cpu'); print('Done!')" || echo "Model will download on first use"

EXPOSE 8100

CMD ["python", "-u", "server.py"]
