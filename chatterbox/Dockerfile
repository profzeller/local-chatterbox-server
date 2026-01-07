# Chatterbox TTS Local Server
# Serves a FastAPI endpoint for text-to-speech

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Configure pip for better reliability (retries, timeout)
RUN pip config set global.retries 10 && \
    pip config set global.timeout 60

# Force reinstall PyTorch stack with pinned versions (base image uses conda, causes conflicts)
# PyTorch 2.4.0 + torchvision 0.19.0 + torchaudio 2.4.0 are compatible
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install FastAPI deps first (small, fast)
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Install chatterbox-tts (has many dependencies, may take time)
RUN pip install --no-cache-dir chatterbox-tts

# Copy server code
COPY server.py /app/server.py

# Pre-download model during build (skip if it fails - will download on first use)
RUN python -c "from chatterbox.tts import ChatterboxTTS; print('Downloading model...'); ChatterboxTTS.from_pretrained(device='cpu'); print('Done!')" || echo "Model will download on first use"

EXPOSE 8100

CMD ["python", "-u", "server.py"]
