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

# Install chatterbox-tts with all dependencies
# Using git install for latest version with proper deps
RUN pip install --no-cache-dir \
    chatterbox-tts \
    fastapi \
    uvicorn \
    python-multipart

# Copy server code
COPY server.py /app/server.py

# Pre-download model during build
RUN python -c "from chatterbox.tts import ChatterboxTTS; print('Downloading model...'); ChatterboxTTS.from_pretrained(device='cpu'); print('Done!')"

EXPOSE 8100

CMD ["python", "-u", "server.py"]
