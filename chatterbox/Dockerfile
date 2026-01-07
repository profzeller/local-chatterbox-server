# Chatterbox TTS Local Server
# Serves a FastAPI endpoint for text-to-speech
#
# Based on working RunPod config from chatterbox-wellness-worker

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install chatterbox-tts WITHOUT dependencies (avoids version conflicts)
RUN pip install --no-cache-dir --no-deps chatterbox-tts

# Manually install chatterbox dependencies (matching working RunPod config)
RUN pip install --no-cache-dir \
    conformer \
    s3tokenizer \
    librosa \
    resemble-perth \
    huggingface_hub \
    safetensors \
    transformers \
    diffusers \
    einops \
    soundfile \
    scipy \
    omegaconf \
    pyloudnorm

# Install FastAPI deps
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy server code
COPY server.py /app/server.py

# Pre-download model during build (skip if it fails - will download on first use)
RUN python -c "from chatterbox.tts import ChatterboxTTS; print('Downloading model...'); ChatterboxTTS.from_pretrained(device='cpu'); print('Done!')" || echo "Model will download on first use"

EXPOSE 8100

CMD ["python", "-u", "server.py"]
