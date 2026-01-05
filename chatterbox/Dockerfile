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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Chatterbox TTS without deps to avoid PyTorch conflicts
RUN pip install --no-cache-dir --no-deps chatterbox-tts

# Install chatterbox dependencies
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

# Install FastAPI server dependencies
RUN pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy server code
COPY server.py /app/server.py

# Pre-download model during build
RUN python -c "from chatterbox.tts import ChatterboxTTS; print('Downloading model...'); ChatterboxTTS.from_pretrained(device='cpu'); print('Done!')"

EXPOSE 8100

CMD ["python", "-u", "server.py"]
