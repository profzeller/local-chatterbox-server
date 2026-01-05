# Local Chatterbox TTS Server

Dedicated Text-to-Speech server using Chatterbox on a local GPU.

Optimized for 24GB VRAM GPUs (RTX 4090, etc.) - uses only ~3GB VRAM, leaving plenty of headroom.

## Features

- High-quality text-to-speech generation
- Voice cloning from reference audio
- Emotion tags support (`[happy]`, `[calm]`, etc.)
- Speed adjustment
- REST API for easy integration

## Requirements

- NVIDIA GPU with 8GB+ VRAM (RTX 3080, 4080, 4090, etc.)
- Ubuntu Server 22.04+ (or any Linux with Docker)
- Docker & Docker Compose
- NVIDIA Driver 525+
- NVIDIA Container Toolkit

## VRAM Usage

| Component | VRAM |
|-----------|------|
| Chatterbox TTS | ~3 GB |

This leaves ~21GB free on a 24GB GPU for other tasks if needed.

## Quick Start

### 1. Install NVIDIA Container Toolkit

```bash
# Add NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Configure Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU is accessible
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 2. Clone and Start

```bash
git clone https://github.com/profzeller/local-chatterbox-server.git
cd local-chatterbox-server
docker compose up -d
```

The first start will build the Docker image and download the Chatterbox model (~3GB).

## API Usage

### Health Check

```bash
curl http://localhost:8100/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": true,
  "device": "cuda"
}
```

### Generate Speech

```bash
curl -X POST http://localhost:8100/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Welcome to your wellness journey."}'
```

Response:
```json
{
  "audio_base64": "UklGR...",
  "sample_rate": 24000,
  "duration_seconds": 2.5,
  "text": "Welcome to your wellness journey.",
  "emotion": null
}
```

### With Emotion Tags

```bash
curl -X POST http://localhost:8100/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "[calm] Take a deep breath and relax."}'
```

### With Speed Adjustment

```bash
curl -X POST http://localhost:8100/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Slow and steady wins the race.", "speed": 0.8}'
```

### Voice Cloning

```bash
curl -X POST http://localhost:8100/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is my cloned voice.",
    "reference_audio_url": "https://example.com/my-voice.wav"
  }'
```

Or with base64-encoded audio:

```bash
curl -X POST http://localhost:8100/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is my cloned voice.",
    "reference_audio_base64": "UklGR..."
  }'
```

### Save Audio to File

```bash
# Using jq to extract and decode
curl -s -X POST http://localhost:8100/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  | jq -r '.audio_base64' | base64 -d > output.wav
```

### Python Example

```python
import requests
import base64

# Generate speech
response = requests.post("http://localhost:8100/generate", json={
    "text": "[calm] Welcome to your daily wellness moment.",
    "temperature": 0.5,
    "speed": 0.9
})

data = response.json()

# Save the audio
with open("output.wav", "wb") as f:
    f.write(base64.b64decode(data["audio_base64"]))

print(f"Generated {data['duration_seconds']:.1f}s of audio")
```

## API Reference

### POST /generate

Generate speech from text.

**Request Body:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to speak (can include emotion tags) |
| `reference_audio_url` | string | null | URL to reference audio for voice cloning |
| `reference_audio_base64` | string | null | Base64 reference audio for voice cloning |
| `emotion` | string | null | Override emotion (happy, sad, calm, etc.) |
| `temperature` | float | 0.7 | Randomness (0.1-1.0) |
| `exaggeration` | float | 1.0 | Emotion intensity |
| `speed` | float | 1.0 | Playback speed (0.5-2.0) |
| `cfg_weight` | float | 0.5 | Classifier-free guidance weight |

**Emotion Tags:**

Embed emotions directly in text:
- `[calm] Your text here`
- `[happy] Your text here`
- `[sad] Your text here`
- `[angry] Your text here`

**Response:**

| Field | Type | Description |
|-------|------|-------------|
| `audio_base64` | string | Base64-encoded WAV audio |
| `sample_rate` | int | Audio sample rate (24000) |
| `duration_seconds` | float | Audio duration |
| `text` | string | Text that was spoken |
| `emotion` | string | Detected/applied emotion |

### GET /health

Check service status.

## Network Configuration

By default, the server binds to all interfaces. Access from other machines:

```
http://<server-ip>:8100
```

### Firewall

```bash
sudo ufw allow 8100
```

## Management

```bash
# View logs
docker compose logs -f

# Restart
docker compose restart

# Stop
docker compose down

# Rebuild after changes
docker compose build --no-cache
docker compose up -d
```

## Troubleshooting

### GPU not detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### Container fails to start

```bash
# Check logs
docker compose logs chatterbox

# Rebuild image
docker compose build --no-cache
```

### Slow generation

- First request loads the model (~10-15s)
- Subsequent requests are faster (~1-3s for short text)
- Long text takes proportionally longer

## License

MIT License - Use freely for personal and commercial projects.

## Credits

- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) - Text-to-speech by Resemble AI
