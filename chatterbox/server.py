"""
Chatterbox TTS Local Server
FastAPI-based HTTP API for text-to-speech generation
"""

import base64
import io
import os
import re
import tempfile
import urllib.request
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI(title="Chatterbox TTS", description="Text-to-Speech API with voice cloning")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
tts_model = None


def load_model():
    """Load Chatterbox TTS model."""
    global tts_model

    if tts_model is not None:
        return tts_model

    print("[Server] Loading Chatterbox TTS model...")
    from chatterbox.tts import ChatterboxTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Server] Using device: {device}")

    tts_model = ChatterboxTTS.from_pretrained(device=device)
    print("[Server] Model loaded successfully")

    return tts_model


class TTSRequest(BaseModel):
    text: str
    reference_audio_url: Optional[str] = None
    reference_audio_base64: Optional[str] = None
    emotion: Optional[str] = None
    temperature: float = 0.7
    exaggeration: float = 1.0
    speed: float = 1.0
    cfg_weight: float = 0.5


class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration_seconds: float
    text: str
    emotion: Optional[str] = None


def parse_emotion_tags(text: str) -> tuple[str, Optional[str]]:
    """Parse emotion tags from text like [happy] or <emotion:sad>"""
    bracket_match = re.match(r"^\[(\w+)\]\s*(.+)$", text, re.DOTALL)
    if bracket_match:
        return bracket_match.group(2).strip(), bracket_match.group(1).lower()

    tag_match = re.match(r"^<emotion:(\w+)>\s*(.+)$", text, re.DOTALL)
    if tag_match:
        return tag_match.group(2).strip(), tag_match.group(1).lower()

    return text, None


def download_reference_audio(url: str) -> str:
    """Download reference audio from URL to temp file."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        urllib.request.urlretrieve(url, temp_file.name)
        return temp_file.name
    except Exception as e:
        os.unlink(temp_file.name)
        raise Exception(f"Failed to download reference audio: {e}")


def base64_to_audio_file(b64_data: str) -> str:
    """Convert base64 audio to temp file."""
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]

    audio_bytes = base64.b64decode(b64_data)
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.write(audio_bytes)
    temp_file.close()
    return temp_file.name


def audio_to_base64(audio_array: np.ndarray, sample_rate: int = 24000) -> str:
    """Convert audio array to base64 encoded WAV."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": tts_model is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/generate", response_model=TTSResponse)
async def generate_speech(request: TTSRequest):
    """Generate speech from text."""

    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        model = load_model()

        # Parse emotion from text if not provided
        text = request.text
        emotion = request.emotion
        clean_text, text_emotion = parse_emotion_tags(text)
        if text_emotion and not emotion:
            emotion = text_emotion
            text = clean_text

        # Handle reference audio
        ref_audio_path = None

        if request.reference_audio_url:
            ref_audio_path = download_reference_audio(request.reference_audio_url)
        elif request.reference_audio_base64:
            ref_audio_path = base64_to_audio_file(request.reference_audio_base64)

        # Generate speech
        print(f"[Server] Generating: {text[:50]}...")

        if ref_audio_path:
            audio = model.generate(
                text=text,
                audio_prompt_path=ref_audio_path,
                temperature=request.temperature,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
            )
            os.unlink(ref_audio_path)
        else:
            audio = model.generate(
                text=text,
                temperature=request.temperature,
                exaggeration=request.exaggeration,
                cfg_weight=request.cfg_weight,
            )

        # Apply speed adjustment
        if request.speed != 1.0:
            from scipy import signal
            original_length = len(audio)
            new_length = int(original_length / request.speed)
            audio = signal.resample(audio, new_length)

        # Convert to numpy
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()

        if len(audio.shape) > 1:
            audio = audio.squeeze()

        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.95

        # Convert to base64
        audio_b64 = audio_to_base64(audio, sample_rate=24000)

        return TTSResponse(
            audio_base64=audio_b64,
            sample_rate=24000,
            duration_seconds=len(audio) / 24000,
            text=text,
            emotion=emotion
        )

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@app.on_event("startup")
async def startup_event():
    """Pre-load model on startup."""
    print("[Server] Starting up, loading model...")
    load_model()
    print("[Server] Ready to serve requests")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
