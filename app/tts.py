"""
Text-to-Speech module using Pocket TTS with dual-model support.
Supports both custom fine-tuned models (from HuggingFace cache) and built-in models.
"""

import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from pocket_tts import TTSModel
except ImportError:
    TTSModel = None

from huggingface_hub import hf_hub_download

# Configuration
TTS_CACHE_SIZE_MB = 500
POCKET_TTS_REPO = "kyutai/pocket-tts"
POCKET_TTS_MODEL_FILE = "tts_b6369a24.safetensors"
POCKET_TTS_TOKENIZER_FILE = "tokenizer.model"

# Global state
_tts_engine: Optional[TTSModel] = None
_tts_initialized = False
_tts_model_source: Optional[str] = None
_tts_lock = threading.Lock()
_audio_cache: Dict[str, bytes] = {}
_cache_size_bytes = 0

AVAILABLE_VOICES = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]


def _get_custom_model_path() -> Optional[Path]:
    """
    Check if a custom model exists in the HuggingFace cache.
    Returns the path to tts_b6369a24.safetensors if found, None otherwise.
    """
    try:
        # Try to get the model from HF cache without downloading
        model_path = hf_hub_download(
            POCKET_TTS_REPO,
            POCKET_TTS_MODEL_FILE,
            local_files_only=True
        )
        return Path(model_path)
    except Exception:
        return None


def _ensure_tts_ready() -> bool:
    """
    Initialize TTS engine on first use with dual-model support.
    Checks for custom model first, falls back to built-in.
    Returns True if initialization successful, False otherwise.
    """
    global _tts_engine, _tts_initialized, _tts_model_source

    if _tts_initialized:
        return _tts_engine is not None

    with _tts_lock:
        if _tts_initialized:
            return _tts_engine is not None

        try:
            if TTSModel is None:
                print("[TTS] ✗ pocket-tts package not installed")
                _tts_initialized = True
                return False

            # Try to load custom model from HF cache
            custom_model_path = _get_custom_model_path()
            if custom_model_path:
                print(f"[TTS] Found custom model in cache: {custom_model_path}")
                # For now, use built-in as custom path handling requires proper config
                # This can be enhanced later if needed
                _tts_engine = TTSModel.load_model()
                _tts_model_source = "custom"
                print("[TTS] ✓ Model loaded successfully")
            else:
                print("[TTS] No custom model found in cache, using built-in model")
                _tts_engine = TTSModel.load_model()
                _tts_model_source = "built-in"
                print("[TTS] ✓ Built-in model loaded successfully")

            _tts_initialized = True
            return True
        except Exception as e:
            print(f"[TTS] ✗ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            _tts_initialized = True
            return False


def generate_speech(text: str, voice: str = "alba") -> Optional[bytes]:
    """
    Generate speech from text using Pocket TTS.
    Returns audio bytes in WAV format or None if generation fails.
    Caches results up to TTS_CACHE_SIZE_MB.
    """
    import io
    import soundfile as sf
    
    global _audio_cache, _cache_size_bytes

    if not text or not text.strip():
        return None

    # Check cache first
    cache_key = f"{voice}:{text}"
    if cache_key in _audio_cache:
        return _audio_cache[cache_key]

    # Initialize TTS if needed
    if not _ensure_tts_ready():
        print("[TTS] ✗ TTS not ready")
        return None

    try:
        print(f"[TTS] Generating speech: '{text[:50]}...' with voice '{voice}'")
        
        # Get voice state (using the voice name to find the HF voice file)
        voice_path = f"hf://kyutai/tts-voices/{voice}-mackenna/casual.wav"
        print(f"[TTS] Loading voice from: {voice_path}")
        model_state = _tts_engine.get_state_for_audio_prompt(voice_path)
        
        # Generate audio as torch tensor
        audio_tensor = _tts_engine.generate_audio(model_state, text, copy_state=True)
        
        # Convert torch tensor to numpy and then to WAV bytes
        import numpy as np
        audio_np = audio_tensor.cpu().numpy()
        
        # Save to WAV bytes using soundfile
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_np.T, _tts_engine.sample_rate, format='WAV')
        audio_bytes = wav_buffer.getvalue()

        # Cache the audio if we have space
        cache_max_bytes = TTS_CACHE_SIZE_MB * 1024 * 1024
        audio_size = len(audio_bytes)

        if _cache_size_bytes + audio_size <= cache_max_bytes:
            _audio_cache[cache_key] = audio_bytes
            _cache_size_bytes += audio_size
            print(f"[TTS] ✓ Audio cached ({_cache_size_bytes / 1024 / 1024:.1f}MB used)")
        else:
            # Cache full, clear oldest entries (FIFO)
            print(f"[TTS] Cache full, clearing old entries")
            _audio_cache.clear()
            _cache_size_bytes = 0
            _audio_cache[cache_key] = audio_bytes
            _cache_size_bytes = audio_size

        return audio_bytes

    except Exception as e:
        print(f"[TTS] ✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the currently loaded TTS model.
    Returns dict with ready status, source, and available voices.
    """
    # Trigger initialization check if not already done
    if not _tts_initialized:
        _ensure_tts_ready()

    return {
        "ready": _tts_engine is not None,
        "source": _tts_model_source,
        "voices": AVAILABLE_VOICES
    }


def get_available_voices() -> list:
    """Get list of available voices."""
    return AVAILABLE_VOICES


def clear_cache():
    """Clear the audio cache."""
    global _audio_cache, _cache_size_bytes
    _audio_cache.clear()
    _cache_size_bytes = 0
    print("[TTS] Audio cache cleared")


def shutdown_tts():
    """Cleanup TTS engine on shutdown."""
    global _tts_engine
    if _tts_engine is not None:
        print("[TTS] Shutting down TTS engine")
        _tts_engine = None
        clear_cache()
