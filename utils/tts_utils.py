"""TTS (Text-to-Speech) utility functions."""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from huggingface_hub import AsyncInferenceClient

from config.settings import settings

logger = logging.getLogger(__name__)

# Temporary directory for audio files
TEMP_AUDIO_DIR = Path("temp_audio")
TEMP_AUDIO_DIR.mkdir(exist_ok=True)


class TTSError(Exception):
    """Custom exception for TTS-related errors."""
    pass


def validate_text_for_tts(text: str) -> None:
    """Validate text before TTS conversion.
    
    Args:
        text: Text to validate
        
    Raises:
        TTSError: If validation fails
    """
    if not text or not text.strip():
        raise TTSError("Text cannot be empty")
    
    if len(text) > 2000:
        raise TTSError(f"Text too long: {len(text)} chars. Maximum: 2000 chars")
    
    logger.debug(f"Text validated for TTS: {len(text)} chars")


def generate_audio_filename(text: str) -> str:
    """Generate a unique filename for audio based on text hash.
    
    Args:
        text: Input text
        
    Returns:
        Unique filename with .wav extension
    """
    # Create hash of text for unique filename
    text_hash = hashlib.md5(text.encode()).hexdigest()[:12]
    return f"audio_{text_hash}.wav"


async def text_to_speech(
    client: AsyncInferenceClient,
    text: str,
) -> bytes:
    """Convert text to speech using HuggingFace TTS API.
    
    Args:
        client: HuggingFace AsyncInferenceClient
        text: Text to convert to speech
        
    Returns:
        Audio bytes (WAV format)
        
    Raises:
        TTSError: If TTS conversion fails
    """
    try:
        logger.info(f"Starting TTS with model: {settings.tts_model}")
        
        # Call HuggingFace Inference API
        audio_bytes = await client.text_to_speech(
            text,
            model=settings.tts_model,
        )
        
        if not audio_bytes:
            raise TTSError("TTS returned empty audio")
        
        logger.info(f"TTS completed successfully. Audio size: {len(audio_bytes)} bytes")
        
        return audio_bytes
        
    except Exception as e:
        logger.error(f"TTS failed: {str(e)}")
        if isinstance(e, TTSError):
            raise
        raise TTSError(f"Text-to-speech conversion failed: {str(e)}")


async def save_audio_file(audio_bytes: bytes, filename: str) -> Path:
    """Save audio bytes to file.
    
    Args:
        audio_bytes: Audio data
        filename: Filename to save as
        
    Returns:
        Path to saved file
        
    Raises:
        TTSError: If file saving fails
    """
    try:
        filepath = TEMP_AUDIO_DIR / filename
        
        with open(filepath, "wb") as f:
            f.write(audio_bytes)
        
        logger.debug(f"Audio saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to save audio file: {e}")
        raise TTSError(f"Failed to save audio: {str(e)}")


async def process_tts_request(
    client: AsyncInferenceClient,
    text: str,
) -> tuple[Path, int]:
    """Process TTS request end-to-end.
    
    Args:
        client: HuggingFace AsyncInferenceClient
        text: Text to convert to speech
        
    Returns:
        Tuple of (file_path, file_size_bytes)
        
    Raises:
        TTSError: If processing fails
    """
    # Validate text
    validate_text_for_tts(text)
    
    # Generate audio
    audio_bytes = await text_to_speech(client, text)
    
    # Save to file
    filename = generate_audio_filename(text)
    filepath = await save_audio_file(audio_bytes, filename)
    
    return filepath, len(audio_bytes)


def cleanup_audio_file(filepath: Path) -> None:
    """Delete temporary audio file.
    
    Args:
        filepath: Path to audio file
    """
    try:
        if filepath.exists():
            filepath.unlink()
            logger.debug(f"Deleted audio file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to delete audio file {filepath}: {e}")


def cleanup_old_audio_files(max_age_seconds: int = 3600) -> None:
    """Clean up audio files older than specified age.
    
    Args:
        max_age_seconds: Maximum age in seconds (default: 1 hour)
    """
    import time
    
    try:
        current_time = time.time()
        deleted_count = 0
        
        for filepath in TEMP_AUDIO_DIR.glob("audio_*.wav"):
            file_age = current_time - filepath.stat().st_mtime
            if file_age > max_age_seconds:
                filepath.unlink()
                deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old audio files")
            
    except Exception as e:
        logger.warning(f"Failed to cleanup old audio files: {e}")
