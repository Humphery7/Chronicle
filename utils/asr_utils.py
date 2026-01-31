"""ASR (Automatic Speech Recognition) utility functions."""

import logging
from io import BytesIO
from typing import BinaryIO

from fastapi import UploadFile
from huggingface_hub import AsyncInferenceClient

from config.settings import settings

logger = logging.getLogger(__name__)


class ASRError(Exception):
    """Custom exception for ASR-related errors."""
    pass


async def validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file.
    
    Args:
        file: Uploaded audio file
        
    Raises:
        ASRError: If file validation fails
    """
    # Check content type
    if file.content_type not in settings.allowed_audio_formats:
        allowed = ", ".join(settings.allowed_audio_formats)
        raise ASRError(
            f"Invalid audio format: {file.content_type}. Allowed formats: {allowed}"
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > settings.max_audio_file_size_bytes:
        max_mb = settings.max_audio_file_size_mb
        actual_mb = file_size / (1024 * 1024)
        raise ASRError(
            f"File too large: {actual_mb:.1f}MB. Maximum allowed: {max_mb}MB"
        )
    
    if file_size == 0:
        raise ASRError("Empty audio file")
    
    logger.debug(f"Audio file validated: {file.filename}, size: {file_size} bytes")


async def transcribe_audio(
    client: AsyncInferenceClient,
    audio_data: bytes,
) -> dict:
    """Transcribe audio using HuggingFace Whisper API.
    
    Args:
        client: HuggingFace AsyncInferenceClient
        audio_data: Raw audio bytes
        
    Returns:
        Dictionary with transcription result
        
    Raises:
        ASRError: If transcription fails
    """
    try:
        logger.info(f"Starting ASR with model: {settings.whisper_model}")
        
        # Call HuggingFace Inference API
        result = await client.automatic_speech_recognition(
            audio_data,
            model=settings.whisper_model,
        )
        
        # Handle different response formats
        if isinstance(result, dict):
            text = result.get("text", "")
        elif isinstance(result, str):
            text = result
        else:
            text = str(result)
        
        if not text or not text.strip():
            raise ASRError("Transcription returned empty text")
        
        logger.info(f"ASR completed successfully. Text length: {len(text)} chars")
        
        return {
            "text": text.strip(),
            "language": "en",  # Whisper auto-detects but we default to English
        }
        
    except Exception as e:
        logger.error(f"ASR failed: {str(e)}")
        if isinstance(e, ASRError):
            raise
        raise ASRError(f"Transcription failed: {str(e)}")


async def process_audio_upload(
    client: AsyncInferenceClient,
    file: UploadFile,
) -> dict:
    """Process uploaded audio file end-to-end.
    
    Args:
        client: HuggingFace AsyncInferenceClient
        file: Uploaded audio file
        
    Returns:
        Dictionary with transcription result
        
    Raises:
        ASRError: If processing fails
    """
    # Validate file
    await validate_audio_file(file)
    
    # Read file contents
    audio_data = await file.read()
    
    # Transcribe
    result = await transcribe_audio(client, audio_data)
    
    return result
