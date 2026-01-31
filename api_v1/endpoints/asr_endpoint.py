"""ASR (Automatic Speech Recognition) endpoint."""

import logging
from datetime import datetime

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api_v1.schemas import ASRResponse, ErrorResponse
from utils.asr_utils import ASRError, process_audio_upload
from utils.dependencies import HFClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/asr", tags=["ASR"])


@router.post(
    "",
    response_model=ASRResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid audio file"},
        500: {"model": ErrorResponse, "description": "ASR processing failed"},
    },
    summary="Transcribe audio to text",
    description="Upload an audio file to transcribe speech to text using Whisper ASR",
)
async def transcribe_audio_endpoint(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, M4A)"),
    hf_client: HFClient = None,
) -> ASRResponse:
    """
    Transcribe audio file to text.
    
    Accepts audio files in WAV, MP3, or M4A format and returns the transcribed text.
    Maximum file size is configurable (default 25MB).
    """
    try:
        logger.info(f"ASR request received - filename: {file.filename}")
        
        # Process audio upload
        result = await process_audio_upload(hf_client, file)
        
        # Return response
        return ASRResponse(
            text=result["text"],
            language=result.get("language", "en"),
            timestamp=datetime.utcnow(),
        )
        
    except ASRError as e:
        logger.warning(f"ASR validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"ASR processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audio transcription failed. Please try again.",
        )
