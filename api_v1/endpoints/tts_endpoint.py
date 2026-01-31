"""TTS (Text-to-Speech) endpoint."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from api_v1.schemas import ErrorResponse, TTSRequest, TTSResponse
from utils.dependencies import HFClient
from utils.tts_utils import TTSError, cleanup_audio_file, process_tts_request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tts", tags=["TTS"])


@router.post(
    "",
    response_class=FileResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "TTS processing failed"},
    },
    summary="Convert text to speech",
    description="Convert text to audio using text-to-speech synthesis",
)
async def text_to_speech_endpoint(
    request: TTSRequest,
    hf_client: HFClient = None,
) -> FileResponse:
    """
    Convert text to speech audio.
    
    Accepts text and returns an audio file (WAV format) with the synthesized speech.
    The audio file is automatically cleaned up after being sent to the client.
    """
    try:
        logger.info(f"TTS request received - text length: {len(request.text)} chars")
        
        # Process TTS request
        filepath, file_size = await process_tts_request(hf_client, request.text)
        
        # Return audio file as response
        return FileResponse(
            path=str(filepath),
            media_type="audio/wav",
            filename=filepath.name,
            headers={
                "Content-Length": str(file_size),
            },
            background=None,  # Keep file until response is sent
        )
        
    except TTSError as e:
        logger.warning(f"TTS validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"TTS processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Text-to-speech conversion failed. Please try again.",
        )


@router.post(
    "/json",
    response_model=TTSResponse,
    status_code=status.HTTP_200_OK,
    summary="Convert text to speech (JSON response)",
    description="Alternative endpoint that returns JSON with audio file information",
)
async def text_to_speech_json_endpoint(
    request: TTSRequest,
    hf_client: HFClient = None,
) -> TTSResponse:
    """
    Convert text to speech and return JSON with file information.
    
    This is an alternative to the main TTS endpoint that returns metadata
    instead of the audio file directly.
    """
    try:
        logger.info(f"TTS JSON request received - text length: {len(request.text)} chars")
        
        # Process TTS request
        filepath, file_size = await process_tts_request(hf_client, request.text)
        
        # Calculate duration estimate (rough approximation)
        duration_estimate = len(request.text) / 15  # ~15 chars per second speech
        
        # Return response
        return TTSResponse(
            audio_url=f"/temp_audio/{filepath.name}",
            duration_seconds=round(duration_estimate, 1),
            format="wav",
            timestamp=datetime.utcnow(),
        )
        
    except TTSError as e:
        logger.warning(f"TTS validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"TTS processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Text-to-speech conversion failed. Please try again.",
        )
