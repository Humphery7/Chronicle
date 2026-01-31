"""Pydantic schemas for API request and response models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ASR Schemas
class ASRResponse(BaseModel):
    """Response from ASR endpoint with transcribed text."""

    text: str = Field(..., description="Transcribed text from audio")
    language: str = Field(default="en", description="Detected language code")
    duration_seconds: Optional[float] = Field(
        None, description="Audio duration in seconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of transcription"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "I had a really stressful day at work today.",
                "language": "en",
                "duration_seconds": 5.2,
                "timestamp": "2026-01-28T02:00:00Z",
            }
        }


# Chat Schemas
class ChatRequest(BaseModel):
    """Request to chat endpoint with user message."""

    message: str = Field(
        ..., min_length=1, max_length=5000, description="User's diary entry or message"
    )
    user_id: Optional[str] = Field(
        None, description="Optional user ID for conversation tracking"
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate message is not empty after stripping."""
        if not v.strip():
            raise ValueError("Message cannot be empty or only whitespace")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "message": "I had a really stressful day at work today.",
                "user_id": "user_123",
            }
        }


class ChatResponse(BaseModel):
    """Response from chat endpoint with AI reflection."""

    response: str = Field(..., description="AI-generated supportive response")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "response": "It sounds like today was challenging for you. What made work particularly stressful?",
                "timestamp": "2026-01-28T02:00:05Z",
            }
        }


# TTS Schemas
class TTSRequest(BaseModel):
    """Request to TTS endpoint with text to convert to speech."""

    text: str = Field(
        ..., min_length=1, max_length=2000, description="Text to convert to speech"
    )
    
    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate text is not empty after stripping."""
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "text": "It sounds like today was challenging for you. What made work particularly stressful?"
            }
        }


class TTSResponse(BaseModel):
    """Response from TTS endpoint with audio file info."""

    audio_url: str = Field(..., description="URL or path to generated audio file")
    duration_seconds: Optional[float] = Field(
        None, description="Audio duration in seconds"
    )
    format: str = Field(default="wav", description="Audio format")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Generation timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "audio_url": "/temp/audio_abc123.wav",
                "duration_seconds": 4.8,
                "format": "wav",
                "timestamp": "2026-01-28T02:00:10Z",
            }
        }


# Health Check Schema
class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment (dev/prod)")
    services: dict[str, str] = Field(
        default_factory=dict, description="Status of dependent services"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Health check timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "environment": "development",
                "services": {
                    "huggingface": "connected",
                    "llm": "connected",
                },
                "timestamp": "2026-01-28T02:00:00Z",
            }
        }


# Error Response Schema
class ErrorResponse(BaseModel):
    """Standardized error response."""

    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid audio file format",
                "detail": "Supported formats: WAV, MP3, M4A",
                "timestamp": "2026-01-28T02:00:00Z",
            }
        }
