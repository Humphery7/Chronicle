"""API endpoints package."""

from .asr_endpoint import router as asr_router
from .health import router as health_router
from .health_chat import router as chat_router
from .tts_endpoint import router as tts_router

__all__ = ["asr_router", "chat_router", "tts_router", "health_router"]
