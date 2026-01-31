"""API v1 router aggregation."""

from fastapi import APIRouter

from .endpoints import asr_router, chat_router, health_router, tts_router

# Create main API v1 router
api_v1_router = APIRouter()

# Include all endpoint routers
api_v1_router.include_router(health_router)
api_v1_router.include_router(asr_router)
api_v1_router.include_router(chat_router)
api_v1_router.include_router(tts_router)

__all__ = ["api_v1_router"]
