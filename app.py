"""AudioDiary FastAPI Application.

A personal AI-powered journaling app with speech-to-text, CBT-style chat, and text-to-speech.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api_v1 import api_v1_router
from config.logging import setup_logging
from config.settings import settings
from utils.dependencies import cleanup_clients, initialize_clients
from utils.tts_utils import cleanup_old_audio_files





setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan context manager for startup and shutdown events."""
 
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info("=" * 60)
    

    missing_keys = settings.validate_required_keys()
    if missing_keys:
        logger.error(f"Missing required API keys: {', '.join(missing_keys)}")
        logger.error("Please set these environment variables before starting the app")
        raise RuntimeError(f"Missing required API keys: {', '.join(missing_keys)}")
    
    try:
        initialize_clients()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    cleanup_clients()
    cleanup_old_audio_files(max_age_seconds=0)  # Clean all temp files on shutdown
    logger.info("Shutdown complete")



app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered journaling app with speech-to-text, CBT-style chat, and text-to-speech",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc) if settings.debug else None,
        },
    )


# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.detail,
        },
    )



app.include_router(api_v1_router, prefix=settings.api_v1_prefix)


# Mount static files for serving generated audio
try:
    from pathlib import Path
    temp_audio_dir = Path("temp_audio")
    temp_audio_dir.mkdir(exist_ok=True)
    app.mount("/temp_audio", StaticFiles(directory=str(temp_audio_dir)), name="temp_audio")
except Exception as e:
    logger.warning(f"Could not mount temp_audio directory: {e}")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "AI-powered journaling app",
        "docs": "/docs",
        "health": f"{settings.api_v1_prefix}/health",
        "endpoints": {
            "asr": f"{settings.api_v1_prefix}/asr",
            "chat": f"{settings.api_v1_prefix}/chat",
            "tts": f"{settings.api_v1_prefix}/tts",
        },
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )
