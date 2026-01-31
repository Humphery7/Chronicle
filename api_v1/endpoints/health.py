"""Health check endpoint."""

import logging
from datetime import datetime

from fastapi import APIRouter, status

from api_v1.schemas import HealthResponse
from config.settings import settings
from utils.dependencies import get_hf_client, get_llm

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check the health status of the API and its dependencies",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns the status of the API and its connected services (HuggingFace, LLM).
    Useful for monitoring and ensuring all services are properly initialized.
    """
    services = {}
    
    # Check HuggingFace client
    try:
        hf_client = get_hf_client()
        services["huggingface"] = "connected" if hf_client else "disconnected"
    except Exception as e:
        services["huggingface"] = f"error: {str(e)}"
        logger.warning(f"HuggingFace health check failed: {e}")
    
    # Check LLM
    try:
        llm = get_llm()
        services["llm"] = "connected" if llm else "disconnected"
    except Exception as e:
        services["llm"] = f"error: {str(e)}"
        logger.warning(f"LLM health check failed: {e}")
    
    # Overall status
    overall_status = "healthy" if all(
        v == "connected" for v in services.values()
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        services=services,
        timestamp=datetime.utcnow(),
    )
