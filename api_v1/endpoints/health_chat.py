"""Health chat endpoint for CBT-style reflective conversations."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, status

from api_v1.schemas import ChatRequest, ChatResponse, ErrorResponse
from utils.chat_utils import ChatError, generate_chat_response
from utils.dependencies import get_conversation_memory, get_llm

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Chat processing failed"},
    },
    summary="Get supportive chat response",
    description="Send a diary entry or message and receive a CBT-style supportive reflection",
)
async def chat_endpoint(
    request: ChatRequest,
) -> ChatResponse:
    """
    Generate a supportive, CBT-style response to user's message.
    
    This endpoint uses LangChain with conversation memory to provide context-aware,
    therapeutic-style reflections on the user's diary entries or messages.
    
    **Note**: This is a journaling assistant, not medical advice.
    """
    try:
        logger.info(f"Chat request received - user_id: {request.user_id or 'default'}")
        
        # Get LLM instance from backend configuration
        llm = get_llm()
        
        # Get conversation memory for this user
        user_id = request.user_id or "default"
        memory = get_conversation_memory(user_id)
        
        # Generate response
        response_text = await generate_chat_response(
            llm=llm,
            memory=memory,
            user_message=request.message,
        )
        
        # Return response
        return ChatResponse(
            response=response_text,
            timestamp=datetime.utcnow(),
        )
        
    except ChatError as e:
        logger.warning(f"Chat validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat response generation failed. Please try again.",
        )
