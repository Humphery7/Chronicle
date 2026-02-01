"""Dependency injection functions for FastAPI endpoints."""

import logging
from typing import Annotated, Any, Generator

from fastapi import Depends
from huggingface_hub import AsyncInferenceClient
from langchain_community.chat_message_histories import ChatMessageHistory

from config.settings import settings

logger = logging.getLogger(__name__)

_hf_client: AsyncInferenceClient | None = None
_llm_chain = None
_conversation_memory: dict[str, ChatMessageHistory] = {}


def initialize_clients() -> None:
    """Initialize HuggingFace and LLM clients at startup."""
    global _hf_client, _llm_chain
    
    try:
        _hf_client = AsyncInferenceClient(
            token=settings.huggingface_api_key,
            timeout=settings.hf_timeout,
        )
        logger.info("HuggingFace AsyncInferenceClient initialized")
        
        # Initialize LLM (supports multiple providers via LangChain)
        if settings.llm_provider == "germini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            _llm_chain = ChatGoogleGenerativeAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                api_key=settings.llm_api_key,
            )
        elif settings.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            _llm_chain = ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                api_key=settings.llm_api_key,
            )
        elif settings.llm_provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            _llm_chain = ChatAnthropic(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                api_key=settings.llm_api_key,
            )
        else:
            _llm_chain = ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                api_key=settings.llm_api_key,
            )
        
        logger.info(f"LLM initialized: {settings.llm_provider}/{settings.llm_model}")
        
    except Exception as e:
        logger.error(f"Failed to initialize clients: {e}")
        raise


def cleanup_clients() -> None:
    """Cleanup resources at shutdown."""
    global _hf_client, _llm_chain, _conversation_memory
    
    _hf_client = None
    _llm_chain = None
    _conversation_memory.clear()
    logger.info("Cleaned up global resources")


def get_hf_client() -> AsyncInferenceClient:
    """Dependency to get HuggingFace client instance.
    
    Returns:
        AsyncInferenceClient instance
        
    Raises:
        RuntimeError: If client not initialized
    """
    if _hf_client is None:
        raise RuntimeError("HuggingFace client not initialized")
    return _hf_client


def get_llm():
    """Dependency to get LLM instance.
    
    Returns:
        LLM instance (ChatOpenAI, ChatAnthropic, etc.)
        
    Raises:
        RuntimeError: If LLM not initialized
    """
    if _llm_chain is None:
        raise RuntimeError("LLM not initialized")
    return _llm_chain


def get_conversation_memory(user_id: str = "default") -> ChatMessageHistory:
    """Get or create conversation memory for a user.
    
    Args:
        user_id: User identifier for memory isolation
        
    Returns:
        ChatMessageHistory instance
    """
    if user_id not in _conversation_memory:
        _conversation_memory[user_id] = ChatMessageHistory()
        logger.debug(f"Created new conversation memory for user: {user_id}")
    
    return _conversation_memory[user_id]


# Type aliases for dependency injection
HFClient = Annotated[AsyncInferenceClient, Depends(get_hf_client)]
LLM = Annotated[Any, Depends(get_llm)]
