"""Chat utility functions using LangChain for CBT-style conversations."""

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from config.settings import settings

logger = logging.getLogger(__name__)


class ChatError(Exception):
    """Custom exception for chat-related errors."""
    pass


# CBT-style prompt template for supportive journaling
CBT_SYSTEM_PROMPT = """You are a supportive AI journaling assistant trained in Cognitive Behavioral Therapy (CBT) principles. Your role is to help users reflect on their thoughts and feelings through gentle, non-judgmental conversation.

Guidelines:
- Be warm, empathetic, and supportive
- Ask open-ended questions to encourage self-reflection
- Help users identify patterns in their thoughts and feelings
- Use CBT techniques like thought examination and reframing when appropriate
- Never provide medical advice or diagnosis
- Keep responses concise (2-3 sentences ideal)
- Reflect back what the user shares to show understanding"""

CBT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CBT_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


async def generate_chat_response(
    llm: Any,
    memory: ChatMessageHistory,
    user_message: str,
) -> str:
    """Generate a CBT-style response to user's message.
    
    Args:
        llm: LangChain LLM instance
        memory: Conversation memory for context
        user_message: User's diary entry or message
        
    Returns:
        AI-generated supportive response
        
    Raises:
        ChatError: If response generation fails
    """
    try:
        logger.info(f"Generating chat response for message length: {len(user_message)} chars")
        
        # Create chain with prompt
        chain = CBT_PROMPT | llm
        
        # Get chat history from memory
        chat_history = memory.messages
        
        # Generate response
        response = await chain.ainvoke({
            "input": user_message,
            "chat_history": chat_history
        })
        
        # Extract response text from AIMessage
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        if not response_text or not response_text.strip():
            raise ChatError("Generated response is empty")
        
        # Add messages to memory
        memory.add_user_message(user_message)
        memory.add_ai_message(response_text)
        
        logger.info(f"Chat response generated successfully. Length: {len(response_text)} chars")
        
        return response_text.strip()
        
    except Exception as e:
        logger.error(f"Chat generation failed: {str(e)}")
        if isinstance(e, ChatError):
            raise
        raise ChatError(f"Failed to generate response: {str(e)}")


def clear_conversation_memory(memory: ChatMessageHistory) -> None:
    """Clear conversation memory for a fresh start.
    
    Args:
        memory: Conversation memory to clear
    """
    memory.clear()
    logger.debug("Conversation memory cleared")


def get_conversation_history(memory: ChatMessageHistory) -> list[dict]:
    """Get conversation history from memory.
    
    Args:
        memory: Conversation memory
        
    Returns:
        List of message dictionaries
    """
    try:
        messages = memory.messages
        return [{"role": msg.type, "content": msg.content} for msg in messages]
    except Exception as e:
        logger.warning(f"Failed to load conversation history: {e}")
        return []