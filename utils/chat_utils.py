"""Chat utility functions using LangChain for CBT-style conversations."""

import logging
from typing import Any

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

from config.settings import settings

logger = logging.getLogger(__name__)


class ChatError(Exception):
    """Custom exception for chat-related errors."""
    pass


# CBT-style prompt template for supportive journaling
CBT_PROMPT_TEMPLATE = """You are a supportive AI journaling assistant trained in Cognitive Behavioral Therapy (CBT) principles. Your role is to help users reflect on their thoughts and feelings through gentle, non-judgmental conversation.

Guidelines:
- Be warm, empathetic, and supportive
- Ask open-ended questions to encourage self-reflection
- Help users identify patterns in their thoughts and feelings
- Use CBT techniques like thought examination and reframing when appropriate
- Never provide medical advice or diagnosis
- Keep responses concise (2-3 sentences ideal)
- Reflect back what the user shares to show understanding

Current conversation:
{chat_history}

User: {input}

Your supportive response:"""

CBT_PROMPT = PromptTemplate(
    input_variables=["chat_history", "input"],
    template=CBT_PROMPT_TEMPLATE,
)


def create_conversation_chain(
    llm: Any,
    memory: ConversationBufferWindowMemory,
) -> ConversationChain:
    """Create a conversation chain with CBT-style prompting.
    
    Args:
        llm: LangChain LLM instance
        memory: Conversation memory instance
        
    Returns:
        Configured ConversationChain
    """
    chain = ConversationChain(
        llm=llm,
        prompt=CBT_PROMPT,
        memory=memory,
        verbose=settings.debug,
    )
    
    logger.debug("Created conversation chain with CBT prompt")
    return chain


async def generate_chat_response(
    llm: Any,
    memory: ConversationBufferWindowMemory,
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
        
        # Create conversation chain
        chain = create_conversation_chain(llm, memory)
        
        # Generate response
        response = await chain.ainvoke({"input": user_message})
        
        # Extract response text
        if isinstance(response, dict):
            response_text = response.get("response", "")
        else:
            response_text = str(response)
        
        if not response_text or not response_text.strip():
            raise ChatError("Generated response is empty")
        
        logger.info(f"Chat response generated successfully. Length: {len(response_text)} chars")
        
        return response_text.strip()
        
    except Exception as e:
        logger.error(f"Chat generation failed: {str(e)}")
        if isinstance(e, ChatError):
            raise
        raise ChatError(f"Failed to generate response: {str(e)}")


def clear_conversation_memory(memory: ConversationBufferWindowMemory) -> None:
    """Clear conversation memory for a fresh start.
    
    Args:
        memory: Conversation memory to clear
    """
    memory.clear()
    logger.debug("Conversation memory cleared")


def get_conversation_history(memory: ConversationBufferWindowMemory) -> list[dict]:
    """Get conversation history from memory.
    
    Args:
        memory: Conversation memory
        
    Returns:
        List of message dictionaries
    """
    try:
        messages = memory.load_memory_variables({}).get("chat_history", [])
        return [{"role": msg.type, "content": msg.content} for msg in messages]
    except Exception as e:
        logger.warning(f"Failed to load conversation history: {e}")
        return []
