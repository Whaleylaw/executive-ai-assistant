"""Memory-related utilities for EAIA."""

import logging
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage
from langsmith import Client
from langchain_core.stores import BaseStore

logger = logging.getLogger(__name__)
client = Client()

def setup_memory_store() -> BaseStore:
    """
    Create and configure the memory store.
    
    Returns:
        Configured memory store
    """
    # This is a placeholder implementation using the correct import
    from langgraph.store import InMemoryStore
    return InMemoryStore()

async def process_email_for_memory(
    email: Dict[str, Any],
    messages: List[BaseMessage],
    config: Dict[str, Any],
    store: Any
) -> None:
    """
    Process an email for memory extraction and storage.
    
    Args:
        email: The email to process
        messages: The conversation messages
        config: Configuration settings
        store: The storage backend
    """
    try:
        # Implementation would extract key information from emails and store in memory
        logger.info(f"Processing email {email.get('id', 'unknown')} for memory")
        
        # Get assistant ID from config
        assistant_id = config["configurable"].get("assistant_id", "default")
        
        # Store email data in memory
        namespace = (assistant_id, "email_memory")
        key = email.get("id", "unknown")
        
        # Extract relevant information
        memory_data = {
            "subject": email.get("subject", ""),
            "from": email.get("from_email", ""),
            "to": email.get("to_email", ""),
            "content": email.get("page_content", ""),
            "processed": True
        }
        
        # Store in memory
        await store.aput(namespace, key, memory_data)
        
    except Exception as e:
        logger.error(f"Error processing email for memory: {e}")


def convert_to_langchain_messages(messages: List[Any]) -> List[BaseMessage]:
    """
    Convert messages to LangChain message format.
    
    Args:
        messages: The messages to convert
    
    Returns:
        The converted messages in LangChain format
    """
    # This is a simplified implementation
    # In a real scenario, this would handle different message types and formats
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    
    result = []
    for message in messages:
        if isinstance(message, BaseMessage):
            result.append(message)
        elif isinstance(message, dict):
            content = message.get("content", "")
            if message.get("role") == "user":
                result.append(HumanMessage(content=content))
            elif message.get("role") == "assistant":
                result.append(AIMessage(content=content))
            elif message.get("type") == "tool":
                result.append(ToolMessage(
                    content=content,
                    tool_call_id=message.get("tool_call_id", ""),
                    name=message.get("name", "")
                ))
    
    return result 