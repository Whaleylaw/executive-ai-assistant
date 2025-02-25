"""Memory-related utilities for EAIA."""

import logging
from typing import Any, Dict, List
import uuid

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

async def process_email_for_memory(email: Dict[str, Any], messages: List[Any], config: Dict[str, Any], store: Any):
    """
    Process email and conversation for memory extraction.
    
    Args:
        email: Dictionary containing email data
        messages: List of messages from the conversation
        config: Configuration dictionary
        store: Storage object for saving memory
        
    This function safely extracts information from the email and conversation,
    then stores it in memory for future reference.
    """
    # Extract namespace from config
    namespace = (
        config.get("configurable", {}).get("assistant_id", "default"),
        "email_memory",
    )
    
    # Extract key information from email
    email_id = email.get("id", str(uuid.uuid4()))
    subject = email.get("subject", "")
    from_email = email.get("from_email", "")
    to_email = email.get("to_email", "")
    
    # Process the conversation to extract key information
    # This will depend on your specific memory format
    
    # Example memory data structure
    memory_data = {
        "email_id": email_id,
        "subject": subject,
        "from": from_email,
        "to": to_email,
        "conversation": _extract_conversation_summary(messages),
        "timestamp": _get_current_timestamp()
    }
    
    # Store in memory
    await store.aput(namespace, email_id, memory_data)
    
    # If using langmem or other memory frameworks, integrate here
    # For example:
    # memory = ConversationBufferMemory()
    # memory.save_context(inputs={"email": email}, outputs={"response": _get_response(messages)})

def _extract_conversation_summary(messages: List[Any]) -> str:
    """
    Extract a summary of the conversation from messages.
    Safely handles both dictionary and object messages.
    """
    summary = []
    
    for message in messages:
        # Check if message is a dictionary or object
        is_dict = isinstance(message, dict)
        
        # Extract information safely
        if is_dict:
            role = message.get("role", "")
            content = message.get("content", "")
        else:
            role = getattr(message, "role", "")
            content = getattr(message, "content", "")
        
        # Add to summary
        if content:
            summary.append(f"{role.capitalize()}: {content}")
    
    return "\n".join(summary)

def _get_response(messages: List[Any]) -> str:
    """
    Get the final response from the conversation.
    Safely handles both dictionary and object messages.
    """
    # Find the last assistant message
    for message in reversed(messages):
        is_dict = isinstance(message, dict)
        
        if is_dict and message.get("role") == "assistant":
            return message.get("content", "")
        elif not is_dict and getattr(message, "role", "") == "assistant":
            return getattr(message, "content", "")
    
    return ""

def _get_current_timestamp():
    """Get current timestamp for memory entry."""
    from datetime import datetime
    return datetime.now().isoformat()

def convert_to_langchain_messages(messages):
    """
    Convert messages to LangChain format, safely handling both dictionary and object messages.
    
    Args:
        messages: List of messages, which could be dictionaries or objects
        
    Returns:
        List of LangChain-formatted messages
    """
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
    
    langchain_messages = []
    
    for message in messages:
        # Determine if message is a dictionary or object
        is_dict = isinstance(message, dict)
        
        # Get the role safely
        if is_dict:
            role = message.get("role", "")
            content = message.get("content", "")
        else:
            role = getattr(message, "role", "")
            content = getattr(message, "content", "")
        
        # Convert based on role
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            # Check for tool calls
            if is_dict and "tool_calls" in message:
                tool_calls = message["tool_calls"]
                # If tool_calls is not a list, make it a list
                if tool_calls and not isinstance(tool_calls, list):
                    tool_calls = [tool_calls]
            elif not is_dict and hasattr(message, "tool_calls"):
                tool_calls = message.tool_calls
                # If tool_calls is not a list, make it a list
                if tool_calls and not isinstance(tool_calls, list):
                    tool_calls = [tool_calls]
            else:
                tool_calls = []
            
            # Process any tool calls
            if tool_calls:
                # Get the first tool call if it exists
                if is_dict:
                    langchain_messages.append(AIMessage(content=content))
                else:
                    langchain_messages.append(AIMessage(content=content))
            else:
                langchain_messages.append(AIMessage(content=content))
                
        elif role == "system":
            langchain_messages.append(SystemMessage(content=content))
        elif role == "tool" or (is_dict and message.get("type") == "tool"):
            # Handle tool message (response from a tool)
            if is_dict:
                name = message.get("name", "")
                tool_content = message.get("content", "")
                tool_call_id = message.get("tool_call_id", "")
            else:
                name = getattr(message, "name", "")
                tool_content = getattr(message, "content", "")
                tool_call_id = getattr(message, "tool_call_id", "")
                
            langchain_messages.append(ToolMessage(
                content=tool_content,
                tool_call_id=tool_call_id,
                name=name
            ))
    
    return langchain_messages 