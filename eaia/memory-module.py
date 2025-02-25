"""Memory management for Executive AI Assistant.

This module provides memory capabilities for the Executive AI Assistant using LangMem.
It handles setup, configuration, and utilities for both semantic and procedural memory.
"""

import uuid
from typing import Optional, List, Dict, Any, Tuple

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.store.base import BaseStore
from langmem import (
    create_manage_memory_tool, 
    create_search_memory_tool, 
    create_memory_manager,
    create_thread_extractor
)

from eaia.schemas import EmailData
from eaia.main.config import get_config


class ContactInfo(BaseModel):
    """Contact information for a person extracted from conversations."""
    name: str = Field(description="The person's name")
    email: Optional[str] = Field(description="The person's email address", default=None)
    company: Optional[str] = Field(description="The company the person works for", default=None)
    role: Optional[str] = Field(description="The person's role or title", default=None)
    relationship: Optional[str] = Field(description="Relationship to the user", default=None)


class EmailPreference(BaseModel):
    """Preferences related to email communication."""
    type: str = Field(description="Type of preference (e.g., 'scheduling', 'tone', 'format')")
    description: str = Field(description="Description of the preference")
    confidence: float = Field(description="Confidence score (0-1)", ge=0, le=1)


class ConversationSummary(BaseModel):
    """Summary of an email conversation."""
    subject: str = Field(description="Subject of the conversation")
    participants: List[str] = Field(description="List of participants' emails")
    key_points: List[str] = Field(description="Key points discussed")
    action_items: Optional[List[str]] = Field(description="Action items identified", default=None)
    follow_up_needed: bool = Field(description="Whether follow-up is needed")


def setup_memory_store(config=None):
    """Configure and initialize the memory store with embeddings."""
    from langgraph.store.memory import InMemoryStore
    
    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }
    )
    return store


def get_memory_namespace(config, store_type="memories"):
    """Get the appropriate namespace based on configuration.
    
    Args:
        config: The configuration dictionary
        store_type: The type of memory store ("memories", "preferences", "contacts", etc.)
        
    Returns:
        A tuple representing the namespace
    """
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    assistant_id = config.get("configurable", {}).get("assistant_id", "default")
    
    return (assistant_id, store_type, user_id)


def create_memory_tools(config=None):
    """Create memory management tools that can be used by the agent.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of memory management tools
    """
    if config is None:
        config = {"configurable": {}}
    
    # Create namespaces for different types of memories
    memories_namespace = get_memory_namespace(config, "memories")
    contacts_namespace = get_memory_namespace(config, "contacts")
    preferences_namespace = get_memory_namespace(config, "preferences")
    
    # Create tools for each namespace
    tools = [
        create_manage_memory_tool(
            namespace=memories_namespace,
            instructions="Store important information from the conversation that may be useful in the future. Focus on facts, preferences, and key details about the user and their contacts."
        ),
        create_search_memory_tool(
            namespace=memories_namespace,
            instructions="Search for relevant information from past conversations to help with the current task."
        ),
        create_manage_memory_tool(
            namespace=contacts_namespace,
            instructions="Store contact information for people mentioned in emails. Include names, email addresses, companies, and relationships."
        ),
        create_search_memory_tool(
            namespace=contacts_namespace,
            instructions="Search for contact information that might be relevant to the current conversation."
        ),
        create_manage_memory_tool(
            namespace=preferences_namespace,
            instructions="Store user preferences about email communication, scheduling, and other assistant behaviors."
        ),
        create_search_memory_tool(
            namespace=preferences_namespace,
            instructions="Search for user preferences that might be relevant to the current task."
        )
    ]
    
    return tools


def create_background_memory_manager(config=None):
    """Create a background memory manager for automatic extraction.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        A memory manager function
    """
    model = config.get("configurable", {}).get("model", "gpt-4o") if config else "gpt-4o"
    
    # Create memory manager with custom instructions
    manager = create_memory_manager(
        model,
        instructions="""
        Extract key information from email conversations. Focus on:
        1. User preferences and communication style
        2. Contact information for people mentioned
        3. Important deadlines or scheduled events
        4. Action items and commitments
        5. Recurring topics or themes
        
        Avoid storing transient information or overly specific details that won't be relevant later.
        """
    )
    
    return manager


async def extract_conversation_summary(messages, config=None):
    """Extract a structured summary from a conversation.
    
    Args:
        messages: List of messages in the conversation
        config: Configuration dictionary
        
    Returns:
        A ConversationSummary object
    """
    model = config.get("configurable", {}).get("model", "gpt-4o") if config else "gpt-4o"
    
    extractor = create_thread_extractor(
        model,
        schema=ConversationSummary,
        instructions="""
        Analyze this email conversation and extract:
        1. The main subject or topic
        2. All participants (by email)
        3. Key points discussed
        4. Any action items or follow-up tasks
        5. Whether follow-up is needed
        
        Focus on information that would be helpful for future reference.
        """
    )
    
    return await extractor(messages)


async def extract_contact_information(messages, config=None):
    """Extract contact information from a conversation.
    
    Args:
        messages: List of messages in the conversation
        config: Configuration dictionary
        
    Returns:
        A list of ContactInfo objects
    """
    model = config.get("configurable", {}).get("model", "gpt-4o") if config else "gpt-4o"
    
    class ContactsExtraction(BaseModel):
        contacts: List[ContactInfo] = Field(description="List of contacts extracted from the conversation")
    
    extractor = create_thread_extractor(
        model,
        schema=ContactsExtraction,
        instructions="""
        Analyze this email conversation and extract contact information for all people mentioned.
        Include:
        1. Names
        2. Email addresses
        3. Companies they work for
        4. Roles or titles
        5. Relationship to the user
        
        Only include information explicitly mentioned in the conversation.
        """
    )
    
    result = await extractor(messages)
    return result.contacts


async def extract_user_preferences(messages, config=None):
    """Extract user preferences from a conversation.
    
    Args:
        messages: List of messages in the conversation
        config: Configuration dictionary
        
    Returns:
        A list of EmailPreference objects
    """
    model = config.get("configurable", {}).get("model", "gpt-4o") if config else "gpt-4o"
    
    class PreferencesExtraction(BaseModel):
        preferences: List[EmailPreference] = Field(description="List of preferences extracted from the conversation")
    
    extractor = create_thread_extractor(
        model,
        schema=PreferencesExtraction,
        instructions="""
        Analyze this email conversation and extract any user preferences related to:
        1. Email communication style
        2. Meeting scheduling preferences
        3. Response formats or templates
        4. Tone preferences
        5. Types of emails to prioritize or ignore
        
        Include a confidence score for each preference based on how explicitly it was stated.
        """
    )
    
    result = await extractor(messages)
    return result.preferences


async def process_email_for_memory(email_data: EmailData, messages, config, store: BaseStore):
    """Process an email exchange and store relevant memories.
    
    Args:
        email_data: Email data
        messages: Messages exchanged
        config: Configuration dictionary
        store: BaseStore instance
    """
    # Extract structured information
    conversation_summary = await extract_conversation_summary(messages, config)
    contacts = await extract_contact_information(messages, config)
    preferences = await extract_user_preferences(messages, config)
    
    # Store conversation summary
    memories_namespace = get_memory_namespace(config, "memories")
    await store.aput(
        memories_namespace,
        str(uuid.uuid4()),
        {
            "content": f"Email thread: {email_data['subject']}\n" +
                      f"Key points: {', '.join(conversation_summary.key_points)}\n" +
                      (f"Action items: {', '.join(conversation_summary.action_items)}\n" if conversation_summary.action_items else "") +
                      f"Follow-up needed: {conversation_summary.follow_up_needed}"
        }
    )
    
    # Store contact information
    if contacts:
        contacts_namespace = get_memory_namespace(config, "contacts")
        for contact in contacts:
            contact_key = f"contact_{contact.name.lower().replace(' ', '_')}"
            await store.aput(
                contacts_namespace,
                contact_key,
                {"content": contact.model_dump_json()}
            )
    
    # Store preferences
    if preferences:
        preferences_namespace = get_memory_namespace(config, "preferences")
        for preference in preferences:
            pref_key = f"preference_{preference.type.lower().replace(' ', '_')}"
            await store.aput(
                preferences_namespace,
                pref_key,
                {"content": preference.model_dump_json()}
            )
    
    # Process email for background memory extraction
    manager = create_background_memory_manager(config)
    formatted_messages = convert_to_langchain_messages(messages)
    memories = await manager(formatted_messages)
    
    # Store extracted memories
    for memory in memories:
        await store.aput(
            memories_namespace,
            str(uuid.uuid4()),
            {"content": memory}
        )


def convert_to_langchain_messages(messages):
    """Convert messages to a format that LangMem can process."""
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                result.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                result.append(AIMessage(content=msg.get("content", "")))
            elif msg.get("type") == "tool":
                result.append(ToolMessage(content=msg.get("content", "")))
        else:
            # Already a LangChain message
            result.append(msg)
    return result


async def retrieve_relevant_memories(query, config, store: BaseStore, limit=5):
    """Retrieve memories relevant to the current query.
    
    Args:
        query: The search query
        config: Configuration dictionary
        store: BaseStore instance
        limit: Maximum number of memories to retrieve
        
    Returns:
        A list of relevant memories
    """
    memories_namespace = get_memory_namespace(config, "memories")
    contacts_namespace = get_memory_namespace(config, "contacts")
    preferences_namespace = get_memory_namespace(config, "preferences")
    
    # Search all namespaces
    memories = await store.asearch(memories_namespace, query=query, limit=limit)
    contacts = await store.asearch(contacts_namespace, query=query, limit=limit)
    preferences = await store.asearch(preferences_namespace, query=query, limit=limit)
    
    results = []
    
    # Format memories
    if memories:
        for memory in memories:
            results.append(f"Memory: {memory.value.get('content', '')}")
    
    # Format contacts
    if contacts:
        for contact in contacts:
            results.append(f"Contact: {contact.value.get('content', '')}")
            
    # Format preferences
    if preferences:
        for preference in preferences:
            results.append(f"Preference: {preference.value.get('content', '')}")
            
    return results


async def format_memories_for_context(memories):
    """Format memories for inclusion in prompt context.
    
    Args:
        memories: List of memory strings
        
    Returns:
        Formatted string of memories
    """
    if not memories:
        return ""
        
    result = "## Relevant Memories\n\n"
    for memory in memories:
        result += f"- {memory}\n"
    
    return result
