"""Graph definition with memory integration for Executive AI Assistant."""

import logging
from typing import Dict, Any, Callable, TypedDict, Optional
from langchain_core.stores import BaseStore

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from eaia.main.triage import triage
from eaia.main.draft_response import draft_response
from eaia.main.find_meeting_time import find_meeting_time
from eaia.main.rewrite import rewrite
from eaia.main.modified_human_inbox import send_email_draft, send_message, notify, send_cal_invite
from eaia.memory import setup_memory_store, process_email_for_memory
from eaia.schemas import State

logger = logging.getLogger(__name__)

class MemoryConfig(TypedDict):
    store: BaseStore
    enable: bool
    reflection_types: Optional[list]

def build_graph_with_memory(memory_config: Optional[MemoryConfig] = None) -> StateGraph:
    """
    Build the graph with memory integration.
    
    Args:
        memory_config: Configuration for memory
    
    Returns:
        The constructed graph
    """
    # Create a new graph
    builder = StateGraph(State)
    
    # Use provided memory store or create a new one
    memory_store = memory_config.get("store") if memory_config else setup_memory_store()
    
    # Add nodes to the graph
    builder.add_node("triage", triage)
    builder.add_node("draft_response", draft_response)
    builder.add_node("find_meeting_time", find_meeting_time)
    builder.add_node("rewrite", rewrite)
    
    # Add human interaction nodes
    builder.add_node("send_email_draft", 
                     lambda state, config: send_email_draft(state, config, memory_store))
    builder.add_node("send_message", 
                     lambda state, config: send_message(state, config, memory_store))
    builder.add_node("notify", 
                     lambda state, config: notify(state, config, memory_store))
    builder.add_node("send_cal_invite", 
                     lambda state, config: send_cal_invite(state, config, memory_store))
    
    # Define the edges of the graph
    builder.add_edge("triage", "draft_response", condition=lambda state: state.get("triage") == "email")
    builder.add_edge("triage", "notify", condition=lambda state: state.get("triage") == "notify")
    builder.add_edge("triage", END, condition=lambda state: state.get("triage") == "no")
    
    builder.add_edge("draft_response", "find_meeting_time", 
                  condition=lambda state: state.get("messages", []) and 
                                          any("meeting" in str(m.get("content", "")).lower() 
                                              for m in state.get("messages", [])))
    builder.add_edge("draft_response", "rewrite", 
                  condition=lambda state: state.get("messages", []) and 
                                         not any("meeting" in str(m.get("content", "")).lower() 
                                                for m in state.get("messages", [])))
    
    builder.add_edge("find_meeting_time", "rewrite")
    builder.add_edge("rewrite", "send_email_draft")
    
    builder.add_edge("send_email_draft", END)
    builder.add_edge("send_message", END)
    builder.add_edge("notify", END)
    builder.add_edge("send_cal_invite", END)
    
    # Set the entry point
    builder.set_entry_point("triage")
    
    return builder.compile()

# Create the graph instance
graph = build_graph_with_memory()

def setup_memory_store() -> BaseStore:
    """
    Create and configure the memory store.
    
    Returns:
        Configured memory store
    """
    # This is a placeholder - in a real implementation, this would create and
    # return an appropriate store for memory
    from langgraph.store import InMemoryStore
    return InMemoryStore() 