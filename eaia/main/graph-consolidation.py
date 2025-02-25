"""Overall agent with enhanced memory capabilities."""
import json
import logging
from typing import TypedDict, Literal, Dict, Any, Optional
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from eaia.main.triage import (
    triage_input,
)
from eaia.main.draft_response import draft_response
from eaia.main.find_meeting_time import find_meeting_time
from eaia.main.rewrite import rewrite
from eaia.main.config import get_config
from langchain_core.messages import ToolMessage
from eaia.main.human_inbox import (
    send_message,
    send_email_draft,
    notify,
    send_cal_invite,
)
from eaia.gmail import (
    send_email,
    mark_as_read,
    send_calendar_invite,
)
from eaia.schemas import (
    State,
)
from eaia.memory import (
    retrieve_relevant_memories,
    process_email_for_memory,
    setup_memory_store,
)
import uuid

logger = logging.getLogger(__name__)


def route_after_triage(
    state: State,
) -> Literal["draft_response", "mark_as_read_node", "notify"]:
    """Route to the appropriate node based on triage result."""
    try:
        if state["triage"].response == "email":
            return "draft_response"
        elif state["triage"].response == "no":
            return "mark_as_read_node"
        elif state["triage"].response == "notify":
            return "notify"
        elif state["triage"].response == "question":
            return "draft_response"
        else:
            logger.warning(f"Unknown triage response: {state['triage'].response}")
            return "notify"  # Default to notify if unknown
    except Exception as e:
        logger.error(f"Error in route_after_triage: {e}")
        return "notify"  # Default to notify in case of error


def take_action(
    state: State,
) -> Literal[
    "send_message",
    "rewrite",
    "mark_as_read_node",
    "find_meeting_time",
    "send_cal_invite",
    "process_memory_node",
    "bad_tool_name",
]:
    """Determine which action to take based on the tool call."""
    try:
        prediction = state["messages"][-1]
        if len(prediction.tool_calls) != 1:
            raise ValueError("Expected exactly one tool call")
        
        tool_call = prediction.tool_calls[0]
        tool_name = tool_call["name"]
        
        if tool_name == "Question":
            return "send_message"
        elif tool_name == "ResponseEmailDraft":
            return "rewrite"
        elif tool_name == "Ignore":
            return "mark_as_read_node"
        elif tool_name == "MeetingAssistant":
            return "find_meeting_time"
        elif tool_name == "SendCalendarInvite":
            return "send_cal_invite"
        # Handle memory tool calls
        elif tool_name in ["ManageMemory", "SearchMemory"]:
            return "process_memory_node"
        else:
            return "bad_tool_name"
    except Exception as e:
        logger.error(f"Error in take_action: {e}")
        return "bad_tool_name"  # Default to bad_tool_name in case of error


def bad_tool_name(state: State):
    """Handle cases where the tool name is not recognized."""
    try:
        tool_call = state["messages"][-1].tool_calls[0]
        message = f"Could not find tool with name `{tool_call['name']}`. Make sure you are calling one of the allowed tools!"
        last_message = state["messages"][-1]
        
        # Remove any colons from the tool name for consistency
        if isinstance(last_message.tool_calls[0]["name"], str):
            last_message.tool_calls[0]["name"] = last_message.tool_calls[0]["name"].replace(
                ":", ""
            )
        
        return {
            "messages": [
                last_message,
                ToolMessage(content=message, tool_call_id=tool_call["id"]),
            ]
        }
    except Exception as e:
        logger.error(f"Error in bad_tool_name: {e}")
        # Generate a generic error message if we can't process the tool call
        return {
            "messages": [
                ToolMessage(
                    content="An error occurred while processing the tool call.",
                    tool_call_id=str(uuid.uuid4())
                ),
            ]
        }


async def process_memory_node(state: State, config, store):
    """Process memory tool calls and respond accordingly."""
    try:
        tool_call = state["messages"][-1].tool_calls[0]
        tool_name = tool_call["name"]
        
        # Generate a response based on the memory tool call
        if tool_name == "ManageMemory":
            message = "Memory stored successfully. You can now continue drafting your response."
        elif tool_name == "SearchMemory":
            # If this was a search request, try to include some relevant memories in the response
            query = tool_call["args"].get("query", "")
            if query:
                memories = await retrieve_relevant_memories(query, config, store, limit=3)
                if memories:
                    memory_text = "\n".join(memories)
                    message = f"Here are some relevant memories I found:\n\n{memory_text}\n\nYou can now continue drafting your response."
                else:
                    message = "I searched but couldn't find any relevant memories. You can now continue drafting your response."
            else:
                message = "Memory search completed. You can now continue drafting your response."
        else:
            message = "Memory operation processed."
        
        return {
            "messages": [
                ToolMessage(content=message, tool_call_id=tool_call["id"]),
            ]
        }
    except Exception as e:
        logger.error(f"Error in process_memory_node: {e}")
        return {
            "messages": [
                ToolMessage(
                    content="An error occurred while processing memory. Please try again.",
                    tool_call_id=str(uuid.uuid4())
                ),
            ]
        }


def enter_after_human(
    state,
) -> Literal[
    "mark_as_read_node", "draft_response", "send_email_node", "send_cal_invite_node"
]:
    """Determine where to go after human interaction."""
    try:
        messages = state.get("messages") or []
        if len(messages) == 0:
            if state["triage"].response == "notify":
                return "mark_as_read_node"
            raise ValueError("No messages found")
        else:
            last_message = messages[-1]
            if isinstance(last_message, (ToolMessage, HumanMessage, BaseMessage)):
                return "draft_response"
            else:
                execute = last_message.tool_calls[0]
                if execute["name"] == "ResponseEmailDraft":
                    return "send_email_node"
                elif execute["name"] == "SendCalendarInvite":
                    return "send_cal_invite_node"
                elif execute["name"] == "Ignore":
                    return "mark_as_read_node"
                elif execute["name"] == "Question":
                    return "draft_response"
                else:
                    logger.warning(f"Unknown tool name after human: {execute['name']}")
                    return "draft_response"  # Default to draft_response if unknown
    except Exception as e:
        logger.error(f"Error in enter_after_human: {e}")
        return "draft_response"  # Default to draft_response in case of error


def send_cal_invite_node(state, config):
    """Send a calendar invitation."""
    try:
        tool_call = state["messages"][-1].tool_calls[0]
        _args = tool_call["args"]
        email = get_config(config)["email"]
        try:
            send_calendar_invite(
                _args["emails"],
                _args["title"],
                _args["start_time"],
                _args["end_time"],
                email,
            )
            message = "Sent calendar invite!"
        except Exception as e:
            message = f"Got the following error when sending a calendar invite: {e}"
        return {"messages": [ToolMessage(content=message, tool_call_id=tool_call["id"])]}
    except Exception as e:
        logger.error(f"Error in send_cal_invite_node: {e}")
        return {
            "messages": [
                ToolMessage(
                    content="An error occurred while sending the calendar invite.",
                    tool_call_id=str(uuid.uuid4())
                ),
            ]
        }


def send_email_node(state, config):
    """Send an email."""
    try:
        tool_call = state["messages"][-1].tool_calls[0]
        _args = tool_call["args"]
        email = get_config(config)["email"]
        new_receipients = _args["new_recipients"]
        if isinstance(new_receipients, str):
            new_receipients = json.loads(new_receipients)
        send_email(
            state["email"]["id"],
            _args["content"],
            email,
            addn_receipients=new_receipients,
        )
    except Exception as e:
        logger.error(f"Error in send_email_node: {e}")


def mark_as_read_node(state):
    """Mark an email as read."""
    try:
        mark_as_read(state["email"]["id"])
        return {
            "messages": [
                ToolMessage(
                    content="Email marked as read",
                    tool_call_id=state["messages"][-1].tool_calls[0]["id"] if state.get("messages") else str(uuid.uuid4())
                )
            ]
        }
    except Exception as e:
        logger.error(f"Error in mark_as_read_node: {e}")
        return {
            "messages": [
                ToolMessage(
                    content="An error occurred while marking the email as read.",
                    tool_call_id=str(uuid.uuid4())
                ),
            ]
        }


def human_node(state: State):
    """Node for human interaction - no action needed."""
    pass


class ConfigSchema(TypedDict):
    db_id: int
    model: str


def build_memory_enhanced_graph():
    """Build the graph with memory enhancements."""
    graph_builder = StateGraph(State, config_schema=ConfigSchema)
    graph_builder.add_node(human_node)
    graph_builder.add_node(triage_input)
    graph_builder.add_node(draft_response)
    graph_builder.add_node(send_message)
    graph_builder.add_node(rewrite)
    graph_builder.add_node(mark_as_read_node)
    graph_builder.add_node(send_email_draft)
    graph_builder.add_node(send_email_node)
    graph_builder.add_node(bad_tool_name)
    graph_builder.add_node(notify)
    graph_builder.add_node(send_cal_invite_node)
    graph_builder.add_node(send_cal_invite)
    # Add memory processing node
    graph_builder.add_node(process_memory_node)
    
    graph_builder.add_conditional_edges("triage_input", route_after_triage)
    graph_builder.set_entry_point("triage_input")
    graph_builder.add_conditional_edges("draft_response", take_action)
    graph_builder.add_edge("send_message", "human_node")
    graph_builder.add_edge("send_cal_invite", "human_node")
    graph_builder.add_node(find_meeting_time)
    graph_builder.add_edge("find_meeting_time", "draft_response")
    graph_builder.add_edge("bad_tool_name", "draft_response")
    graph_builder.add_edge("send_cal_invite_node", "draft_response")
    graph_builder.add_edge("send_email_node", "mark_as_read_node")
    graph_builder.add_edge("rewrite", "send_email_draft")
    graph_builder.add_edge("send_email_draft", "human_node")
    graph_builder.add_edge("mark_as_read_node", END)
    graph_builder.add_edge("notify", "human_node")
    # Add edge from process_memory_node back to draft_response
    graph_builder.add_edge("process_memory_node", "draft_response")
    graph_builder.add_conditional_edges("human_node", enter_after_human)
    
    return graph_builder.compile()


# Initialize the memory store
memory_store = setup_memory_store()

# Build the graph with memory enhancements
graph = build_memory_enhanced_graph()
