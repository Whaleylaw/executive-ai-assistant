"""Parts of the graph that require human input with integrated memory processing."""

import uuid

from langsmith import traceable
from eaia.schemas import State, email_template
from langgraph.types import interrupt
from langchain_core.stores import BaseStore
from typing import TypedDict, Literal, Union, Optional
from langgraph_sdk import get_client
from eaia.main.config import get_config
from eaia.memory import process_email_for_memory, convert_to_langchain_messages

LGC = get_client()


class HumanInterruptConfig(TypedDict):
    allow_ignore: bool
    allow_respond: bool
    allow_edit: bool
    allow_accept: bool


class ActionRequest(TypedDict):
    action: str
    args: dict


class HumanInterrupt(TypedDict):
    action_request: ActionRequest
    config: HumanInterruptConfig
    description: Optional[str]


class HumanResponse(TypedDict):
    type: Literal["accept", "ignore", "response", "edit"]
    args: Union[None, str, ActionRequest]


TEMPLATE = """# {subject}

[Click here to view the email]({url})

**To**: {to}
**From**: {_from}

{page_content}
"""


def _generate_email_markdown(state: State):
    contents = state["email"]
    return TEMPLATE.format(
        subject=contents["subject"],
        url=f"https://mail.google.com/mail/u/0/#inbox/{contents['id']}",
        to=contents["to_email"],
        _from=contents["from_email"],
        page_content=contents["page_content"],
    )


async def save_email(state: State, config, store: BaseStore, status: str):
    namespace = (
        config["configurable"].get("assistant_id", "default"),
        "triage_examples",
    )
    key = state["email"]["id"]
    response = await store.aget(namespace, key)
    if response is None:
        data = {"input": state["email"], "triage": status}
        await store.aput(namespace, str(uuid.uuid4()), data)


def _safely_prepare_messages_for_conversion(messages):
    """
    Prepare messages for conversion to langchain format, ensuring dictionary access is used for dictionaries.
    """
    prepared_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            # Make a copy to avoid modifying the original
            prepared_msg = msg.copy()
            
            # Ensure tool_calls is properly formatted as a list if it exists
            if "tool_calls" in prepared_msg and prepared_msg["tool_calls"]:
                if not isinstance(prepared_msg["tool_calls"], list):
                    prepared_msg["tool_calls"] = [prepared_msg["tool_calls"]]
            
            prepared_messages.append(prepared_msg)
        else:
            # It's an object, so we can append it directly
            prepared_messages.append(msg)
    
    return prepared_messages


@traceable
async def send_message(state: State, config, store):
    prompt_config = get_config(config)
    memory = prompt_config["memory"]
    user = prompt_config['name']
    
    # Check if messages[-1] is a dict or an object and access tool_calls accordingly
    last_message = state["messages"][-1]
    
    # Handle both dictionary and object cases
    if isinstance(last_message, dict):
        tool_call = last_message.get("tool_calls", [{}])[0]
        tool_call_id = tool_call.get("id", "")
        tool_call_name = tool_call.get("name", "")
        tool_call_args = tool_call.get("args", {})
        msg_id = last_message.get("id", "")
    else:
        # Object-style access
        tool_calls = getattr(last_message, "tool_calls", [{}])
        tool_call = tool_calls[0] if tool_calls else {}
        tool_call_id = tool_call.get("id", "")
        tool_call_name = tool_call.get("name", "")
        tool_call_args = tool_call.get("args", {})
        msg_id = getattr(last_message, "id", "")
    
    request: HumanInterrupt = {
        "action_request": {"action": tool_call_name, "args": tool_call_args},
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,
            "allow_accept": False,
        },
        "description": _generate_email_markdown(state),
    }
    response = interrupt([request])[0]
    _email_template = email_template.format(
        email_thread=state["email"]["page_content"],
        author=state["email"]["from_email"],
        subject=state["email"]["subject"],
        to=state["email"].get("to_email", ""),
    )
    if response["type"] == "response":
        msg = {
            "type": "tool",
            "name": tool_call_name,
            "content": response["args"],
            "tool_call_id": tool_call_id,
        }
        if memory:
            await save_email(state, config, store, "email")
            # Safely prepare messages for conversion
            safe_messages = _safely_prepare_messages_for_conversion(state["messages"] + [msg])
            # Process email for memory extraction
            await process_email_for_memory(
                state["email"],
                convert_to_langchain_messages(safe_messages),
                config,
                store
            )
            rewrite_state = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Draft a response to this email:\n\n{_email_template}",
                    }
                ]
                + state["messages"],
                "feedback": f"{user} responded in this way: {response['args']}",
                "prompt_types": ["background"],
                "assistant_key": config["configurable"].get("assistant_id", "default"),
            }
            await LGC.runs.create(None, "multi_reflection_graph", input=rewrite_state)
    elif response["type"] == "ignore":
        msg = {
            "role": "assistant",
            "content": "",
            "id": msg_id,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "name": "Ignore",
                    "args": {"ignore": True},
                }
            ],
        }
        if memory:
            await save_email(state, config, store, "no")
    else:
        raise ValueError(f"Unexpected response: {response}")

    return {"messages": [msg]}


@traceable
async def send_email_draft(state: State, config, store):
    prompt_config = get_config(config)
    memory = prompt_config["memory"]
    user = prompt_config['name']
    
    # Check if messages[-1] is a dict or an object and access tool_calls accordingly
    last_message = state["messages"][-1]
    
    # Handle both dictionary and object cases
    if isinstance(last_message, dict):
        tool_call = last_message.get("tool_calls", [{}])[0]
        tool_call_id = tool_call.get("id", "")
        tool_call_name = tool_call.get("name", "")
        tool_call_args = tool_call.get("args", {})
        msg_id = last_message.get("id", "")
        msg_content = last_message.get("content", "")
    else:
        # Object-style access
        tool_calls = getattr(last_message, "tool_calls", [{}])
        tool_call = tool_calls[0] if tool_calls else {}
        tool_call_id = tool_call.get("id", "")
        tool_call_name = tool_call.get("name", "")
        tool_call_args = tool_call.get("args", {})
        msg_id = getattr(last_message, "id", "")
        msg_content = getattr(last_message, "content", "")
    
    request: HumanInterrupt = {
        "action_request": {"action": tool_call_name, "args": tool_call_args},
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": True,
            "allow_accept": True,
        },
        "description": _generate_email_markdown(state),
    }
    response = interrupt([request])[0]
    _email_template = email_template.format(
        email_thread=state["email"]["page_content"],
        author=state["email"]["from_email"],
        subject=state["email"]["subject"],
        to=state["email"].get("to_email", ""),
    )
    
    if response["type"] == "response":
        msg = {
            "type": "tool",
            "name": tool_call_name,
            "content": response["args"],
            "tool_call_id": tool_call_id,
        }
        if memory:
            await save_email(state, config, store, "email")
            # Safely prepare messages for conversion
            safe_messages = _safely_prepare_messages_for_conversion(state["messages"] + [msg])
            # Process email for memory extraction
            await process_email_for_memory(
                state["email"],
                convert_to_langchain_messages(safe_messages),
                config,
                store
            )
            rewrite_state = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Draft a response to this email:\n\n{_email_template}",
                    }
                ]
                + state["messages"],
                "feedback": f"{user} responded in this way: {response['args']}",
                "prompt_types": ["tone", "email", "background", "calendar"],
                "assistant_key": config["configurable"].get("assistant_id", "default"),
            }
            await LGC.runs.create(None, "multi_reflection_graph", input=rewrite_state)
    
    elif response["type"] == "ignore":
        msg = {
            "role": "assistant",
            "content": "",
            "id": msg_id,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "name": "Ignore",
                    "args": {"ignore": True},
                }
            ],
        }
        if memory:
            await save_email(state, config, store, "no")
    
    elif response["type"] == "edit":
        # Get the content from the response args
        corrected = response["args"]["args"].get("content", "")
        
        msg = {
            "role": "assistant",
            "content": corrected,
            "id": msg_id,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "name": tool_call_name,
                    "args": response["args"]["args"],
                }
            ],
        }
        
        if memory:
            await save_email(state, config, store, "email")
            # Safely prepare messages for conversion
            safe_messages = _safely_prepare_messages_for_conversion(state["messages"] + [msg])
            # Process email for memory extraction
            await process_email_for_memory(
                state["email"],
                convert_to_langchain_messages(safe_messages),
                config,
                store
            )
            
            # Extract content from the tool_call safely
            content_value = ""
            if isinstance(tool_call, dict) and "args" in tool_call:
                args = tool_call["args"]
                if isinstance(args, dict) and "content" in args:
                    content_value = args["content"]
            
            rewrite_state = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Draft a response to this email:\n\n{_email_template}",
                    },
                    {
                        "role": "assistant",
                        "content": content_value,
                    },
                ],
                "feedback": f"A better response would have been: {corrected}",
                "prompt_types": ["tone", "email", "background", "calendar"],
                "assistant_key": config["configurable"].get("assistant_id", "default"),
            }
            await LGC.runs.create(None, "multi_reflection_graph", input=rewrite_state)
    
    elif response["type"] == "accept":
        if memory:
            await save_email(state, config, store, "email")
            # Safely prepare messages for conversion
            safe_messages = _safely_prepare_messages_for_conversion(state["messages"])
            # Process email for memory extraction
            await process_email_for_memory(
                state["email"],
                convert_to_langchain_messages(safe_messages),
                config,
                store
            )
        return None
    
    else:
        raise ValueError(f"Unexpected response: {response}")
    
    return {"messages": [msg]}


@traceable
async def notify(state: State, config, store):
    prompt_config = get_config(config)
    memory = prompt_config["memory"]
    user = prompt_config['name']
    
    # Check if there are messages and extract the last one safely
    last_message = None
    if state["messages"] and len(state["messages"]) > 0:
        last_message = state["messages"][-1]
    
    request: HumanInterrupt = {
        "action_request": {"action": "Notify", "args": {}},
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,
            "allow_accept": False,
        },
        "description": _generate_email_markdown(state),
    }
    response = interrupt([request])[0]
    _email_template = email_template.format(
        email_thread=state["email"]["page_content"],
        author=state["email"]["from_email"],
        subject=state["email"]["subject"],
        to=state["email"].get("to_email", ""),
    )
    if response["type"] == "response":
        msg = {"type": "user", "content": response["args"]}
        if memory:
            await save_email(state, config, store, "email")
            # Safely prepare messages for conversion
            safe_messages = _safely_prepare_messages_for_conversion(state["messages"] + [msg])
            # Process email for memory extraction
            await process_email_for_memory(
                state["email"],
                convert_to_langchain_messages(safe_messages),
                config,
                store
            )
            rewrite_state = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Draft a response to this email:\n\n{_email_template}",
                    }
                ]
                + state["messages"],
                "feedback": f"{user} gave these instructions: {response['args']}",
                "prompt_types": ["email", "background", "calendar"],
                "assistant_key": config["configurable"].get("assistant_id", "default"),
            }
            await LGC.runs.create(None, "multi_reflection_graph", input=rewrite_state)
    elif response["type"] == "ignore":
        # Generate a new ID instead of trying to access it from last_message
        msg = {
            "role": "assistant",
            "content": "",
            "id": str(uuid.uuid4()),
            "tool_calls": [
                {
                    "id": "foo",
                    "name": "Ignore",
                    "args": {"ignore": True},
                }
            ],
        }
        if memory:
            await save_email(state, config, store, "no")
    else:
        raise ValueError(f"Unexpected response: {response}")

    return {"messages": [msg]}


@traceable
async def send_cal_invite(state: State, config, store):
    prompt_config = get_config(config)
    memory = prompt_config["memory"]
    user = prompt_config['name']
    
    # Check if messages[-1] is a dict or an object and access tool_calls accordingly
    last_message = state["messages"][-1]
    
    # Handle both dictionary and object cases
    if isinstance(last_message, dict):
        tool_call = last_message.get("tool_calls", [{}])[0]
        tool_call_id = tool_call.get("id", "")
        tool_call_name = tool_call.get("name", "")
        tool_call_args = tool_call.get("args", {})
        msg_id = last_message.get("id", "")
        msg_content = last_message.get("content", "")
    else:
        # Object-style access
        tool_calls = getattr(last_message, "tool_calls", [{}])
        tool_call = tool_calls[0] if tool_calls else {}
        tool_call_id = tool_call.get("id", "")
        tool_call_name = tool_call.get("name", "")
        tool_call_args = tool_call.get("args", {})
        msg_id = getattr(last_message, "id", "")
        msg_content = getattr(last_message, "content", "")
    
    request: HumanInterrupt = {
        "action_request": {"action": tool_call_name, "args": tool_call_args},
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": True,
            "allow_accept": True,
        },
        "description": _generate_email_markdown(state),
    }
    response = interrupt([request])[0]
    _email_template = email_template.format(
        email_thread=state["email"]["page_content"],
        author=state["email"]["from_email"],
        subject=state["email"]["subject"],
        to=state["email"].get("to_email", ""),
    )
    if response["type"] == "response":
        msg = {
            "type": "tool",
            "name": tool_call_name,
            "content": f"Error, {user} interrupted and gave this feedback: {response['args']}",
            "tool_call_id": tool_call_id,
        }
        if memory:
            await save_email(state, config, store, "email")
            # Safely prepare messages for conversion
            safe_messages = _safely_prepare_messages_for_conversion(state["messages"] + [msg])
            # Process email for memory extraction
            await process_email_for_memory(
                state["email"],
                convert_to_langchain_messages(safe_messages),
                config,
                store
            )
            rewrite_state = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Draft a response to this email:\n\n{_email_template}",
                    }
                ]
                + state["messages"],
                "feedback": f"{user} interrupted gave these instructions: {response['args']}",
                "prompt_types": ["email", "background", "calendar"],
                "assistant_key": config["configurable"].get("assistant_id", "default"),
            }
            await LGC.runs.create(None, "multi_reflection_graph", input=rewrite_state)
    elif response["type"] == "ignore":
        msg = {
            "role": "assistant",
            "content": "",
            "id": msg_id,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "name": "Ignore",
                    "args": {"ignore": True},
                }
            ],
        }
        if memory:
            await save_email(state, config, store, "no")
    elif response["type"] == "edit":
        msg = {
            "role": "assistant",
            "content": msg_content,
            "id": msg_id,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "name": tool_call_name,
                    "args": response["args"]["args"],
                }
            ],
        }
        if memory:
            await save_email(state, config, store, "email")
            # Safely prepare messages for conversion
            safe_messages = _safely_prepare_messages_for_conversion(state["messages"] + [msg])
            # Process email for memory extraction
            await process_email_for_memory(
                state["email"],
                convert_to_langchain_messages(safe_messages),
                config,
                store
            )
            rewrite_state = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Draft a response to this email:\n\n{_email_template}",
                    }
                ]
                + state["messages"],
                "feedback": f"{user} interrupted gave these instructions: {response['args']}",
                "prompt_types": ["email", "background", "calendar"],
                "assistant_key": config["configurable"].get("assistant_id", "default"),
            }
            await LGC.runs.create(None, "multi_reflection_graph", input=rewrite_state)
    elif response["type"] == "accept":
        if memory:
            await save_email(state, config, store, "email")
            # Safely prepare messages for conversion
            safe_messages = _safely_prepare_messages_for_conversion(state["messages"])
            # Process email for memory extraction
            await process_email_for_memory(
                state["email"],
                convert_to_langchain_messages(safe_messages),
                config,
                store
            )
        return None
    else:
        raise ValueError(f"Unexpected response: {response}")

    return {"messages": [msg]} 