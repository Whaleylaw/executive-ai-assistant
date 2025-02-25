"""Agent responsible for triaging the email, can either ignore it, try to respond, or notify user.
Includes memory integration for improved contextual understanding.
"""

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.messages import RemoveMessage
from langgraph.store.base import BaseStore

from eaia.schemas import (
    State,
    RespondTo,
)
from eaia.main.fewshot import get_few_shot_examples
from eaia.main.config import get_config
from eaia.memory import retrieve_relevant_memories, format_memories_for_context


triage_prompt = """You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.

{background}. 

{name} gets lots of emails. Your job is to categorize the below email to see whether is it worth responding to.

Emails that are not worth responding to:
{triage_no}

Emails that are worth responding to:
{triage_email}

There are also other things that {name} should know about, but don't require an email response. For these, you should notify {name} (using the `notify` response). Examples of this include:
{triage_notify}

For emails not worth responding to, respond `no`. For something where {name} should respond over email, respond `email`. If it's important to notify {name}, but no email is required, respond `notify`. \

If unsure, opt to `notify` {name} - you will learn from this in the future.

{memories}

{fewshotexamples}

Please determine how to handle the below email thread:

From: {author}
To: {to}
Subject: {subject}

{email_thread}"""


async def triage_input(state: State, config: RunnableConfig, store: BaseStore):
    model = config["configurable"].get("model", "gpt-4o")
    llm = ChatOpenAI(model=model, temperature=0)
    
    # Get few-shot examples
    examples = await get_few_shot_examples(state["email"], store, config)
    
    # Retrieve relevant memories
    query = f"{state['email']['subject']} {state['email']['from_email']} {state['email']['page_content'][:200]}"
    memories = await retrieve_relevant_memories(query, config, store, limit=5)
    formatted_memories = await format_memories_for_context(memories)
    
    prompt_config = get_config(config)
    input_message = triage_prompt.format(
        email_thread=state["email"]["page_content"],
        author=state["email"]["from_email"],
        to=state["email"].get("to_email", ""),
        subject=state["email"]["subject"],
        fewshotexamples=examples,
        memories=formatted_memories,
        name=prompt_config["name"],
        full_name=prompt_config["full_name"],
        background=prompt_config["background"],
        triage_no=prompt_config["triage_no"],
        triage_email=prompt_config["triage_email"],
        triage_notify=prompt_config["triage_notify"],
    )
    model = llm.with_structured_output(RespondTo).bind(
        tool_choice={"type": "function", "function": {"name": "RespondTo"}}
    )
    response = await model.ainvoke(input_message)
    if len(state["messages"]) > 0:
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
        return {"triage": response, "messages": delete_messages}
    else:
        return {"triage": response}
