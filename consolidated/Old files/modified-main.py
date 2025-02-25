"""Main entry point for the Executive AI Assistant with LangMem integration."""

from typing import TypedDict

from eaia.main.graph_with_memory import graph
from eaia.schemas import EmailData
from eaia.memory import setup_memory_store


class EmailInput(TypedDict):
    email: EmailData


# Set up the memory store at startup
memory_store = setup_memory_store()

# Using the memory-enhanced graph
app = graph
