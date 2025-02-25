"""Main entry point for the Executive AI Assistant with LangMem integration."""

from typing import TypedDict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from eaia.main.graph import graph
from eaia.schemas import EmailData
from eaia.memory import setup_memory_store


class EmailInput(TypedDict):
    """Input to the main application."""
    email: EmailData


# Set up memory store
try:
    logger.info("Initializing memory store...")
    memory_store = setup_memory_store()
    logger.info("Memory store initialized successfully")
except Exception as e:
    logger.error(f"Error initializing memory store: {e}")
    logger.exception(e)
    # Fall back to simpler store if needed
    from langgraph.store import InMemoryStore
    memory_store = InMemoryStore()
    logger.info("Fallback to basic InMemoryStore due to error")

# Using the memory-enhanced graph
app = graph

logger.info("Executive AI Assistant with LangMem integration initialized successfully")
