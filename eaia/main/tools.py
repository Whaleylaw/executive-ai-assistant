"""Tools for the Executive AI Assistant."""

from eaia.schemas import (
    NewEmailDraft,
    ResponseEmailDraft,
    Question,
    MeetingAssistant,
    SendCalendarInvite,
    Ignore,
)

# Define the tools available for the LLM to use
# These are Pydantic models defined in schemas.py
tools = [
    NewEmailDraft,
    ResponseEmailDraft,
    Question,
    MeetingAssistant,
    SendCalendarInvite,
    Ignore,
] 