"""
Shared type definitions
"""

# Standard
from enum import Enum

# Third Party
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class ScholarState(TypedDict):
    agent_messages: dict[str, list[BaseMessage]]
    current_document: str
    background_topics: list
    next_agent: str
    prev_agent: str


class Agents(Enum):
    SUPERVISOR = "supervisor"
    DOCUMENT_INGESTION = "document_ingestion"
    ARTICLE_ANALYZER = "article_analyzer"
    BACKGROUND_RESEARCHER = "background_researcher"


class ModelProvider(Enum):
    OLLAMA = "ollama"
