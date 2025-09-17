"""
Shared type definitions
"""

# Standard
from enum import Enum

# Third Party
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class Agents(Enum):
    SUPERVISOR = "supervisor"
    DOCUMENT_INGESTION = "document_ingestion"
    ARTICLE_ANALYZER = "article_analyzer"
    BACKGROUND_RESEARCHER = "background_researcher"


class ModelProvider(Enum):
    OLLAMA = "ollama"


## State Types ##


class BackgroundTopic(TypedDict):
    """A single background topic and known information about it"""

    description: str
    articles: list[str]
    passages: list[str]


class ScholarState(TypedDict):
    """The state of the Scholar session seen by the user between turns"""

    document_id: str
    document_summary: str
    background_topics: list[BackgroundTopic]


class ScholarStateInternal(ScholarState):
    """The state of the Scholar session used by the agents between steps"""

    agent_messages: dict[str, list[BaseMessage]]
    next_agent: str
    prev_agent: str


## Agent Response Types ##


class DocumentIngestionOutput(TypedDict):
    document_id: str


class ArticleAnalyzerOutput(TypedDict):
    document_summary: str
    background_topics: list[BackgroundTopic]


class BackgroundResearcherOutput(TypedDict):
    background_topics: list[BackgroundTopic]
