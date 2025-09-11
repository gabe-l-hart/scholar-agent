"""
This agent implements a scholarly research assistant using langgraph with a supervisor multi-agent approach
"""

# Standard
from contextlib import contextmanager
from enum import Enum
from typing import Annotated, Any, Dict, Generator, List, Literal, Optional, Union
import argparse
import asyncio
import json
import os
import tempfile

# Third Party
from humanfriendly.terminal import ansi_wrap
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

# MCP configuration for document processing and arxiv search
mcp_config = {
    "docling": {
        "command": "uvx",
        "args": [
            "--quiet",
            "--no-progress",
            "--color=never",
            "--from",
            "docling-mcp@1.2.0",
            "docling-mcp-server",
        ],
        "transport": "stdio",
    },
    "arxiv": {
        "command": "uvx",
        "args": [
            "--quiet",
            "--no-progress",
            "--color=never",
            "arxiv-mcp-server",
            "--storage-path",
            "./paper_storage",
        ],
        "transport": "stdio",
    },
}


class ModelProvider(Enum):
    OLLAMA = "ollama"


# Define tool names for each agent
document_ingestion_tools = [
    "convert_document_into_docling_document",
    "export_docling_document_to_markdown",
    "is_document_in_local_cache",
]

article_analyzer_tools = []  # No specific tools needed, works with document content

background_researcher_tools = [
    "search_papers",
]

# Define the state schema for our multi-agent system
class Document(BaseModel):
    source: Optional[str] = None
    cache_id: Optional[str] = None
    content: Optional[str] = None


class BackgroundTopic(BaseModel):
    topic: str
    passages: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    background_articles: Dict[str, str] = Field(default_factory=dict)


# Define a simpler state schema for the graph
class AgentState(dict):
    """A dictionary-based state that can be easily serialized for the graph."""

    @classmethod
    def from_dict(cls, data):
        return cls(data)


# Agent prompts
document_ingestion_prompt = """
You are a document ingestion specialist. Your job is to fetch documents, convert them to markdown, and return their content.

When given a URL, arxiv ID, or document reference:
1. Check if the document is in the local cache
2. If not, convert the document into a docling document
3. Export the docling document to markdown
4. Return the markdown content

Always return your response in the following JSON format:
{
    "document": {
        "source": "URL or path to document",
        "cache_id": "ID of document in local cache",
        "content": "Full markdown content of the document"
    }
}
"""

article_analyzer_prompt = """
You are an article analysis specialist. Your job is to analyze scholarly articles and papers to:
1. Create a prose summary of the article's novel points
2. Identify key topics that the article expects the reader to already understand
3. Extract citations related to each identified topic

Always return your response in the following JSON format:
{
    "summary": "Prose summary of the article's novel points",
    "background_topics": [
        {
            "topic": "description of the topic",
            "passages": [
                "passage from the paper where the topic is present",
                "another passage on this topic from the paper"
            ],
            "citations": [
                "citation from the article relevant to the topic",
                "another citation from the article on the topic"
            ]
        }
    ]
}
"""

background_researcher_prompt = """
You are a background research specialist. Your job is to search for additional material on given topics and identify key sources for further reading.

When given a topic and related citations:
1. Search for relevant papers and articles
2. Identify the most relevant sources
3. Return a list of sources with brief descriptions

Always return your response in the following JSON format:
{
    "topic": "The topic being researched",
    "background_articles": {
        "article_1_source": "Brief description of article 1",
        "article_2_source": "Brief description of article 2"
    }
}
"""

supervisor_prompt = """
You are a supervisor for a scholarly research assistant system. Your goal is to coordinate three specialized agents to help users read and understand scholarly articles and papers.

The three agents you supervise are:
1. Document Ingestion Agent: Fetches documents, converts them to markdown, and returns their content
2. Article Analyzer Agent: Analyzes articles to create summaries and identify background topics with citations
3. Background Researcher Agent: Searches for additional material on given topics and identifies key sources

Based on the current state and user input, you need to:
1. Determine which agent should handle the current task
2. Provide clear instructions to that agent
3. Process the agent's response and decide the next steps

Here are your guidelines on the action to take based on the current state of the conversation:

- If there is no current article being analyzed, your goal is to find one based on the user's input. You should prompt the user to provide an article or a search term.

- If the user presents a paper or article (such as a URL, arxiv link, or arxiv ID), you should instruct the Document Ingestion Agent to fetch the article content. Once retrieved, you should instruct the Article Analyzer Agent to analyze it.

- If the user does not present an article or paper, but instead presents a topic for research, you should instruct the Background Researcher Agent to search for relevant papers and articles. Once papers have been found, you should present brief summaries of each and ask the user to select one for further reading.

- If the user asks for more details on a given background topic, you should instruct the Background Researcher Agent to find background material. It should then summarize this background material and present it to the user.

- If the user asks a general question, you should use the current context to answer it, including the current article being read and any background material that has been gathered so far.

Throughout the session, you should keep track of which paper is currently being analyzed and which background topics have been covered.

Always return your response in the following JSON format:
{
    "thought": "Your reasoning about the current state and what needs to be done",
    "task": "The task to be performed (INGEST_DOCUMENT, ANALYZE_ARTICLE, RESEARCH_BACKGROUND, RESPOND_TO_USER)",
    "agent": "The agent to handle the task (DOCUMENT_INGESTION, ARTICLE_ANALYZER, BACKGROUND_RESEARCHER, or SUPERVISOR)",
    "instructions": "Specific instructions for the agent",
    "response_to_user": "Your response to the user (if applicable)"
}
"""


@contextmanager
def ensure_workdir(wd: str | None) -> Generator[str, Any, None]:
    if wd:
        os.makedirs(wd, exist_ok=True)
        os.chdir(wd)
        yield wd
    else:
        with tempfile.TemporaryDirectory() as wd:
            os.chdir(wd)
            yield wd


def red(text: str) -> str:
    return ansi_wrap(text, color="red")


def yellow(text: str) -> str:
    return ansi_wrap(text, color="yellow")


def blue(text: str) -> str:
    return ansi_wrap(text, color="blue")


def magenta(text: str) -> str:
    return ansi_wrap(text, color="magenta")


def faint(text: str) -> str:
    return ansi_wrap(text, faint=True)


def create_document_ingestion_agent(model, tools):
    """Create the document ingestion agent"""
    return create_react_agent(
        model=model, tools=tools, prompt=document_ingestion_prompt
    )


def create_article_analyzer_agent(model, tools):
    """Create the article analyzer agent"""
    return create_react_agent(model=model, tools=tools, prompt=article_analyzer_prompt)


def create_background_researcher_agent(model, tools):
    """Create the background researcher agent"""
    return create_react_agent(
        model=model, tools=tools, prompt=background_researcher_prompt
    )


def create_supervisor_agent(model):
    """Create the supervisor agent"""
    return create_react_agent(model=model, tools=[], prompt=supervisor_prompt)


def route_by_task(state: AgentState) -> str:
    """Route to the next node based on the task"""
    task = state.get("task", "RESPOND_TO_USER")
    if task == "INGEST_DOCUMENT":
        return "document_ingestion_agent"
    elif task == "ANALYZE_ARTICLE":
        return "article_analyzer_agent"
    elif task == "RESEARCH_BACKGROUND":
        return "background_researcher_agent"
    elif task == "RESPOND_TO_USER":
        return "supervisor_agent"
    else:
        return "supervisor_agent"


def process_supervisor_output(state: AgentState, output: Dict) -> AgentState:
    """Process the output from the supervisor agent"""
    # Extract the supervisor's decision
    thought = output.get("thought", "")
    task = output.get("task", "RESPOND_TO_USER")
    agent = output.get("agent", "SUPERVISOR")
    instructions = output.get("instructions", "")
    response_to_user = output.get("response_to_user", "")

    # Update the state with the task and any response to the user
    new_state = AgentState(state.copy())
    new_state["task"] = task

    # If the supervisor is responding directly to the user
    if task == "RESPOND_TO_USER":
        if response_to_user:
            new_state["task_result"] = response_to_user
        return new_state

    # Otherwise, prepare instructions for the appropriate agent
    new_state["instructions"] = instructions
    return new_state


def process_document_ingestion_output(state: AgentState, output: Dict) -> AgentState:
    """Process the output from the document ingestion agent"""
    new_state = AgentState(state.copy())

    # Extract document information
    document_data = output.get("document", {})
    if document_data:
        new_state["current_document"] = document_data
        new_state["task_result"] = "Document successfully ingested"
        new_state["task"] = "ANALYZE_ARTICLE"  # Next task is to analyze the article
    else:
        new_state["task_result"] = "Failed to ingest document"
        new_state[
            "task"
        ] = "RESPOND_TO_USER"  # Return to supervisor to handle the failure

    return new_state


def process_article_analyzer_output(state: AgentState, output: Dict) -> AgentState:
    """Process the output from the article analyzer agent"""
    new_state = AgentState(state.copy())

    # Extract analysis results
    summary = output.get("summary", "")
    background_topics_data = output.get("background_topics", [])

    if summary and background_topics_data:
        new_state["background_topics"] = background_topics_data
        new_state["task_result"] = {
            "summary": summary,
            "background_topics": background_topics_data,
        }
        new_state["task"] = "RESPOND_TO_USER"  # Return to supervisor to present results
    else:
        new_state["task_result"] = "Failed to analyze article"
        new_state[
            "task"
        ] = "RESPOND_TO_USER"  # Return to supervisor to handle the failure

    return new_state


def process_background_researcher_output(state: AgentState, output: Dict) -> AgentState:
    """Process the output from the background researcher agent"""
    new_state = AgentState(state.copy())

    # Extract research results
    topic = output.get("topic", "")
    background_articles = output.get("background_articles", {})

    if topic and background_articles:
        # Update the background topic with the research results
        background_topics = new_state.get("background_topics", [])
        for bg_topic in background_topics:
            if bg_topic.get("topic") == topic:
                bg_topic["background_articles"] = background_articles
                break

        new_state["task_result"] = {
            "topic": topic,
            "background_articles": background_articles,
        }
    else:
        new_state["task_result"] = "Failed to find background research"

    new_state["task"] = "RESPOND_TO_USER"  # Return to supervisor to present results
    return new_state


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gabegoodhart/granite4-preview:tiny",
        help="The model to use for the agent",
    )
    parser.add_argument(
        "--model-provider",
        "-mp",
        choices=ModelProvider,
        default=ModelProvider.OLLAMA,
        help="The provider to use for the model",
    )
    parser.add_argument(
        "--ollama-host",
        "-oh",
        type=str,
        default=None,
        help="Non-default host to use for ollama",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Working directory for saving intermediate artifacts",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        nargs="*",
        help="Prompt(s) to script the interaction",
    )
    parser.add_argument(
        "--auto",
        "-a",
        action="store_true",
        help="Automatically exit after processing all provided prompts",
    )
    args = parser.parse_args()

    # Set up the model
    if args.model_provider == ModelProvider.OLLAMA:
        model = ChatOllama(model=args.model, base_url=args.ollama_host)
    else:
        raise ValueError(f"Unsupported model provider: {args.model_provider}")

    with ensure_workdir(args.output_dir) as workdir:
        # Overwrite paper output path
        paper_storage = os.path.join(workdir, "arxiv_papers")
        os.makedirs(paper_storage, exist_ok=True)
        mcp_config["arxiv"]["args"][-1] = paper_storage

        # Set up MCP client
        mcp_client = MultiServerMCPClient(mcp_config)

        async with (
            mcp_client.session("docling") as docling_session,
            mcp_client.session("arxiv") as arxiv_session,
        ):
            # Load tools for each agent
            all_tools = await load_mcp_tools(docling_session) + await load_mcp_tools(
                arxiv_session
            )

            # Filter tools for each agent
            doc_ingestion_tools = [
                tool for tool in all_tools if tool.name in document_ingestion_tools
            ]
            article_analyzer_tools = []  # No specific tools needed
            bg_researcher_tools = [
                tool for tool in all_tools if tool.name in background_researcher_tools
            ]

            # Create agents
            document_ingestion_agent = create_document_ingestion_agent(
                model, doc_ingestion_tools
            )
            article_analyzer_agent = create_article_analyzer_agent(
                model, article_analyzer_tools
            )
            background_researcher_agent = create_background_researcher_agent(
                model, bg_researcher_tools
            )
            supervisor_agent = create_supervisor_agent(model)

            # Create the graph
            workflow = StateGraph(AgentState)

            # Add nodes for each agent
            workflow.add_node("supervisor_agent", supervisor_agent)
            workflow.add_node("document_ingestion_agent", document_ingestion_agent)
            workflow.add_node("article_analyzer_agent", article_analyzer_agent)
            workflow.add_node(
                "background_researcher_agent", background_researcher_agent
            )

            # Define a router function
            def router(state):
                task = state.get("task", "RESPOND_TO_USER")
                if task == "INGEST_DOCUMENT":
                    return "document_ingestion_agent"
                elif task == "ANALYZE_ARTICLE":
                    return "article_analyzer_agent"
                elif task == "RESEARCH_BACKGROUND":
                    return "background_researcher_agent"
                else:
                    return "supervisor_agent"

            # Add edges with conditional routing
            workflow.add_conditional_edges(
                "supervisor_agent",
                router,
                [
                    "document_ingestion_agent",
                    "article_analyzer_agent",
                    "background_researcher_agent",
                    "supervisor_agent",
                ],
            )

            # Add edges back to supervisor
            workflow.add_edge("document_ingestion_agent", "supervisor_agent")
            workflow.add_edge("article_analyzer_agent", "supervisor_agent")
            workflow.add_edge("background_researcher_agent", "supervisor_agent")

            # Set the entry point
            workflow.set_entry_point("supervisor_agent")

            # Compile the graph
            graph = workflow.compile()

            # Initialize the state
            state = AgentState(
                {
                    "messages": [],
                    "current_document": None,
                    "background_topics": [],
                    "task": "RESPOND_TO_USER",
                    "task_result": None,
                    "next": None,
                    "instructions": None,
                }
            )

            # Get pre-scripted prompts
            prompts = args.prompt or []
            turn_idx = -1

            # Run the conversation loop
            while True:
                turn_idx += 1
                if turn_idx < len(prompts):
                    user_input = prompts[turn_idx]
                    print(blue("Prompt: ") + user_input)
                else:
                    # If auto mode is enabled and we've processed all prompts, exit
                    if args.auto and prompts:
                        print(blue("Auto mode: ") + "All prompts processed. Exiting.")
                        break

                    user_input = input(blue("You: "))
                    if not user_input.strip():
                        continue
                    if user_input == "exit":
                        break

                # Add user message to state
                if "messages" not in state:
                    state["messages"] = []
                state["messages"].append({"role": "user", "content": user_input})

                # Create a copy of the state for the graph
                graph_state = state.copy()

                # Process through the graph
                try:
                    async for event in graph.astream(graph_state):
                        print(f"Event: {event}")  # Debug print

                        # Check if event is a dictionary with expected keys
                        if isinstance(event, dict):
                            if "agent" in event:
                                # This is an agent event
                                node_name = event.get("agent", {}).get(
                                    "name", "unknown"
                                )
                                agent_output = event.get("agent", {}).get("output", {})

                                # Process agent output based on which agent it came from
                                if "supervisor_agent" in node_name:
                                    updated_state = process_supervisor_output(
                                        AgentState(graph_state), agent_output
                                    )
                                    graph_state = updated_state.copy()
                                    state = updated_state
                                elif "document_ingestion_agent" in node_name:
                                    updated_state = process_document_ingestion_output(
                                        AgentState(graph_state), agent_output
                                    )
                                    graph_state = updated_state.copy()
                                    state = updated_state
                                elif "article_analyzer_agent" in node_name:
                                    updated_state = process_article_analyzer_output(
                                        AgentState(graph_state), agent_output
                                    )
                                    graph_state = updated_state.copy()
                                    state = updated_state
                                elif "background_researcher_agent" in node_name:
                                    updated_state = (
                                        process_background_researcher_output(
                                            AgentState(graph_state), agent_output
                                        )
                                    )
                                    graph_state = updated_state.copy()
                                    state = updated_state

                                # Print agent output for debugging
                                print(
                                    magenta(f"{node_name.upper()}: ")
                                    + faint(json.dumps(agent_output, indent=2))
                                )

                            elif "end" in event:
                                # This is an end event
                                current_state = AgentState(graph_state)
                                result = current_state.get(
                                    "task_result", "No result produced"
                                )
                                if isinstance(result, dict):
                                    result = json.dumps(result, indent=2)
                                print(yellow(f"ASSISTANT: {result}"))

                                # Add assistant message to state
                                if "messages" not in state:
                                    state["messages"] = []
                                state["messages"].append(
                                    {"role": "assistant", "content": result}
                                )
                except Exception as e:
                    print(f"Error processing graph: {e}")
                    # Standard
                    import traceback

                    traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

# Made with Bob
