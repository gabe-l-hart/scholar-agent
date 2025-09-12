"""
This agent implements a scholarly research assistant using langgraph
"""

# Standard
from contextlib import contextmanager
from enum import Enum
from typing import Any, Generator
import argparse
import asyncio
import json
import os
import re
import tempfile

# Third Party
from humanfriendly.terminal import ansi_wrap
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from typing_extensions import TypedDict

## Prompts and Config ##########################################################

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


tool_names = [
    "convert_document_into_docling_document",
    "export_docling_document_to_markdown",
    "is_document_in_local_cache",
    "search_papers",
]

supervisor_prompt = """
You are a supervisor agent that manages a scholarly research workflow with three specialized agents:

1. document_ingestion - Fetches documents, converts them to markdown, and returns their content
2. article_analyzer - Analyzes documents to create summaries and identify background topics with citations
3. background_researcher - Searches for additional material on topics and identifies key sources

Your role is to:
- Analyze user requests and current state to determine which agent should handle the task
- Coordinate the workflow between agents to accomplish the user's goals
- Synthesize responses from multiple agents into coherent outputs

Current workflow logic:
- If no current article exists and user provides URL/arxiv ID -> delegate to document_ingestion
- If document_ingestion responds without actually ingesting the document -> delegate to document_ingestion
- If document exists (has document_id) but not analyzed -> delegate to article_analyzer
- If user asks about background topics -> delegate to background_researcher
- If user asks general questions -> use current context to respond directly
- If user provides search terms without specific article -> delegate to background_researcher first

Always respond with JSON indicating the next agent or END:
{"next_agent": "<AGENT_NAME>", "reasoning": "brief explanation"} where AGENT_NAME is one of document_ingestion, article_analyzer, background_researcher, or END
"""

document_ingestion_prompt = """
You are a document ingestion specialist. Your job is to:
1. Fetch documents from URLs or arxiv IDs using available tools
2. Return the document_id for the ingested document to the supervisor

Tools available: convert_document_into_docling_document, is_document_in_local_cache

Always route back to supervisor after completing ingestion.

**NOTE**: If provided with an arxiv ID, make sure to get the full pdf, not just the abstract. For example if the ID is 1234.56789, use the URL https://arxiv.org/pdf/1234.56789 not https://arxiv.org/abs/1234.56789.

Always respond with JSON indicating the document_id based on the returned document_key from the tool. For example:
{"document_id": "abcd1234"}
"""

article_analyzer_prompt = """
You are an article analysis specialist. Your job is to:
1. Analyze document content to create prose summaries of novel points
2. Identify background topics the article expects readers to understand
3. Extract citations related to each background topic

Provide detailed analysis including:
- Summary of the article's contributions
- List of background topics with associated passages and citations
- Clear organization of findings

Always route back to supervisor after analysis.
"""

background_researcher_prompt = """
You are a background research specialist. Your job is to:
1. Search for papers and articles on specific topics using search_papers tool
2. Provide summaries of found materials
3. Identify key sources for further reading

Tools available: search_papers

Focus on finding relevant academic sources and presenting clear summaries.
Always route back to supervisor after completing research.
"""

## Types #######################################################################


class ModelProvider(Enum):
    OLLAMA = "ollama"


class ScholarState(TypedDict):
    agent_messages: dict[str, list[dict]]
    current_document: dict | None
    background_topics: list
    next_agent: str | None


## Helpers #####################################################################


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


## Agents ######################################################################


def create_supervisor_agent(model):
    def supervisor_node(state: ScholarState):
        messages = state["agent_messages"].get("supervisor", [])
        current_doc = state.get("current_document")
        background_topics = state.get("background_topics", [])

        document_info = "None"
        if current_doc:
            document_info = f"Document ID: {current_doc.get('document_id', 'unknown')}, Source: {current_doc.get('source', 'unknown')}"

        context = f"""
Current document: {document_info}
Background topics covered: {len(background_topics)}
Recent messages: {messages[-3:] if len(messages) >= 3 else messages}
"""

        messages.append({"role": "user", "content": context})
        response = model.invoke(messages)

        try:
            decision = json.loads(response.content)
            next_agent = decision.get("next_agent", "END")
        except:
            next_agent = "END"

        return {"next_agent": next_agent, "messages": messages + [response]}

    return supervisor_node


def create_document_ingestion_agent(model, tools):
    ingestion_tools = [
        t
        for t in tools
        if t.name
        in [
            "convert_document_into_docling_document",
            "is_document_in_local_cache",
        ]
    ]
    agent = create_react_agent(model, ingestion_tools, prompt=document_ingestion_prompt)

    async def ingestion_node(state: ScholarState):
        messages = []
        document_id = None

        async for resp in agent.astream(
            input={"messages": state["agent_messages"].get("document_ingestion", [])},
            config={},
        ):
            if agent_response := resp.get("agent"):
                new_messages = agent_response["messages"]
            elif tool_response := resp.get("tools"):
                new_messages = tool_response["messages"]
            else:
                continue

            for msg in new_messages:
                messages.append(msg)
                # Look for document_id in the message content
                if hasattr(msg, "content") and msg.content:
                    # Extract document ID from tool responses or agent messages
                    content = msg.content
                    if "document_key" in content:

                        # Try parsing json first
                        try:
                            parsed = json.loads(content)
                            document_id = parsed.get("document_key")
                        except json.decoder.JSONDecodeError:
                            pass
                        if document_id is None:
                            # Fall back to regex matching
                            id_match = re.search(
                                r'document_key["\s:]+([^"\s,}]+)',
                                content,
                                re.IGNORECASE,
                            )
                            if id_match:
                                document_id = id_match.group(1)

        # Store the document reference in the state
        current_document = state.get("current_document", {})
        if document_id:
            current_document = {
                "source": "URL provided by user",
                "document_id": document_id,
            }

        state["agent_messages"]["document_ingestion"] = messages
        state["current_document"] = current_document
        state["next_agent"] = "supervisor"
        return state

    return ingestion_node


def create_article_analyzer_agent(model, tools):
    analyzer_tools = [
        t for t in tools if t.name in ["export_docling_document_to_markdown"]
    ]
    agent = create_react_agent(model, analyzer_tools, prompt=article_analyzer_prompt)

    async def analyzer_node(state: ScholarState):
        messages = state["agent_messages"].get("article_analyzer", [])
        current_doc = state.get("current_document")
        if not current_doc or not current_doc.get("document_id"):
            response = AIMessage("No document available for analysis")
            state["agent_messages"]["article_analyzer"] = messages + [response]
            state["next_agent"] = "supervisor"
        else:
            document_id = current_doc.get("document_id")

            # Create a message asking the agent to fetch and analyze the document
            analysis_request = f"Use the export_docling_document_to_markdown tool to fetch the content for document_id: {document_id}, then analyze the document content to provide a summary of its novel points and background topics."

            # Add the request to the messages and let the agent handle tool calls
            messages.append(HumanMessage(content=analysis_request))

            async for resp in agent.astream(input={"messages": messages}, config={}):
                if agent_response := resp.get("agent"):
                    new_messages = agent_response["messages"]
                elif tool_response := resp.get("tools"):
                    new_messages = tool_response["messages"]
                else:
                    continue

                for msg in new_messages:
                    messages.append(msg)

            state["agent_messages"]["article_analyzer"] = messages
            state["next_agent"] = "supervisor"
        return state

    return analyzer_node


def create_background_researcher_agent(model, tools):
    research_tools = [t for t in tools if t.name == "search_papers"]
    agent = create_react_agent(
        model, research_tools, prompt=background_researcher_prompt
    )

    async def researcher_node(state: ScholarState):
        messages = state["agent_messages"].get("background_researcher", [])
        async for resp in agent.astream(input={"messages": messages}, config={}):
            if agent_response := resp.get("agent"):
                new_messages = agent_response["messages"]
            elif tool_response := resp.get("tools"):
                new_messages = tool_response["messages"]
            else:
                continue
            for msg in new_messages:
                messages.append(msg)

        state["agent_messages"]["background_researcher"] = messages
        state["next_agent"] = "supervisor"
        return state

    return researcher_node


def create_scholar_workflow(model, tools):
    supervisor_agent = create_supervisor_agent(model)
    document_ingestion_agent = create_document_ingestion_agent(model, tools)
    article_analyzer_agent = create_article_analyzer_agent(model, tools)
    background_researcher_agent = create_background_researcher_agent(model, tools)

    def route_after_supervisor(state: ScholarState):
        next_agent = state.get("next_agent", "END")
        if next_agent == "END":
            return END
        return next_agent

    workflow = StateGraph(ScholarState)

    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("document_ingestion", document_ingestion_agent)
    workflow.add_node("article_analyzer", article_analyzer_agent)
    workflow.add_node("background_researcher", background_researcher_agent)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges("supervisor", route_after_supervisor)
    workflow.add_edge("document_ingestion", "supervisor")
    workflow.add_edge("article_analyzer", "supervisor")
    workflow.add_edge("background_researcher", "supervisor")

    return workflow.compile()


## Main ########################################################################


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
            tools = await load_mcp_tools(docling_session) + await load_mcp_tools(
                arxiv_session
            )
            tools = [tool for tool in tools if tool.name in tool_names]
            workflow = create_scholar_workflow(model, tools)

            state = ScholarState(
                agent_messages={},
                current_document=None,
                background_topics=[],
                next_agent=None,
            )

            # Get pre-scripted prompts
            prompts = args.prompt or []
            turn_idx = -1

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

                # Initialize the supervisor with its system prompt and the user
                # input
                supervisor_messages = state["agent_messages"].setdefault(
                    "supervisor", []
                )
                supervisor_messages.append(SystemMessage(content=supervisor_prompt))
                supervisor_messages.append(HumanMessage(content=user_input))

                try:
                    async for step in workflow.astream(state):
                        for node_name, node_output in step.items():
                            if node_name != "__end__":
                                print(f"{magenta(f'[{node_name.upper()}]')}")
                                response_color = (
                                    yellow if node_name == "supervisor" else faint
                                )

                                for msg in filter(
                                    lambda m: m in state["agent_messages"][node_name],
                                    node_output.get("agent_messages", {}).get(
                                        node_name, []
                                    ),
                                ):
                                    content = getattr(msg, "content", None)
                                    tool_calls = getattr(msg, "tool_calls", None)
                                    if tool_calls:
                                        print(magenta(f"TOOL CALL: {msg.tool_calls}"))
                                    if content:
                                        print(response_color(f"RESPONSE: {content}"))

                                state.update(node_output)

                except Exception as e:
                    print(red(f"Error: {e}"))
                    # Standard
                    import traceback

                    traceback.print_exc()
                    # DEBUG
                    breakpoint()
                    continue


if __name__ == "__main__":
    asyncio.run(main())
