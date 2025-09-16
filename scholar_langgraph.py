"""
This agent implements a scholarly research assistant using langgraph
"""

# Standard
from contextlib import contextmanager
from enum import Enum
from types import TracebackType
from typing import Any, Generator
import argparse
import asyncio
import json
import os
import re
import tempfile
import traceback
import uuid

# Third Party
from humanfriendly.terminal import ansi_wrap
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from mcp.client.session import ClientSession as MCPClientSession
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

1. document_ingestion - Fetches documents and ingests them for later retrieval
2. article_analyzer - Analyzes ingested documents to create summaries and identify background topics with citations
3. background_researcher - Searches for additional material on topics and identifies key sources

Your role is to:
- Analyze user requests and current state to determine which agent should handle the task
- Coordinate the workflow between agents to accomplish the user's goals
- Synthesize responses from multiple agents into coherent outputs

Current workflow logic:
- If current_document is empty and user provides URL/arxiv ID -> delegate to document_ingestion
- If current_document is empty, but user only provides search terms (no URL/arxiv ID) -> delegate to background_researcher first
- If current_document is not empty but there are no background_topics -> delegate to article_analyzer
- If current_document is not empty and user asks about a background topic -> delegate to background_researcher
- If current_document is not empty and user asks general questions -> delegate to article_analyzer

Always respond with JSON indicating the next agent or END:
{"next_agent": "<AGENT_NAME>", "instruction": "instruction to next agent", "thought": "reason for decision"} where AGENT_NAME is one of document_ingestion, article_analyzer, background_researcher, or END
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


## Helpers #####################################################################


def state_summary(state: ScholarState) -> str:
    summary = {
        "current_document": state["current_document"],
        "background_topics": state["background_topics"],
        "prev_agent": state["prev_agent"],
        "prev_agent_response": "",
    }
    if prev_agent_messages := state["agent_messages"].get(state["prev_agent"]):
        summary["prev_agent_response"] = prev_agent_messages[-1].content
    return json.dumps(summary, indent=2)


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
        messages = state["agent_messages"][Agents.SUPERVISOR.value]

        # Give the supervisor the summary of the current state
        messages.append(HumanMessage(content=state_summary(state)))

        response = model.invoke(messages)
        state["agent_messages"][Agents.SUPERVISOR.value].append(response)

        try:
            decision = json.loads(response.content)
            next_agent = decision.get("next_agent", "END")
            if next_agent in Agents and (instruction := decision.get("instruction")):
                state["agent_messages"][next_agent].append(
                    HumanMessage(content=instruction)
                )
        except:
            next_agent = "END"

        state["prev_agent"] = Agents.SUPERVISOR.value
        state["next_agent"] = next_agent
        return state

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
        messages = state["agent_messages"][Agents.DOCUMENT_INGESTION.value]
        document_id = None

        async for resp in agent.astream(
            input={"messages": messages}, stream_mode="updates"
        ):
            if agent_response := resp.get("agent"):
                new_messages = agent_response["messages"]
            elif tool_response := resp.get("tools"):
                new_messages = tool_response["messages"]
            else:
                continue

            for msg in new_messages:
                messages.append(msg)
                if "document_key" in msg.content:

                    # Try parsing json first
                    try:
                        parsed = json.loads(msg.content)
                        document_id = parsed.get("document_key")

                    # Fall back to regex matching
                    except json.decoder.JSONDecodeError:
                        if id_match := re.search(
                            r'document_key["\s:]+([^"\s,}]+)',
                            msg.content,
                            re.IGNORECASE,
                        ):
                            document_id = id_match.group(1)

        # Store the document reference in the state
        if document_id:
            state["current_document"] = document_id

        state["agent_messages"][Agents.DOCUMENT_INGESTION.value] = messages
        state["next_agent"] = Agents.SUPERVISOR.value
        state["prev_agent"] = Agents.DOCUMENT_INGESTION.value
        return state

    return ingestion_node


def create_article_analyzer_agent(model, tools):
    analyzer_tools = [
        t for t in tools if t.name in ["export_docling_document_to_markdown"]
    ]
    agent = create_react_agent(model, analyzer_tools, prompt=article_analyzer_prompt)

    async def analyzer_node(state: ScholarState):
        messages = state["agent_messages"][Agents.ARTICLE_ANALYZER.value]
        current_doc = state["current_document"]
        if not current_doc:
            response = AIMessage("No document available for analysis")
            messages.append(response)
        else:
            analysis_request = f"Use the export_docling_document_to_markdown tool to fetch the content for document_id: {current_doc}, then analyze the document content to provide a summary of its novel points and background topics."
            messages.append(HumanMessage(content=analysis_request))

            async for resp in agent.astream(
                input={"messages": messages}, stream_mode="updates"
            ):
                messages.extend(resp.get("agent", {}).get("messages", []))
                messages.extend(resp.get("tools", {}).get("messages", []))

            state["agent_messages"][Agents.ARTICLE_ANALYZER.value] = messages

        state["next_agent"] = Agents.SUPERVISOR.value
        state["prev_agent"] = Agents.ARTICLE_ANALYZER.value
        return state

    return analyzer_node


def create_background_researcher_agent(model, tools):
    research_tools = [t for t in tools if t.name == "search_papers"]
    agent = create_react_agent(
        model, research_tools, prompt=background_researcher_prompt
    )

    async def researcher_node(state: ScholarState):
        messages = state["agent_messages"][Agents.BACKGROUND_RESEARCHER.value]
        async for resp in agent.astream(
            input={"messages": messages}, stream_mode="updates"
        ):
            messages.extend(resp.get("agent", {}).get("messages", []))
            messages.extend(resp.get("tools", {}).get("messages", []))

        state["agent_messages"][Agents.BACKGROUND_RESEARCHER.value] = messages
        state["next_agent"] = Agents.SUPERVISOR.value
        state["prev_agent"] = Agents.BACKGROUND_RESEARCHER.value
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

    workflow.add_node(Agents.SUPERVISOR.value, supervisor_agent)
    workflow.add_node(Agents.DOCUMENT_INGESTION.value, document_ingestion_agent)
    workflow.add_node(Agents.ARTICLE_ANALYZER.value, article_analyzer_agent)
    workflow.add_node(Agents.BACKGROUND_RESEARCHER.value, background_researcher_agent)

    workflow.add_edge(START, Agents.SUPERVISOR.value)
    workflow.add_conditional_edges(Agents.SUPERVISOR.value, route_after_supervisor)
    workflow.add_edge(Agents.DOCUMENT_INGESTION.value, Agents.SUPERVISOR.value)
    workflow.add_edge(Agents.ARTICLE_ANALYZER.value, Agents.SUPERVISOR.value)
    workflow.add_edge(Agents.BACKGROUND_RESEARCHER.value, Agents.SUPERVISOR.value)

    return workflow.compile()


class ScholarAgentSession:
    """Class encapsulating a session with the scholar agent"""

    # Public
    uuid: str
    state: ScholarState
    model: BaseChatModel
    mcp_client: MultiServerMCPClient
    tool_names: list[str]

    # Private
    _workflow: CompiledStateGraph | None
    _mcp_session_cms: list  # Context manager references
    _mcp_sessions: list[MCPClientSession]  # Actual sessions

    def __init__(
        self,
        model: BaseChatModel,
        mcp_config: dict,
        tool_names: list[str],
    ):
        self.uuid = str(uuid.uuid4())
        self.model = model
        self.state = ScholarState(
            agent_messages={},
            current_document="",
            background_topics=[],
            next_agent="",
            prev_agent="",
        )
        self.mcp_client = MultiServerMCPClient(mcp_config)
        self.tool_names = tool_names

        # Initialize message sequences for all agents
        for agent_type in Agents:
            self.state["agent_messages"][agent_type.value] = []

        # Initialize the supervisor's message sequence
        supervisor_messages = self.state["agent_messages"][Agents.SUPERVISOR.value]
        supervisor_messages.append(SystemMessage(content=supervisor_prompt))

        # Initialize private members that will be instantiated in start/enter
        self._workflow = None
        self._mcp_session_cms = []
        self._mcp_sessions = []

    async def start(self):
        """Start the agent session by creating MCP sessions and initializing the workflow"""
        # Create persistent MCP sessions for each server
        # NOTE: We need to store references to the contextmanager alongside the
        #   session to avoid the contextmanager getting exited prematurely when
        #   the CM gets garbage collected
        self._mcp_session_cms = [
            self.mcp_client.session(server_name)
            for server_name in self.mcp_client.connections
        ]
        self._mcp_sessions = await asyncio.gather(
            *[cm.__aenter__() for cm in self._mcp_session_cms]
        )

        # Get tools from all active sessions
        tools = [
            tool
            for session in self._mcp_sessions
            for tool in await (load_mcp_tools(session))
            if tool.name in self.tool_names
        ]
        self._workflow = create_scholar_workflow(self.model, tools)

    async def __aenter__(self):
        """Enter the async context manager"""
        await self.start()

    async def stop(self, *args, **kwargs):
        """Stop the agent session by cleaning up MCP sessions"""
        # Clean up MCP sessions
        for session in self._mcp_sessions:
            await session.__aexit__(*args, **kwargs)
        self._mcp_sessions.clear()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Exit the async context manager"""
        await self.stop(exc_type, exc_val, exc_tb)
        # Don't suppress exceptions
        return False

    async def user_input(self, content: str):
        """Add a user input message to the supervisor and process the result"""
        if self._workflow is None:
            raise RuntimeError(
                "Agent session not started. Call start() or use as async context manager."
            )

        user_msg = HumanMessage(content=content)
        for agent in Agents:
            self.state["agent_messages"][agent.value].append(user_msg)

        async for step in self._workflow.astream(self.state):
            for node_name, node_output in step.items():
                if node_name == "__end__":
                    break
                self.state.update(node_output)


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

        agent = ScholarAgentSession(model, mcp_config, tool_names)
        async with agent:

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

                # DEBUG
                breakpoint()

                await agent.user_input(user_input)
                # DEBUG
                print(
                    yellow(
                        agent.state["agent_messages"][Agents.SUPERVISOR.value][
                            -1
                        ].content
                    )
                )

        # # Set up MCP client
        # mcp_client = MultiServerMCPClient(mcp_config)

        # async with (
        #     mcp_client.session("docling") as docling_session,
        #     mcp_client.session("arxiv") as arxiv_session,
        # ):
        #     # tools = await load_mcp_tools(docling_session) + await load_mcp_tools(
        #     #     arxiv_session
        #     # )
        #     tools = await mcp_client.get_tools()
        #     tools = [tool for tool in tools if tool.name in tool_names]
        #     workflow = create_scholar_workflow(model, tools)

        #     #DEBUG
        #     breakpoint()

        #     # Initialize the state
        #     state = ScholarState(
        #         agent_messages={},
        #         current_document="",
        #         background_topics=[],
        #         next_agent="",
        #         prev_agent="",
        #     )
        #     for agent_type in Agents:
        #         state["agent_messages"][agent_type.value] = []

        #     # Initialize the supervisor's message sequence
        #     supervisor_messages = state["agent_messages"][Agents.SUPERVISOR.value]
        #     supervisor_messages.append(SystemMessage(content=supervisor_prompt))

        #     # Get pre-scripted prompts
        #     prompts = args.prompt or []
        #     turn_idx = -1

        #     while True:
        #         turn_idx += 1
        #         if turn_idx < len(prompts):
        #             user_input = prompts[turn_idx]
        #             print(blue("Prompt: ") + user_input)
        #         else:
        #             # If auto mode is enabled and we've processed all prompts, exit
        #             if args.auto and prompts:
        #                 print(blue("Auto mode: ") + "All prompts processed. Exiting.")
        #                 break

        #             user_input = input(blue("You: "))
        #             if not user_input.strip():
        #                 continue
        #             if user_input == "exit":
        #                 break

        #         # Add the user message to all agents' history
        #         user_msg = HumanMessage(content=user_input)
        #         for agent in Agents:
        #             state["agent_messages"][agent.value].append(user_msg)

        #         try:
        #             async for step in workflow.astream(state):
        #                 for node_name, node_output in step.items():
        #                     if node_name == "__end__":
        #                         break

        #                     print(f"{magenta(f'[{node_name.upper()}]')}")
        #                     response_color = (
        #                         yellow if node_name == "supervisor" else faint
        #                     )

        #                     for msg in filter(
        #                         lambda m: m in state["agent_messages"][node_name],
        #                         node_output.get("agent_messages", {}).get(
        #                             node_name, []
        #                         ),
        #                     ):
        #                         content = getattr(msg, "content", None)
        #                         tool_calls = getattr(msg, "tool_calls", None)
        #                         if tool_calls:
        #                             print(magenta(f"TOOL CALL: {msg.tool_calls}"))
        #                         if content:
        #                             print(response_color(f"RESPONSE: {content}"))

        #                     state.update(node_output)

        #         except Exception as e:
        #             print(red(f"Error: {e}"))
        #             traceback.print_exc()
        #             continue


if __name__ == "__main__":
    asyncio.run(main())
