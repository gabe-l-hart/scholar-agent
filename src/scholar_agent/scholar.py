"""
This is the core scholar agent implementation that powers the CLI and server
"""

# Standard
from types import TracebackType
from typing import Callable
import asyncio
import json
import re
import uuid

# Third Party
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from mcp.client.session import ClientSession as MCPClientSession
import aconfig
import alog

# Local
from scholar_agent.types import Agents, ScholarState

log = alog.use_channel("SCHLR")


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


## Agents ######################################################################


class ScholarAgentSession:
    """Class encapsulating a session with the scholar agent"""

    # Public
    uuid: str
    state: ScholarState
    model: BaseChatModel
    mcp_client: MultiServerMCPClient
    tool_names: list[str]
    node_callbacks: list[Callable[[str, StateGraph], None]]
    supervisor_thinking: bool

    # Private
    _workflow: CompiledStateGraph | None
    _mcp_session_cms: list  # Context manager references
    _mcp_sessions: list[MCPClientSession]  # Actual sessions

    def __init__(
        self,
        model: BaseChatModel,
        config: aconfig.Config,
        node_callbacks: list[Callable[[str, StateGraph], None]] | None = None,
    ):
        self.uuid = str(uuid.uuid4())
        self.model = model
        self.config = config
        self.state = ScholarState(
            agent_messages={},
            current_document="",
            background_topics=[],
            next_agent="",
            prev_agent="",
        )
        self.mcp_client = MultiServerMCPClient(config.mcp_config)
        self.tool_names = config.tool_names
        self.node_callbacks = node_callbacks or []

        # Initialize message sequences for all agents
        for agent_type in Agents:
            self.state["agent_messages"][agent_type.value] = []

        # Initialize the supervisor's message sequence
        supervisor_messages = self.state["agent_messages"][Agents.SUPERVISOR.value]
        supervisor_messages.append(SystemMessage(content=config.prompts.supervisor))

        # Initialize private members that will be instantiated in start/enter
        self._workflow = None
        self._mcp_session_cms = []
        self._mcp_sessions = []

    ## Context Manager ##

    async def start(self):
        """Start the agent session by creating MCP sessions and initializing the workflow"""
        log.info("[%s] Starting session", self.uuid)
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
        self._workflow = self._create_scholar_workflow(tools)

    async def __aenter__(self):
        """Enter the async context manager"""
        await self.start()

    async def stop(self, *args, **kwargs):
        """Stop the agent session by cleaning up MCP sessions"""
        log.info("[%s] Stopping session", self.uuid)
        for cm in self._mcp_session_cms:
            # Calling __aexit__ on the contextmanager will raise a RuntimeError
            # because it's being done outside the scope it was created in, but
            # it will still successfully shut down the connection.
            try:
                await cm.__aexit__(*args, **kwargs)
            except RuntimeError:
                pass

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

    ## Public API ##

    async def user_input(self, content: str):
        """Add a user input message to the supervisor and process the result"""
        if self._workflow is None:
            raise RuntimeError(
                "Agent session not started. Call start() or use as async context manager."
            )

        user_msg = HumanMessage(content=content)
        for agent in Agents:
            self.state["agent_messages"][agent.value].append(user_msg)

        log.debug("[%s] Processing user input", self.uuid)
        log.debug2("[%s] %s", self.uuid, user_msg)
        async for step in self._workflow.astream(self.state):
            for node_name, node_output in step.items():
                log.debug("[%s] Processing agent output: %s", self.uuid, node_name)
                log.debug("[%s] %s", self.uuid, node_output)
                log.debug("")
                for callback in self.node_callbacks:
                    callback(node_name, node_output)
                if node_name == "__end__":
                    log.debug("[%s] Processing complete", self.uuid)
                    break
                self.state.update(node_output)

    ## Implementation Details ##

    def _create_scholar_workflow(self, tools):
        supervisor_agent = self._create_supervisor_agent()
        document_ingestion_agent = self._create_document_ingestion_agent(tools)
        article_analyzer_agent = self._create_article_analyzer_agent(tools)
        background_researcher_agent = self._create_background_researcher_agent(tools)

        def route_after_supervisor(state: ScholarState):
            next_agent = state.get("next_agent", "END")
            if next_agent == "END":
                return END
            return next_agent

        workflow = StateGraph(ScholarState)

        workflow.add_node(Agents.SUPERVISOR.value, supervisor_agent)
        workflow.add_node(Agents.DOCUMENT_INGESTION.value, document_ingestion_agent)
        workflow.add_node(Agents.ARTICLE_ANALYZER.value, article_analyzer_agent)
        workflow.add_node(
            Agents.BACKGROUND_RESEARCHER.value, background_researcher_agent
        )

        workflow.add_edge(START, Agents.SUPERVISOR.value)
        workflow.add_conditional_edges(Agents.SUPERVISOR.value, route_after_supervisor)
        workflow.add_edge(Agents.DOCUMENT_INGESTION.value, Agents.SUPERVISOR.value)
        workflow.add_edge(Agents.ARTICLE_ANALYZER.value, Agents.SUPERVISOR.value)
        workflow.add_edge(Agents.BACKGROUND_RESEARCHER.value, Agents.SUPERVISOR.value)

        return workflow.compile()

    def _create_supervisor_agent(self):
        def supervisor_node(state: ScholarState):
            messages = state["agent_messages"][Agents.SUPERVISOR.value]

            # Give the supervisor the summary of the current state
            messages.append(HumanMessage(content=state_summary(state)))

            response = self.model.invoke(
                messages, think=self.config.supervisor_thinking
            )
            state["agent_messages"][Agents.SUPERVISOR.value].append(response)

            try:
                decision = json.loads(response.content)
                next_agent = decision.get("next_agent", "END")
                if next_agent in Agents and (
                    instruction := decision.get("instruction")
                ):
                    state["agent_messages"][next_agent].append(
                        HumanMessage(content=instruction)
                    )
            except Exception as err:
                state["agent_messages"][Agents.SUPERVISOR.value].append(
                    AIMessage(content=f"Error parsing previous response: {err}")
                )
                next_agent = "END"

            state["prev_agent"] = Agents.SUPERVISOR.value
            state["next_agent"] = next_agent
            return state

        return supervisor_node

    def _create_document_ingestion_agent(self, tools):
        ingestion_tools = [
            t
            for t in tools
            if t.name
            in [
                "convert_document_into_docling_document",
                "is_document_in_local_cache",
            ]
        ]
        agent = create_react_agent(
            self.model,
            ingestion_tools,
            prompt=self.config.prompts.document_ingestion,
        )

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

    def _create_article_analyzer_agent(self, tools):
        analyzer_tools = [
            t for t in tools if t.name in ["export_docling_document_to_markdown"]
        ]
        agent = create_react_agent(
            self.model,
            analyzer_tools,
            prompt=self.config.prompts.article_analyzer,
        )

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

    def _create_background_researcher_agent(self, tools):
        research_tools = [t for t in tools if t.name == "search_papers"]
        agent = create_react_agent(
            self.model,
            research_tools,
            prompt=self.config.prompts.background_researcher,
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
