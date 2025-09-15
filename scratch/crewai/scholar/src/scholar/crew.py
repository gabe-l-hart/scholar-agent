# Standard
from typing import List, cast
import os

# Third Party
from crewai import Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, crew
from crewai_tools.adapters.tool_collection import ToolCollection
from dotenv import load_dotenv
from mcp import StdioServerParameters
from scholar.tools.multi_mcp_adapter import (
    MCPServerAdapterConfig,
    MultiMCPServerAdapter,
)

load_dotenv()


@CrewBase
class Scholar:
    """Scholar crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    server_adapter: MultiMCPServerAdapter | None = None
    tool_sets: dict[str, ToolCollection] | None = None

    verbose: bool = False

    def __init__(self, *_, **__):
        """Override to also load mcp server configurations"""
        config_path = self.base_directory / "config" / "config.yaml"
        self.config = self.load_yaml(config_path)
        self.tools_config = self.config.get("tools", {})
        self.verbose = os.environ.get("VERBOSE", "") == 1 or self.config.get(
            "verbose", False
        )
        self.output_log_file: str | None = os.environ.get(
            "LOG_FILE", self.config.get("output_log_file")
        )
        for agent_config in self.config.get("agents", []):
            agent_config["verbose"] = self.verbose

    def __del__(self):
        if self.server_adapter:
            self.server_adapter.stop()

    @staticmethod
    def _parse_mcp_config(server_config: dict) -> MCPServerAdapterConfig:
        """Load the necessary inputs to MCPServerAdapter from config

        TODO: Use pydantic for full validation
        """
        isolate = server_config.get("isolate", True)
        transport = server_config.get("transport", "stdio")
        if transport == "stdio":
            env = os.environ.copy() if not isolate else {}
            env.update(server_config.get("env", {}))
            serverparams = StdioServerParameters(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=env,
            )
        elif transport == "sse":
            serverparams = {"url": server_config["url"]}
        else:
            raise ValueError(f"Unsupported transport: {transport}")
        return MCPServerAdapterConfig(
            serverparams=serverparams,
            isolate=isolate,
            kwargs=server_config.get("kwargs", {}),
        )

    def _init_mcp(self):
        if self.server_adapter is None:
            adapter_configs = [
                self._parse_mcp_config(cfg)
                for cfg in self.tools_config.get("mcpServers", {}).values()
            ]
            self.server_adapter = MultiMCPServerAdapter(*adapter_configs)
            self.tool_sets = {
                ts_name: self.server_adapter.tools.filter_by_names(
                    [n.replace("-", "_") for n in ts_names]
                )
                for ts_name, ts_names in self.tools_config.get("toolsets", {}).items()
            }

    @crew
    def crew(self) -> Crew:
        """Creates the Scholar crew"""

        # Initialize tools
        self._init_mcp()

        # Instantiate the crew
        crew = Crew(
            config=self.config,
            process=Process.hierarchical,
            manager_llm=os.getenv("MODEL"),
            verbose=self.verbose,
            output_log_file=self.output_log_file,
            chat_llm=os.getenv("MODEL"),
            planning_llm=os.getenv("MODEL"),
            planning=True,
            # DEBUG
            # planning=True,
            # planning_llm="ollama/PRIVATE/granite4-prerelease:tiny-r250825a-Q4_K_M-128k",
        )

        # Connect tools to agents
        agents_config = {a["role"]: a for a in self.config.get("agents")}
        for agent in crew.agents:
            for toolset_name in agents_config.get(agent.role, {}).get("toolsets", []):
                toolset = cast(dict, self.tool_sets or {}).get(toolset_name)
                assert (
                    toolset is not None
                ), f"Unkown toolset for {agent.role}: {toolset}"
                cast(BaseAgent, agent).tools.extend(cast(ToolCollection, toolset))

        return crew
