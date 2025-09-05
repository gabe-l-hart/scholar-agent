# Standard
from typing import List, cast
import os

# Third Party
from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai_tools.adapters.tool_collection import ToolCollection
from mcp import StdioServerParameters

from scholar.tools.multi_mcp_adapter import (
    MCPServerAdapterConfig,
    MultiMCPServerAdapter,
)

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class Scholar:
    """Scholar crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    server_adapter: MultiMCPServerAdapter | None = None
    tool_sets: dict[str, ToolCollection] | None = None

    def __init__(self, *_, **__):
        """Override to also load mcp server configurations"""
        tools_config_path = self.base_directory / "config" / "tools.yaml"
        self.tools_config = self.load_yaml(tools_config_path)

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

    ## Agents ##

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def background_researcher(self) -> Agent:
        self._init_mcp()
        cfg = self.agents_config["background_researcher"]
        tools = []
        for toolset in cfg.pop("toolsets", []):
            tools.extend(cast(dict, self.tool_sets)[toolset])
        return Agent(config=cfg, verbose=True, tools=tools)

    @agent
    def research_citation_reviewer(self) -> Agent:
        self._init_mcp()
        return Agent(
            config=self.agents_config["research_citation_reviewer"],  # type: ignore[index]
            verbose=True,
        )

    ## Tasks ##

    @task
    def conduct_article_search(self) -> Task:
        return Task(
            config=self.tasks_config["conduct_article_search"],  # type: ignore[index]
        )

    @task
    def convert_document_task(self) -> Task:
        return Task(
            config=self.tasks_config["convert_document_task"],  # type: ignore[index]
        )

    @task
    def find_background_topics_task(self) -> Task:
        return Task(
            config=self.tasks_config["find_background_topics_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Scholar crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # output_log_file="logs.json",
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
