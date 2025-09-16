"""
This module provides an abstraction over the built in MCPServerAdapter to hold
a collection of MCP servers as a single set of tools. It is useful for managing
multiple MCP servers as a single logical server.
"""

# Standard
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import re
import tempfile

from crewai.tools import BaseTool
from crewai_tools import MCPServerAdapter
from crewai_tools.adapters.tool_collection import ToolCollection

# Third Party
from mcp import StdioServerParameters


@dataclass
class MCPServerAdapterConfig:
    serverparams: StdioServerParameters | dict
    kwargs: dict[str, any] = field(default_factory=dict)
    isolate: bool = True


class MultiMCPServerAdapter:
    """This is wrapper around MCPServerAdapter that can adapt multiple MCP
    servers into a single logical set of tools
    """

    def __init__(
        self,
        *adapter_configs: MCPServerAdapterConfig,
        tool_filters: list[str] | None = None
    ):
        self._adapter_configs = adapter_configs
        self._tool_filters = [re.compile(f) for f in (tool_filters or [])]
        self._contexts = []
        self._all_tools = []
        self.start()

    @contextmanager
    def _maybe_isolate(self, isolate: bool):
        if isolate:
            workdir = tempfile.TemporaryDirectory()
            path = workdir.__enter__()
            self._contexts.append(workdir)
            cwd = os.getcwd()
            os.chdir(path)
            yield
            os.chdir(cwd)
        else:
            yield

    def _enter_adapter(self, adapter_config: MCPServerAdapterConfig):
        with self._maybe_isolate(adapter_config.isolate):
            adapter = MCPServerAdapter(
                adapter_config.serverparams, **adapter_config.kwargs
            )
            adapter.__enter__()
            self._contexts.append(adapter)
            self._all_tools.append(adapter.tools)

    def _tool_predicate(self, tool: BaseTool) -> bool:
        return not self._tool_filters or any(
            f.match(tool.name) for f in self._tool_filters
        )

    def start(self):
        """Start the MCP servers and initialize the tools."""
        self.__enter__()

    def stop(self):
        """Stop the MCP servers"""
        self.__exit__(None, None, None)

    @property
    def tools(self) -> ToolCollection[BaseTool]:
        return ToolCollection(
            [
                tool
                for collection in self._all_tools
                for tool in collection.filter_where(self._tool_predicate)
            ]
        )

    def __enter__(self) -> ToolCollection[BaseTool]:
        for cfg in self._adapter_configs:
            self._enter_adapter(cfg)
        return self.tools

    def __exit__(self, *args, **kwargs) -> bool | None:
        out_val = None
        for ctx in self._contexts:
            if ctx.__exit__(*args, **kwargs):
                out_val = True
        return out_val
