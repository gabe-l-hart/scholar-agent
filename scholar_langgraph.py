"""
This agent implements a scholarly research assistant using langgraph
"""

# Standard
from contextlib import contextmanager
from enum import Enum
from typing import Any, Generator
import argparse
import asyncio
import os
import tempfile

# Third Party
from humanfriendly.terminal import ansi_wrap
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

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


tool_names = [
    "convert_document_into_docling_document",
    "export_docling_document_to_markdown",
    "is_document_in_local_cache",
    "search_papers",
]

system_prompt = """
You are an AI agent for helping read scholarly articles and papers. Your goal is to respond to user input by first analyzing what the current state of the conversation is and taking the appropriate action based on the current state of the conversation. Here are your guidelines on the action to take based on the current state of the conversation:

- If there is no current article being analyzed, your goal is to find one based on the user's input. You should prompt the user to provide an article or a search term.

- If the user presents a paper or article (such as a URL, arxiv link, or arxiv ID), you should fetch the article content. This may require multiple tool calls to fetch, convert, and retrieve it. Once fully retrieved, you should provide the user a prose summary of the article's novel points and it should analyze the key topics that the article expects the reader to already understand that are not explained directly in the article. For each of these topics, you should extract any citations related to the given topic and present them alongside the topic.

- If you see a tool call response that indicates the article is not yet available, you should try fetching it again with a tool call. Do not say anything to the user until the article is fully retrieved and analyzed.

- If the user does not present an article or paper, but instead presents a topic for research, you should search for relevant papers and articles using available tools. Once papers have been found, you should present brief summaries of each and ask the user to select one for further reading.

- If the user asks for more details on a given background topic, you should use related citations in conjunction with the article search tools to find background material. It should then summarize this background material and present it to the user.

- If the user asks a general question, you should use the current context to answer it, including the current article being read and any background material that has been gathered so far.

Throughout the session, you should keep track of which paper you are currently analyzing and which background topics have been covered. You should always present your output as a JSON object with the following output format:

{
    "current_document": {
        "source": "URL or path to document",
        "cache_id": "ID of document in local cache"
    },
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
            ],
            "background_articles": {
                "article_1_source",
                "article_2_source"
            }
        }
    ],
    "content": "assistant response to the current conversation context"
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
            agent = create_react_agent(model=model, tools=tools, prompt=system_prompt)

            messages = []
            while True:
                user_input = input(blue("You: "))
                if not user_input.strip():
                    continue
                if user_input == "exit":
                    break
                messages.append({"role": "user", "content": user_input})
                async for resp in agent.astream(
                    input={"messages": messages}, config={}
                ):
                    if agent_response := resp.get("agent"):
                        new_messages = agent_response["messages"]
                    elif tool_response := resp.get("tools"):
                        new_messages = tool_response["messages"]
                    else:
                        raise RuntimeError(f"Unexpected response: {resp}")

                    for msg in new_messages:
                        messages.append(msg)
                        if msg.type == "tool":
                            print(magenta(f"TOOL RESPONSE: ") + faint(msg.content))
                        else:
                            content = msg.content
                            tool_calls = msg.tool_calls
                            if tool_calls:
                                print(magenta(f"TOOL CALL: {tool_calls}"))
                            if content:
                                # DEBUG
                                breakpoint()
                                print(yellow(f"ASSISTANT: {content}"))


if __name__ == "__main__":
    asyncio.run(main())
