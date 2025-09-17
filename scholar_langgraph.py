"""
This agent implements a scholarly research assistant using langgraph
"""

# Standard
from contextlib import contextmanager
from typing import Any, Generator
import argparse
import asyncio
import os
import tempfile

# Third Party
from humanfriendly.terminal import ansi_wrap
from langchain_ollama import ChatOllama

# Local
from scholar_agent import config
from scholar_agent.scholar import ScholarAgentSession
from scholar_agent.types import Agents, ModelProvider, ScholarState

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


## Main ########################################################################


class NodeCallback:
    def __init__(self, truncate_responses: int | None = None):
        self.truncate_responses = truncate_responses
        self.printed_idxs = {}

    def __call__(self, agent_name: str, state: ScholarState):
        last_printed_idx = self.printed_idxs.setdefault(agent_name, 0)
        agent_messages = state["agent_messages"][agent_name]
        new_messages = agent_messages[last_printed_idx:]
        self.printed_idxs[agent_name] = len(agent_messages) - 1
        if new_messages:
            print(f"{magenta(f'[{agent_name.upper()}]')}")
            response_color = yellow if agent_name == Agents.SUPERVISOR.value else faint
            for msg in new_messages:
                if tool_calls := getattr(msg, "tool_calls", None):
                    print(magenta(f"TOOL CALL: {tool_calls}"))
                if content := getattr(msg, "content", None):
                    response_trunc = content
                    if self.truncate_responses:
                        response_trunc = response_trunc[: self.truncate_responses]
                    print(response_color(f"RESPONSE: {response_trunc}"))


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
    parser.add_argument(
        "--supervisor-thinking",
        "-t",
        action="store_true",
        help="Use thinking for the supervisor",
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
        config.mcp_config["arxiv"]["args"][-1] = paper_storage

        cb = NodeCallback()
        agent = ScholarAgentSession(
            model=model,
            config=config,
            node_callbacks=[cb],
        )
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

                await agent.user_input(user_input)


if __name__ == "__main__":
    asyncio.run(main())
