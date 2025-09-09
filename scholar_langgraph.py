# Standard
import asyncio
import os

# Third Party
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

mcp_client = MultiServerMCPClient(
    {
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
)

tool_names = [
    "convert_document_into_docling_document",
    "export_docling_document_to_markdown",
    "is_document_in_local_cache",
    "search_papers",
]

model = ChatOllama(
    model=os.getenv("MODEL", "gabegoodhart/granite4-preview:tiny"),
)

system_prompt = """
You are an AI agent for helping read scholarly articles and papers. Your goal is to respond to user input by first analyzing what the current state of the conversation is and taking the appropriate action based on the current state of the conversation. Here are your guidelines on the action to take based on the current state of the conversation:

- If the user presents a paper or article (such as a URL, arxiv link, or arxiv ID), you should fetch the article content and provide it to the user. Additionally, you should provide the user a prose summary of the article's novel points and it should analyze the key topics that the article expects the reader to already understand that are not explained directly in the article. For each of these topics, you should extract any citations related to the given topic and present them alongside the topic.

- If the user does not present an article or paper, but instead presents a topic for research, you should search for relevant papers and articles using available tools. Once papers have been found, you should present brief summaries of each and ask the user to select one for further reading.

- If the user asks for more details on a given background topic, you should use related citations in conjunction with the article search tools to find background material. It should then summarize this background material and present it to the user.

- If the user asks a general question, you should use the current context to answer it, including the current article being read and any background material that has been gathered so far.

Throughout the session, you should keep track of which paper you are currently analyzing and which background topics have been covered. Each time you respond to the user, you should present your response in the following format, leaving sections blank if the user has not yet selected a paper to focus on:

---- CURRENT ARTICLE ----
This is the markdown representation of the current article

---- SUMMARY ----
This is a short prose summary of the current article's novel contributions or conclusions

--- BACKGROUND TOPICS ----
This is the list of identified background topics a user needs to understand in order to understand the paper. For each background topic, you should list any relevant citations from the article's citations section (if it has one), and the most relevant passages from the article referencing this topic (no more than 3).
"""


async def main():
    async with (
        mcp_client.session("docling") as docling_session,
        mcp_client.session("arxiv") as arxiv_session,
    ):
        tools = await load_mcp_tools(docling_session) + await load_mcp_tools(
            arxiv_session
        )
        tools = [tool for tool in tools if tool.name in tool_names]
        agent = create_react_agent(model=model, tools=tools)

        messages = []
        while True:
            user_input = input("You: ")
            if user_input == "exit":
                break
            messages.append({"role": "user", "content": user_input})
            n_input_msgs = len(messages)
            response = await agent.ainvoke(input={"messages": messages}, config={})
            messages = response["messages"]
            for msg in messages[n_input_msgs:]:
                # #DEBUG
                # breakpoint()
                if msg.type == "tool":
                    print(f"TOOL RESPONSE: {msg.content}")
                else:
                    content = msg.content
                    tool_calls = msg.tool_calls
                    if tool_calls:
                        print(f"TOOL CALL: {tool_calls}")
                    if content:
                        print(f"ASSISTANT: {content}")


if __name__ == "__main__":
    asyncio.run(main())
