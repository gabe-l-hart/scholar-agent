# Dev workflow

## Define problem statement

- Is the problem good for an agentic solution?
  - Does it require a bunch of different steps through different systems?
  - Are there many different ways the steps could be combined depending on the task?
- Will it be helpful if the agent gets the solution part-way correct but you have to do the rest?

## Develop agent logic

- What does the input look like to start a session?
- What steps should the agent be able to take?
- Which steps of the task should require a human to verify?
- Which steps are prone to error? How can/should the agent recover from errors?

## Connect data sources

- Public data via web search tool
  - This almost _always_ requires an API Key of some sort. How do you hide this from the LLM?
- For private data sources, how do you protect your secrets? How do you detect expired secrets and recommend fixes?
- How should data sources be presented to the agent? Should they be

## Select tool components

- What pre-existing tool servers (MCP) are out there that can solve parts of your problem?
- What secrets do these tools require?
- What is the form-factor for accessing the MCP server(s) (stdio command, docker, pre-existing server)?

## Build workflow

- System prompt engineering
- Decide on how much autonomy the agent should have when planning
  - Autonomous: ReAct / ReWoo + eventing driven by external sources to stimulate agent actions
  - Full reasoning: ReAct / ReWoo
  - Partial reasoning: Code written to dictate flow boundaries, but agent can decide the path through the workflow
  - Rigid: Code written to dictate flow boundaries. LLM calls used to facilitate chat experience, but flow is strictly controlled

## Make extensible

- Add extensibility to your problem space depending on the problem.
  - For fully open-ended agents, this could be plugging in MCP servers
  - For more constrained problem spaces, this could be adding new data sources (eg additional GitHub repos to scan)

## Deploy

- Determine the full set of components that need to run to make the system work (central agent, MCP servers, models)
- Find the serving framework that best matches the architecture (eg existing agent framework like wxO, models hosted in an existing service w/ agent in raw kubernetes)

# Project Ideas

## Deep Granite

### Define problem statement

Given a topic or a seed paper, conduct research using papers, public sources, and private sources. Identify topics that may need further clarification (or let the user select) and perform follow-up research to fill in user's understanding.

### Develop agent logic

- Start with a prompt:

  - Search relevant papers / blogs
  - Present article candidates for seed
  - Proceed with each for as if seed paper, but keep context of all for cross-reference

  - Start with a paper:
    - Summarize
    - Identify key dependency topics and present them to the user
    - Allow user to select dependency topic (or give own) and perform follow-up research in the background while user does other things (chat with doc, just read it, etc)
    - Once follow up research available, summarize and incorporate in overall topic summary

### Choose model

- Try local model with strong long-context and RAG support (granite 4 tiny?)
- Compare to hosted larger models
- Keep configurable

### Connect data sources

- Web search w/out rate limit issues
  - Maybe walk you through sign-up for search API key?
- ArXiv search / download
- Docling MCP
- GitHub search
- HuggingFace search?

### Select tool components

- Memory (session, global cache?, user?)
  - Tools for saving / retrieving memory
  - Session RAG?
- Notify user of background research?

# SCRATCH NOTES

## Define problem statement

- Write it down in words! This can be (a) very helpful to crystalize your vision, and (b) the seed of a prompt to an AI assistant to help in the development work.
