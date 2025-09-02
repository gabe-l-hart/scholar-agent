---
name: Project-wide instructions
alwaysApply: true
---

## Project description

- **Primary user flows**

  - **Document‑centric**: User uploads a document (PDF, Word, etc.) and optionally a goal prompt.
  - **Topic‑centric**: User enters a research topic prompt → system performs literature search → user selects a document → enters document‑centric flow.

- **Core functionality after a document is selected**

  1. Summarize the paper’s main findings and novel contribution.
  2. Identify essential background topics required for full comprehension.
  3. Present summary + upstream topics to the user.

- **Interaction style**

  - Free‑form chat with the document.
  - If user requests deeper info on a background topic, trigger a new literature search (topic‑centric flow).

- **Memory architecture**

  - **Session memory**: Isolated, per‑session context for the user’s current interaction.
  - **User memory**: Pieces of information the user opts to save for later; system may suggest saving useful items.
  - **System memory**: Shared across users/sessions; stores accumulated literature knowledge when the user allows it.

- **Optional UI**
  - Chat interface is primary.
  - Optional terminal or lightweight UI for document navigation and research exploration.

## Project style

- This project uses `black` for code formatting, `ruff` for linting, and `isort` for import sorting and labeling
- Unit tests can be run with `./scripts/run_tests.sh` from the root directory

## Project Components

1 Document Ingestion
• Accept PDFs, Word, DOCX, Markdown, HTML, etc.
• Extract clean text, metadata (title, authors, year, keywords).
• Chunk text into semantically coherent units (≈300–500 words).
• Compute embeddings for each chunk.

2 Literature Search
• Query external APIs (Semantic Scholar, arXiv, PubMed, Crossref).
• Return ranked list of relevant papers & metadata.
• Support pagination & filters.

3 Retrieval Service
• Vector similarity search across ingested docs.
• Hybrid retrieval (vector + keyword/semantic search).
• Return top‑N relevant chunks with doc IDs.

4 Summarization & Novelty Detection
• Summarize a full document or a set of chunks.
• Identify novel contributions vs. background.
• Output concise “abstract‑style” summary + “background topics” list. (body: )

5 Background‑Research Engine
• When user asks for deeper knowledge on a topic, automatically: a. Generate follow‑up queries. b. Perform literature search (component 2). c. Return candidate papers.

6 Memory Service
• Session Memory: short‑term context (last 20 messages + selected doc).
• User Memory: persistent key‑value store of facts the user wants to remember.
• System Memory: shared knowledge base (common facts, ontology).
• Provide “prompt‑enrichment” API to inject relevant memory into LLM calls.

7 UI Helper Service
• Convert raw LLM responses into UI widgets: _ Summaries as collapsible panels. _ Document list with thumbnails. _ Highlighted citations. _ Terminal UI for CLI mode.

8 Auth & Configuration
• User authentication (OAuth2, JWT).
• Per‑session config (remember‑me toggle, system‑memory opt‑in).
