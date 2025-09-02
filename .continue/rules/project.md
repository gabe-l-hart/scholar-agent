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