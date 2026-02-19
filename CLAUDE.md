# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **Run with hot reload**: `make dev` (uses `watchfiles` to restart on changes)
- **Run directly**: `uv run python main.py`
- **Lint**: `make lint` (flake8)
- **Test**: `make test` (pytest tests/)
- **Add dependencies**: `uv add <package>`

Requires an `OPENAI_API_KEY` in a `.env` file.

## Architecture

**Friday** is a Gradio chatbot that lets users query a CV/resume via natural language. It uses the OpenAI Responses API with streaming and tool calling.

- `main.py` — Gradio UI, OpenAI streaming loop, and tool dispatch via `handle_function_call`
- `tools/get_sections.py` — tool that lists available CV sections (profile, work experience, education, other projects, skills)
- `tools/get_section.py` — tool that returns content for a specific CV section

Each tool file exports two things: a Python function and a tool definition dict (in OpenAI function-calling schema format) used when constructing the `tools=` list passed to `client.responses.stream`.

The conversation `history` list is global and follows the OpenAI messages format. The chatbot streams text deltas directly into `history[-1]["content"]` and yields updates to the Gradio `chatbot` component.
