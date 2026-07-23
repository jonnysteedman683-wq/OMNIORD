# Omniord

An enterprise-grade, autonomous, **local-first** AI orchestration framework. It
plans, decomposes, executes, verifies, and reflects on complex multi-step
workflows across local environments and cloud models.

Omniord prefers local models (via Ollama) for routing, task splitting, and tool
drafting, and escalates to cloud APIs (Anthropic / OpenAI) only when a local
model's confidence or capability is insufficient.

> **Status:** early development. **Phase 1 (project setup & CLI)** is in place.
> See [`CLAUDE.md`](./CLAUDE.md) for the full architecture blueprint, the
> phased build plan, and the conventions every change must follow.

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
omniord                      # show the banner
omniord version             # print the version
omniord config              # show the resolved configuration (local + cloud tiers)
omniord run "your task"     # orchestration lands in later phases
```

## Configuration

Settings come from environment variables (prefix `OMNIORD_`, `__` to descend
into a group) or a local `.env` file:

```bash
OMNIORD_LOCAL__FAST_MODEL=llama3.1
OMNIORD_LOCAL__CODE_MODEL=qwen2.5-coder
OMNIORD_CLOUD__PROVIDER=anthropic
OMNIORD_CLOUD__ANTHROPIC_API_KEY=sk-ant-...
OMNIORD_MAX_RETRIES=3
```

## Development

```bash
pytest                      # run the test suite
```

The project is built phase by phase; each phase ships with passing tests before
the next begins.
