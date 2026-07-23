# CLAUDE.md — Omniord

Guidance for AI assistants (Claude Code and others) working in this repository.

> **Status: greenfield.** This repo is being built from the architecture
> blueprint below. When you implement a piece, keep this file in sync with what
> actually exists — mark modules as implemented, correct anything that drifts
> from reality, and add concrete commands as they become runnable. Treat the
> "Blueprint" sections as the target design and the "Working agreements" and
> "Current state" sections as the source of truth for what is done.

---

## 1. What Omniord is

Omniord is an **enterprise-grade, autonomous, local-first AI orchestration
framework**. It plans, decomposes, executes, verifies, and reflects on complex
multi-step workflows across local environments and cloud models.

The defining idea is a **hybrid edge/cloud engine**: local models (via Ollama)
handle routing, task splitting, and tool drafting by default, and heavy cloud
APIs (Anthropic / OpenAI) are used only as a fallback when a local model's
confidence or capability is insufficient.

---

## 2. Core architectural principles

These are non-negotiable and should shape every change:

1. **Async-first.** Every I/O-bound operation — agent calls, execution loops,
   sub-processes, HTTP — MUST use `asyncio`. No blocking I/O on the event loop.
   Use `httpx.AsyncClient`, `asyncio.subprocess`, and `asyncio.gather` for
   concurrency.
2. **Strict type safety & schemas.** Use **Pydantic v2** for all state
   representations, configuration, message schemas, and agent payload parsing.
   Prefer parsing into typed models over passing raw dicts.
3. **Deterministic tool factory with AST inspection.** Any dynamically
   generated code MUST be parsed with Python's built-in `ast` module and pass
   static safety checks *before* it is executed inside the isolated sandbox.
4. **Self-healing execution loops.** When an agent step fails or generated code
   throws, capture `stderr`/tracebacks and route them back to a reflection step
   for repair — up to **3 auto-retry attempts** — before surfacing a user-facing
   error.
5. **Local-first hybrid model engine.** Default to local Ollama endpoints
   (e.g. `llama3.1`, `qwen2.5-coder`). Escalate to cloud APIs only when local
   confidence/capability thresholds fail or the user explicitly requests maximum
   reasoning depth.

---

## 3. System topology

```
[ USER CLI / API ENTRY ]
          │
          ▼
┌───────────────────────────────────────────────────────────────┐
│              1. OMNIORD CORE ORCHESTRATOR                       │
│   Intent Parser • Task DAG Engine • Event Bus • State Manager  │
└──────────────────────────────┬────────────────────────────────┘
                               ▼
┌───────────────────────────────────────────────────────────────┐
│              2. HYBRID EDGE/CLOUD ROUTER                        │
│    Ollama / Local LLM tier  <──>  Cloud API tier (fallback)   │
└──────────────────────────────┬────────────────────────────────┘
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
 ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
 │ Dynamic Tool   │   │ Ephemeral      │   │ Executive      │
 │ Factory        │   │ Agent Swarm    │   │ Safety         │
 │ (AST + Sandbox)│   │ (Sub-Workers)  │   │ Guardrails     │
 └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
         └────────────────────┼────────────────────┘
                              ▼
┌───────────────────────────────────────────────────────────────┐
│              3. MULTI-TIER MEMORY MATRIX                        │
│   Working Context • Vector Store (sqlite-vec) • Knowledge Graph │
└───────────────────────────────────────────────────────────────┘
```

---

## 4. Directory layout (target)

Build the project to this exact layout. Each package maps to one subsystem.

```text
omniord/
├── pyproject.toml
├── README.md
├── CLAUDE.md                       # this file
├── omniord/
│   ├── __init__.py
│   ├── main.py                     # CLI entry point (Typer + Rich)
│   ├── config.py                   # Pydantic settings & env config
│   ├── core/
│   │   ├── __init__.py
│   │   ├── dag.py                  # TaskNode & DAG structure
│   │   ├── engine.py               # Async execution engine
│   │   └── events.py               # Pub/sub event bus
│   ├── router/
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract provider interface
│   │   ├── router.py               # Hybrid routing logic
│   │   └── providers/              # Ollama, Anthropic, OpenAI implementations
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── ast_checker.py          # Static AST security analysis
│   │   ├── factory.py              # Self-evolution & repair loop
│   │   ├── registry.py             # Dynamic module persistence
│   │   └── sandbox.py              # Isolated subprocess runner
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                 # Base agent interface
│   │   └── swarm.py                # Swarm orchestrator
│   ├── safety/
│   │   ├── __init__.py
│   │   └── guard.py                # Risk assessment & interactive diff
│   └── memory/
│       ├── __init__.py
│       ├── working.py              # Short-term scratchpad
│       └── store.py                # Persistent SQLite vector store
└── tests/
    ├── test_dag.py
    ├── test_router.py
    ├── test_tool_factory.py
    └── test_safety.py
```

---

## 5. Module responsibilities

### `omniord.core` — Task graph & orchestration
- **`dag.py`** — `TaskNode` model (`id`, `description`, `status`, `dependencies`,
  `outputs`) and the DAG container. Nodes are typed Pydantic models.
- **`engine.py`** — resolves dependencies via topological sort and executes
  independent branches concurrently with `asyncio.gather`.
- **`events.py`** — an internal `EventBus` that emits real-time state events for
  terminal progress rendering. Keep it pub/sub and async-friendly.

### `omniord.router` — Hybrid LLM router
- **`base.py`** — abstract provider interface with a unified surface:
  `generate()`, `stream()`, `embed()`. Hides formatting differences across
  backends.
- **`router.py`** — the tiered routing policy:
  - **Tier 0 (local fast):** intent classification, DAG generation, parameter
    extraction.
  - **Tier 1 (local code/reasoning):** code generation, tool synthesis,
    verification.
  - **Tier 2 (cloud fallback):** triggered when local generation fails
    validation, a health check/latency limit is exceeded, or the user requests
    maximum reasoning depth.
- **`providers/`** — one module per backend (Ollama, Anthropic, OpenAI), each
  implementing the `base.py` interface.

### `omniord.tools` — Self-evolving tool factory
- **`ast_checker.py`** — static analysis that blocks unsafe imports/calls
  (`os.system`, `eval`, `exec`, `subprocess.Popen` outside sandbox bounds,
  raw network sockets, destructive filesystem calls). Fail closed.
- **`sandbox.py`** — isolated subprocess runner (`asyncio.subprocess`) with
  strict execution timeouts.
- **`factory.py`** — the self-healing reflection loop:
  `generate code → AST analysis → sandbox test → repair on error → save to
  registry`. Retries are bounded (≤ 3).
- **`registry.py`** — persists validated tool modules alongside their JSON
  schemas for future retrieval.

### `omniord.agents` — Ephemeral agent swarm
- **`base.py`** — base agent interface.
- **`swarm.py`** — spawns task-focused micro-agents (`CoderAgent`, `SearchAgent`,
  `ReviewerAgent`, `SysAdminAgent`) that share a thread-safe `WorkingMemory`
  context, then self-terminate and release resources when their sub-task
  completes.

### `omniord.safety` — Executive safety layer
- **`guard.py`** — the action risk assessor:
  - `SAFE` (read-only / math / in-memory) → auto-execute.
  - `MODERATE` (file creation / API fetch) → log to console and execute.
  - `CRITICAL` (system mutation / file deletion / shell execution) → **halt,
    render a visual diff, and await explicit user confirmation.**

### `omniord.memory` — Memory matrix
- **`store.py`** — episodic store: SQLite + `sqlite-vec` (or `chromadb`) for
  vector semantic search and metadata lookups.
- **`working.py`** — high-speed in-memory scratchpad holding task state across
  the DAG execution lifecycle.

---

## 6. Tech stack

| Concern            | Choice                                             |
|--------------------|----------------------------------------------------|
| Python             | 3.11+                                               |
| Async core         | `asyncio`, `httpx`                                  |
| CLI & terminal UI  | `typer`, `rich`                                     |
| Data validation    | `pydantic>=2.0` (+ `pydantic-settings` for config)  |
| Database / vector  | `sqlite3`, `sqlite-vec` (or `chromadb`)             |
| Code inspection    | built-in `ast` module                              |
| Testing            | `pytest`, `pytest-asyncio`                          |

Local model tier: **Ollama** (default models `llama3.1`, `qwen2.5-coder`).
Cloud tier: Anthropic and OpenAI SDKs, used only on fallback.

---

## 7. Development workflow

Build **iteratively, phase by phase**. Validate each phase with `pytest` before
moving on. Do not start a phase until the previous one's tests pass.

1. **Phase 1 — Project setup & CLI.** `pyproject.toml` with the dependencies
   above; `config.py` via `pydantic-settings` (local vs. cloud endpoints);
   `main.py` with a Rich terminal UI, banner, and basic command handlers.
2. **Phase 2 — Hybrid router.** `router/base.py` streaming + non-streaming
   interfaces; Ollama and cloud connectors under `providers/`; the `Router` with
   automatic fallback on health-check failure or latency limits. Tests in
   `tests/test_router.py`.
3. **Phase 3 — DAG engine & event bus.** `core/dag.py` (`TaskNode`, topological
   sort, dependency evaluation); `core/engine.py` async execution. Tests in
   `tests/test_dag.py`.
4. **Phase 4 — AST safety, sandbox, tool factory.** `tools/ast_checker.py`,
   `tools/sandbox.py`, `tools/factory.py` (the reflection loop). Tests in
   `tests/test_tool_factory.py`.
5. **Phase 5 — Safety guardrails & swarm.** `safety/guard.py` risk levels +
   interactive confirmation; `agents/swarm.py` concurrent workers; integrate the
   guard into every worker execution step.
6. **Phase 6 — Memory & end-to-end.** `memory/store.py` (SQLite vector
   persistence); wire memory recall into the orchestrator; end-to-end test that
   auto-creates a tool, executes it, saves the result, and retrieves it from
   memory.

### Commands

```bash
# Install (editable, with dev extras)
pip install -e ".[dev]"

# Run the CLI
omniord                     # banner + hint
omniord version             # print the version
omniord config              # show the resolved local/cloud configuration
omniord run "<task>"        # orchestration is stubbed until later phases

# Run the full test suite
pytest

# Run one file's tests
pytest tests/test_config.py -q
```

---

## 8. Working agreements (conventions for every change)

- **Async or nothing.** If you introduce blocking I/O, wrap it (`asyncio.to_thread`)
  or replace it with an async equivalent. Never block the event loop.
- **Model everything.** New state, config, or message shapes are Pydantic v2
  models, not dicts. Parse at the boundary; pass typed objects inward.
- **Safety is not optional.** Any code path that generates and runs code goes
  through `ast_checker` → `sandbox`. Any action with side effects goes through
  `safety.guard` and respects its risk tiers. `CRITICAL` actions always require
  explicit confirmation — never auto-approve them.
- **Bounded self-healing.** Repair loops retry at most 3 times, then raise a
  clear user-facing error with the captured traceback.
- **Local-first routing.** Default to the local tier; only escalate to cloud on a
  real failure/threshold or explicit user request. Don't hardcode cloud calls
  into subsystems — go through the `Router`.
- **Test before advancing.** Each phase ships with passing `pytest` coverage for
  its module. Prefer `pytest-asyncio` for coroutine tests.
- **Match the code around you.** Follow existing naming, formatting, and idioms
  once the codebase has them. Do the simplest thing that satisfies the phase;
  don't build for hypothetical future requirements.
- **Secrets stay in the environment.** Read credentials from env/config only;
  never write them to files, memory, or logs.
- **Keep this file honest.** When you implement or change a subsystem, update
  the relevant section here and the "Current state" checklist below.

---

## 9. Current state

**Phase 1 is implemented.** The package (`omniord/`) has `config.py`
(`pydantic-settings`, nested local/cloud tiers) and `main.py` (Typer + Rich CLI
with a banner and the `version`, `config`, and `run` commands). `run` is a
recognized command but reports that orchestration is not yet implemented.
Tests: `tests/test_config.py`, `tests/test_cli.py` (10 tests, passing). The
`core`, `router`, `tools`, `agents`, `safety`, and `memory` subpackages do not
exist yet — they arrive in their respective phases.

- [x] Phase 1 — Project setup & CLI
- [ ] Phase 2 — Hybrid LLM router
- [ ] Phase 3 — DAG engine & event bus
- [ ] Phase 4 — AST safety, sandbox, tool factory
- [ ] Phase 5 — Safety guardrails & agent swarm
- [ ] Phase 6 — Memory system & end-to-end integration
