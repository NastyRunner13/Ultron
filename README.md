# Ultron

> A self-evolving AI agent that version-controls its own mind.

Ultron is an experimental AI agent that improves itself over time using a Git-like evolution tree. Every proven state is a *commit*. Improvements happen on *branches*. Successful experiments get *merged* into `main`. Failed branches are *killed* — but their postmortems become memory, so the agent never repeats the same mistake.

---

## Quick Start

```bash
# Prerequisites: Python 3.11+, uv

# Install dependencies
uv sync --all-extras

# Copy and configure your API key
cp .env.example .env
# Edit .env and add at least one provider key (OpenRouter, OpenAI, Anthropic, Groq, or Ollama)

# Run a single task
uv run python -m ultron.main --task "List the files in the current directory"

# Interactive mode
uv run python -m ultron.main

# Run tests
uv run pytest tests/ -v
```

---

## The Core Idea

Most AI agents are static. You configure them once and they stay that way. Ultron treats the agent's configuration — its model, system prompt, tools, and self-written code — as a living artifact that can be branched, mutated, evaluated, and evolved. The result is an agent with a full evolutionary lineage: a record of every version of itself, why each change was made, and what was learned from every failure.

```
main:  C0 ─── C1 ─── C2 ──────────────── C5 (merge) ─── C6
                │                           ^
                ├── exp/try-claude: C3 ─────┘  (merged: +12% reasoning)
                │
                └── exp/add-tool: C4 ✗  (dead: tool caused hallucinations)
                          │
                          └── postmortem → failure memory
```

---

## Key Concepts

**Body (Blueprint)** — A fully serializable agent configuration: LLM model, system prompt, tools, and self-written code modules. Defined as a Pydantic v2 model with YAML/JSON serialization, content hashing, and structural diffing. The thing that evolves.

**Consciousness** — The persistent identity that survives body swaps. Holds long-term memory, goals, and the reasoning engine that decides *how* to evolve.

**Evolution Tree** — A Git-like DAG tracking the agent's entire history. Supports commit, branch, merge, kill, cherry-pick, diff, bisect, and tag.

**Arena** — A tiered benchmark suite that scores each body across reasoning, coding, creativity, tool use, and meta-tasks. The fitness function that determines whether a branch lives or dies.

**Failure Memory** — Structured postmortems from dead branches. Reviewed before every evolution attempt so Ultron never tries the same failed approach twice.

**Inner Voice** — Three internal perspectives that debate before every evolution decision: Explorer (push for radical change), Conservative (argue for stability), Strategist (plan which capability to unlock next).

---

## Architecture

```
ultron/
├── core/
│   ├── settings.py              # YAML + env var config loader (Pydantic)
│   └── logging.py               # Loguru + Rich structured logging
├── body/
│   ├── blueprint.py             # Blueprint, ModelConfig, ToolSpec schemas
│   ├── llm.py                   # Async LiteLLM client (multi-provider, retry, token tracking)
│   ├── agent.py                 # ReAct loop (reason → tool call → observe → respond)
│   └── factory.py               # BodyFactory — Blueprint → working AgentBody
├── tools/
│   ├── registry.py              # Dynamic tool import, async execution, OpenAI format
│   └── builtins/
│       ├── browse.py            # Async URL fetch + text extraction (httpx + BS4)
│       ├── shell.py             # Async subprocess with blocked-command safety
│       └── filesystem.py        # Async read/write/delete/list (aiofiles)
├── main.py                      # CLI entry point (single task + interactive mode)
└── __main__.py                  # python -m ultron
config/
├── genesis_blueprint.yaml       # Gen 0 body — default model + all built-in tools
└── settings.yaml                # Token budgets, safety rules, thresholds
tests/
├── test_blueprint.py            # Schema serialization, hashing, diffing
├── test_tools.py                # Registry, built-in tool execution, safety
└── test_agent.py                # ReAct loop, LLM client, token tracking
```

### Planned (Future Phases)

```
ultron/
├── consciousness/               # Memory, identity, goals, dream state
├── tree/                        # Evolution tree: commits, branches, merge, diff, bisect
├── evolution/                   # Reflection, mutation generators, tournament, journal
├── arena/                       # Benchmark runner, scorer, leaderboard, skill tree
└── dashboard/                   # Streamlit visualizer
data/
├── tree/                        # Commit and branch storage
├── memory/                      # ChromaDB vector store
├── postmortems/                 # Dead branch postmortems (JSON)
└── ultron.db                    # SQLite (tree, scores, leaderboard, skill tree)
```

---

## The Evolution Loop

Each iteration of the main loop:

1. **Wake** — checkout `main` HEAD
2. **Evaluate** — run the current body against the Arena
3. **Reflect** — analyze scores and review failure memory from past dead branches
4. **Decide** — should I evolve? If yes, create a branch and apply mutations
5. **Compete** — run Arena on the new branch body
6. **Judge** — compare branch scores against `main`
7. **Act** — merge if better, keep alive if mixed, kill and write postmortem if worse
8. **Repeat**

---

## Built-in Tools

| Tool | Description |
|---|---|
| `browse_url` | Fetch a URL and extract readable text (async httpx + BeautifulSoup) |
| `execute_shell` | Run shell commands with blocked-command safety and timeout |
| `read_file` | Read file contents asynchronously |
| `write_file` | Write content to a file (creates parent dirs) |
| `delete_file` | Delete a file from the filesystem |
| `list_directory` | List directory contents with sizes and timestamps |

All tools are **async**, dynamically imported at runtime, and automatically converted to OpenAI function-calling format for LLM integration.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11+ (fully async) |
| Package management | uv |
| LLM abstraction | LiteLLM (OpenRouter, OpenAI, Anthropic, Ollama, Groq) |
| Data models | Pydantic v2 |
| Structured storage | SQLite via SQLModel |
| Vector memory | ChromaDB |
| HTTP client | httpx (async) |
| File I/O | aiofiles (async) |
| Terminal output | Rich + Loguru |
| Testing | pytest + pytest-asyncio |
| Linting | Ruff |
| Dashboard *(planned)* | Streamlit + Plotly |

---

## Roadmap

- [x] **Phase 1 — Foundation**: Project setup, Blueprint schema, Body Factory, LiteLLM integration, built-in tools, async agent runtime
- [ ] **Phase 2 — Arena**: Benchmark suite (5 tiers), scorer, runner, leaderboard, skill tree
- [ ] **Phase 3 — Consciousness**: ChromaDB memory, failure memory, identity and goals
- [ ] **Phase 4 — Evolution Tree**: Commit store, branch manager, merge, kill, cherry-pick, diff, bisect, tag
- [ ] **Phase 5 — Evolution Engine**: Reflection, mutation generators, inner voice, tournament mode
- [ ] **Phase 6 — The Loop**: Wire everything into `main.py` — the autonomous branching self-improvement cycle
- [ ] **Phase 7 — Dashboard**: Interactive evolution tree graph, score trends, skill tree, failure memory browser, live journal

---

## Status

**Phase 1 complete.** The foundation is built — Ultron can load a blueprint, create an agent body, and execute tasks using LLM + tools via an async ReAct loop. 40 tests passing.

---

## Philosophy

> Dead branches are not failures. They are lessons.

Every agent architecture eventually faces the question: what do you do when an improvement makes things worse? Most systems silently overwrite the previous state. Ultron keeps everything — the failed branches, the postmortems, the reasoning behind each decision — and uses that history to inform every future evolution attempt. The lineage *is* the intelligence.

---

## License

MIT