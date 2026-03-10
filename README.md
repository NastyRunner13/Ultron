# Ultron

> A self-evolving AI agent that version-controls its own mind.

Ultron is an experimental AI agent that improves itself over time using a Git-like evolution tree. Every proven state is a *commit*. Improvements happen on *branches*. Successful experiments get *merged* into `main`. Failed branches are *killed* — but their postmortems become memory, so the agent never repeats the same mistake.

---

## The Core Idea

Most AI agents are static. You configure them once and they stay that way. Linage treats the agent's configuration — its model, system prompt, tools, and self-written code — as a living artifact that can be branched, mutated, evaluated, and evolved. The result is an agent with a full evolutionary lineage: a record of every version of itself, why each change was made, and what was learned from every failure.

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

**Body** — A fully serializable agent configuration: LLM model, system prompt, tools, and self-written code modules. The thing that evolves.

**Consciousness** — The persistent identity that survives body swaps. Holds long-term memory, goals, and the reasoning engine that decides *how* to evolve.

**Evolution Tree** — A Git-like DAG tracking the agent's entire history. Supports commit, branch, merge, kill, cherry-pick, diff, bisect, and tag.

**Arena** — A tiered benchmark suite that scores each body across reasoning, coding, creativity, tool use, and meta-tasks. The fitness function that determines whether a branch lives or dies.

**Failure Memory** — Structured postmortems from dead branches. Reviewed before every evolution attempt so Ultron never tries the same failed approach twice.

**Inner Voice** — Three internal perspectives that debate before every evolution decision: Explorer (push for radical change), Conservative (argue for stability), Strategist (plan which capability to unlock next).

---

## Architecture

```
ultron/
  config/
    genesis_blueprint.yaml     # Gen 0 — the starting body
    arena_benchmarks.yaml      # Benchmark definitions (5 tiers)
    settings.yaml              # Budget, thresholds, global config
  ultron/
    main.py                    # The evolution loop
    consciousness/             # Memory, identity, goals, dream state
    body/                      # Blueprint schema (Pydantic) + Body Factory
    tree/                      # Evolution tree: commits, branches, merge, diff, bisect
    evolution/                 # Reflection, mutation generators, tournament, journal
    arena/                     # Benchmark runner, scorer, leaderboard, skill tree
    tools/                     # Built-in tools + directory for self-written tools
    dashboard/                 # Streamlit visualizer
  data/
    tree/                      # Commit and branch storage
    memory/                    # ChromaDB vector store
    postmortems/               # Dead branch postmortems (JSON)
    ultron.db                  # SQLite (tree, scores, leaderboard, skill tree)
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

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11+ |
| Package management | uv |
| LLM abstraction | LiteLLM (OpenAI, Anthropic, Ollama, Groq) |
| Data models | Pydantic v2 |
| Structured storage | SQLite via SQLModel |
| Vector memory | ChromaDB |
| Loop orchestration | LangGraph |
| Code sandbox | E2B |
| Terminal output | Rich + Loguru |
| Dashboard | Streamlit + Plotly |

---

## Roadmap

- [ ] **Phase 1 — Foundation**: Project setup, Blueprint schema, Body Factory, LiteLLM integration
- [ ] **Phase 2 — Arena**: Benchmark suite, scorer, runner, leaderboard, skill tree
- [ ] **Phase 3 — Consciousness**: ChromaDB memory, failure memory, identity and goals
- [ ] **Phase 4 — Evolution Tree**: Commit store, branch manager, merge, kill, cherry-pick, diff, bisect, tag
- [ ] **Phase 5 — Evolution Engine**: Reflection, mutation generators, inner voice, tournament mode
- [ ] **Phase 6 — The Loop**: Wire everything into `main.py` — the autonomous branching self-improvement cycle
- [ ] **Phase 7 — Dashboard**: Interactive evolution tree graph, score trends, skill tree, failure memory browser, live journal

---

## Status

Early design stage. The architecture is defined. Implementation starts with Phase 1.

---

## Philosophy

> Dead branches are not failures. They are lessons.

Every agent architecture eventually faces the question: what do you do when an improvement makes things worse? Most systems silently overwrite the previous state. Linage keeps everything — the failed branches, the postmortems, the reasoning behind each decision — and uses that history to inform every future evolution attempt. The lineage *is* the intelligence.

---

## License

MIT