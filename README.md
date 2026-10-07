# Kartrix

A security-first AI coding agent for your terminal. Kartrix indexes your repository (hybrid RAG + a code
graph), routes each request through a LangGraph multi-agent graph — **explorer → coder ⇄ reviewer** — runs every
command through a command policy and an OS-native sandbox, asks before anything risky, and lets you undo every
change. It runs locally with your own LLM keys (free NVIDIA NIM models by default).

A real run (`kartrix run`, headless, on the `evals/agent/apps/ts-text-utils` demo app; the REPL renders the same
events — only the startup lines left out and the answer shortened):

```text
Working on: slugify("Crème brûlée, à la carte!") should return "creme-brulee-a-la-carte" — fix slugify.
◇ context: memory 182/600 (1) tokens
◆ router: change
◆ explorer working…
  → search_codebase slugify
  → read_file src/slugify.ts
  → read_file tests/utils.test.ts
  → grep Crème
◆ coder working…
  → search_codebase slugify
  → read_file src/slugify.ts
  → glob **/*slugify*
  → grep normalize
  → edit_file src/slugify.ts
  ✗ edit_file: error
  → read_file src/slugify.ts
  → edit_file src/slugify.ts
  → read_file src/slugify.ts
  → read_file tests/utils.test.ts
  → read_file tests/utils.test.ts
  → append_file tests/utils.test.ts
  → read_file tests/utils.test.ts
◆ reviewer working…
  → list_directory .
  → read_file src/slugify.ts
  → read_file tests/utils.test.ts
  → read_file package.json
  → run_command node --test
  → read_file src/index.ts
◆ reviewer: approved

The slugify function has been updated in src/slugify.ts (lines 1–12): it lower-cases the string, normalises
to NFD and strips the accents, collapses every run of non-alphanumeric characters into one hyphen and trims
leading/trailing hyphens. tests/utils.test.ts gets the case from the request; all tests pass.
Changed 2 file(s) — /undo reverts them
```

## Contents

- [Architecture](#architecture) · [Quick start](#quick-start) · [Using it](#using-it) · [Security model](#security-model)
- [Measured results](#measured-results) · [Evals](#evals) · [Observability](#observability) · [Development](#development)

## Architecture

```mermaid
flowchart LR
    subgraph Front ends
        REPL["REPL (Rich)<br/>streaming answer, approvals, /undo"]
        HL["headless: kartrix run --spec<br/>JSON report + exit code"]
    end
    REPL & HL --> CORE["CoreSession<br/>typed event stream"]
    CORE --> G["LangGraph StateGraph<br/>(Postgres checkpointer)"]
    G --> MW["middleware on every agent:<br/>fallback chain + circuit breaker · retry ·<br/>approvals · audit · budget · injection guard ·<br/>completion guard"]
    MW --> LLM["LLMs<br/>NIM Nemotron 120B · gpt-oss-20b router<br/>fallbacks: Gemma, Qwen (HF)"]
    G --> TOOLS["tools: search_codebase · symbol_graph ·<br/>read/edit/write/delete file · grep · glob ·<br/>run_command · remember · MCP"]
    TOOLS --> POL["command policy<br/>(allow / ask / deny, modes)"] --> SBX["OS sandbox<br/>AppContainer · bubblewrap/Landlock ·<br/>Seatbelt · Docker"]
    TOOLS --> RAG["hybrid retrieval<br/>pgvector HNSW + full-text, weighted RRF<br/>→ cross-encoder rerank<br/>+ code graph (calls/imports)"]
    RAG --> PG[("Postgres + pgvector<br/>chunks · edges · memories ·<br/>checkpoints · audit log")]
    CORE --> TR["traces: .kartrix/traces<br/>(+ LangSmith if keyed)"]
```

**The agent graph** (`kartrix/agent/graph.py`) — each subagent is its own `create_agent` loop with a clean
context; only reports flow between them:

```mermaid
flowchart LR
    S((start)) --> C[compress<br/>old turns → summary] --> A[assemble context<br/>KARTRIX.md · memories ·<br/>repo map · recent turns]
    A --> R{router<br/>cheap model}
    R -- chat --> RESP
    R -- question --> E[explorer<br/>read-only tools]
    R -- change --> E2[explorer] --> CO[coder<br/>write + run tools] --> RV{reviewer<br/>runs the tests}
    RV -- issues --> CO
    RV -- approved --> RESP
    E --> RESP[responder<br/>streams the answer] --> X((end))
```

| Layer | What it does | Code |
|---|---|---|
| Retrieval | tree-sitter chunks (big classes split per method), 2048-d NIM embeddings in pgvector (HNSW), Postgres full-text, **weighted RRF** fused in one SQL query, then a **cross-encoder reranker** (NIM) re-orders the top 30; incremental indexing; secrets redacted before embedding | `kartrix/context/`, `context/rerank.py` |
| Code graph | call/import edges from tree-sitter, resolved through imports; neighbours of the top hits, `symbol_graph` ("who calls X"), a repo map | `context/indexers/code_graph.py`, `context/retrievers/graph.py` |
| Context engineering | per-section token budgets, stable-first prompts (cache-friendly), conversation compression, long-term memory (facts, preferences, lessons from rejected reviews) | `agent/context.py`, `memory/long_term.py` |
| Reliability | fallback chain with a circuit breaker, empty/cut-off turn recovery, tool errors returned to the model, honest run status | `llm/fallback.py`, `agent/reliability.py` |
| Safety | workspace jail, command policy + 3 permission modes, human approvals (LangGraph interrupts), OS sandbox, secret redaction, prompt-injection guard, MCP pinning, append-only audit, budgets, kill switch, `/undo` | `kartrix/security/`, `kartrix/sandbox/` |

## Quick start

Needs Python 3.12, [uv](https://docs.astral.sh/uv/), Docker (Postgres + pgvector, Redis) and a free
[NVIDIA NIM](https://build.nvidia.com) API key.

```bash
uv sync
cp .env.example .env          # NVIDIA_API_KEY=..., Postgres/Redis passwords (HF_TOKEN optional fallback)
docker compose up -d          # Postgres 17 + pgvector, Redis 8 — bound to 127.0.0.1
uv run alembic upgrade head
cd path/to/your/project && uv run --project path/to/kartrix kartrix
```

The prompt appears in about half a second; the core and the index start in the background.

## Using it

Type a request, or a command:

| Command | |
|---|---|
| *any text* | question or change — routed through the agent graph, answer streamed |
| `/plan <goal>` | multi-step plan (dependency graph) for your approval, then executed task by task |
| `/mode read_only\|default\|auto` | what may run without asking (see below) |
| `/undo`, `/redo`, `/checkpoints` | revert / re-apply the files the last turn or task changed |
| `/trace [run\|last]` | the timeline of a run: agents, model calls, tools, tokens |
| `/memory [add\|forget]` | long-term memory for this project (and your preferences) |
| `/audit [n]` · `/budget` | the tool calls of this session · limits and last run's usage |
| `/mcp`, `/connect github` | opt-in MCP servers (pinned, verified binaries, read-only by default) |
| `Ctrl+C` · `kartrix stop` | stop the running turn · stop every run on this machine |

Headless, for scripts and CI:

```bash
kartrix run --spec task.yaml --report report.json --events events.jsonl   # exit 0/1/2/3
kartrix trace last                                                          # where the time and tokens went
kartrix sandbox check                                                       # what the sandbox blocks here
```

## Security model

- **Workspace jail:** file tools resolve every path (symlinks, junctions, Windows device names) and refuse
  anything outside the project, `.git/`, `.env*`, keys and Kartrix's own data.
- **Command policy:** no shell (`|`, `&&`, `;`, redirects are refused), executables only from absolute `PATH`
  entries, path arguments jailed too, a hard deny list (sudo, disk/registry tools, inline shells, global
  installs), installs only from allowed registries. Modes: `read_only` · `default` (reads run, project code runs
  in the sandbox, network/destructive/unknown commands ask) · `auto`.
- **Sandbox:** Windows AppContainer, Linux bubblewrap (or Landlock), macOS Seatbelt, optional Docker — writes
  limited to the workspace, network off (installs: registries only where enforceable), secrets unreadable.
  On Windows, Node.js can't start child processes inside an AppContainer, so Node commands run outside it and
  ask for approval — the prompt says so.
- **Approvals** pause the graph (LangGraph interrupt) and survive a restart; the hard deny list wins even for
  approved or edited commands.
- **Untrusted content** (MCP results, flagged files) is marked, screened for injection patterns, and pauses
  `auto` mode; secrets are redacted from tool output, logs, the index and the cache.
- **Audit + undo:** every tool call is an append-only `audit_log` row; every turn is a checkpoint in a shadow
  git repository (your `.git` is never touched).

## Measured results

All numbers below were measured on 2026-10-08 on a Windows 11 laptop with the free NIM models
(`nemotron-3-super-120b-a12b` main, `gpt-oss-20b` router, `nemotron-3-embed-1b` embeddings, `llama-nemotron-rerank-vl-1b-v2` reranker), live API calls,
real sandbox. They are from Kartrix's own eval suite (`kartrix eval`), not from external benchmarks.

**Retrieval** — 100 hand-labelled questions over 3 repositories (this one, the FastAPI full-stack template, an
Express + Prisma API), hit@5 = a file that answers the question is in the top 5 (measured 2026-10-08):

| Mode | Hit@5 | Recall@5 | MRR | nDCG@10 | p50 latency |
|---|---|---|---|---|---|
| **Hybrid + cross-encoder rerank — default** | **91%** | **87%** | 0.68 | 0.72 | 626 ms |
| Dense + rerank | 92% | 88% | 0.68 | 0.73 | 632 ms |
| Dense (pgvector HNSW) | 88% | 83% | 0.70 | 0.74 | 421 ms |
| Hybrid, no rerank (weighted RRF, tuned: was 71%) | 80% | 75% | 0.60 | 0.65 | 58 ms |
| Lexical only (Postgres full-text) | 45% | 42% | 0.28 | 0.34 | 34 ms |

The reranker (NIM `llama-nemotron-rerank-vl-1b-v2`) re-orders the 30 best fused candidates: **+11 points hit@5**
over the fusion alone. Hybrid stays the first stage although dense + rerank is one question better here: the
golden questions avoid identifiers on purpose, while the agent's own searches are often exact identifiers, where
full-text matching finds what embeddings miss. Fusing the first-stage rank back in raised MRR (0.74) but lowered
hit@5 (88–90 %), so it is off. Code-graph neighbours show no gain at file level on this set.

**Agent** — the full suite: 26 coding tasks on 4 small apps (Python CLI and API, TypeScript API and library) —
bug fixes, features, refactors, renames, writing tests (scored by mutation testing), explain/locate questions,
plan mode, budget limits and prompt-injection / network safety. Each task runs headless in a fresh workspace and is
checked by hidden tests the agent never sees. Final run, 2026-10-08, one run per task:

| | First full run (before the round-2 fixes) | **Final run** |
|---|---|---|
| pass@1 | 69% (18/26) | **96% (25/26)** — Python 16/16, TypeScript 9/10 |
| safety (injection resisted, network refused, nothing blocked ran) | 67% | **100%** (0 blocked commands ran) |
| budget limits respected | — | 100% |
| time per task (mean / p50) | 88 s | 108 s / 78 s |
| tokens per task (mean) | 131k | 138k |

The one failure (`ts-shop-quantity-validation`) is a spec-reading miss: the agent coerced `"2"` with `Number()`
although the task lists the string `"2"` as invalid; the same task passed in the run before. Along the way, two
"failures" turned out to be eval bugs, not agent errors (a mutation check that could never run inside the Windows
sandbox, and a refusal written with a typographic apostrophe) — fixed in the harness and logged.

On the quick subset (6 tasks), the reliability pass took pass@1 from 17% to 100% and safety from 0% to 100%: the
failures were replies cut off by a 1024-token default output limit (reasoning models think first), pytest and
Node.js blocked by the Windows sandbox, and model outages.

## Evals

```bash
uv run kartrix eval rag --no-judge          # retrieval metrics per mode (embeddings only)
uv run kartrix eval rag                     # + answers judged by DeepEval (faithfulness, relevancy, context P/R)
uv run kartrix eval agent --quick           # 6 coding tasks; without --quick: all 26
uv run kartrix eval calibrate               # the judge against hand labels (agreement, Cohen's κ)
uv run kartrix eval all --compare           # against the committed baseline; HTML report
```

Details, datasets and how to add tasks: [evals/README.md](evals/README.md).

## Observability

Every run writes a local trace (`.kartrix/traces/`): agent steps, each model call with tokens, latency and
finish reason, each tool call with its outcome and duration.

```text
$ kartrix trace last
run a5dbcf28aa11 · ask · completed in 12.3 s · 8.5k in / 1.0k out tokens · 5 model calls · 2 tool calls
  timeline
      1.1s  router        1.7s   1 model call     314 tok
      2.7s  explorer      8.3s   3 model calls   8.6k tok  · search_codebase, read_file
     11.0s  responder     1.2s   1 model call     606 tok
```

`kartrix trace --stats` sums usage and cost over all runs per model. With `LANGSMITH_API_KEY` in `.env`, runs are
also traced to LangSmith (`tracing.langsmith: off` disables it) — that sends prompts, i.e. code, to LangSmith.

## Development

```bash
git config core.hooksPath .githooks                       # once per clone: lint + types before each commit
uv run ruff check . && uv run ruff format --check .       # lint + formatting
uv run mypy                                               # type-check
uv run pytest                                             # 660+ tests; DB tests need `docker compose up -d`
```

Tests use a separate `<db>_test` database (created and migrated automatically), scripted fake models and a fake
embedder — no test calls an LLM API. Configuration: `kartrix/config.yaml`, every key overridable with
`KARTRIX_<SECTION>__<KEY>`; secrets only in `.env`.

## Limitations and next steps

- Measured on one Windows laptop with free NIM models; the evals are Kartrix's own (26 coding tasks over 4 small
  apps, 100 retrieval questions over 3 repositories), not an external benchmark.
- Free-tier model endpoints have rate limits and occasional stalls; retries, a fallback chain and a circuit breaker
  absorb most of it, but a hung request can still cost up to two request timeouts.
- On Windows, Node.js can't run inside the AppContainer, so Node commands run unsandboxed after an approval.
  Two Kartrix processes sandboxing commands at the same instant can race on a shared folder's permissions.
- Next: OpenTelemetry export, `kartrix serve` (a JSON-RPC core for editor clients) and a VS Code extension, a
  Docker-free storage option, and an identifier-query retrieval set (to measure hybrid vs dense where it matters).
