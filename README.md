# Kartrix

An autonomous, security-first AI coding agent for your terminal. Kartrix understands your codebase
through RAG, plans multi-step changes for your approval, executes them with least-privilege tools in a
sandbox, and verifies the results — so you can build easy to medium applications with your own LLM keys.

> Work in progress — see [docs/ROADMAP.md](docs/ROADMAP.md) and [docs/PROGRESS.md](docs/PROGRESS.md).

## Quickstart

```bash
uv sync
cp .env.example .env   # add your keys and set the Postgres/Redis passwords
docker compose up -d   # Postgres 17 + pgvector, Redis 8 (localhost only)
uv run alembic upgrade head
uv run kartrix         # run from inside the repository you want to work on
```

## Features

- **Codebase understanding (RAG):** tree-sitter chunking, pgvector + Postgres full-text search fused with
  reciprocal rank fusion, incremental re-indexing, secrets redacted before anything is embedded.
- **Plan → approve → execute:** multi-step plans as a dependency graph, each task run by an agent with
  least-privilege tools and checked by an LLM judge; crash recovery and `/undo` checkpoints.
- **Security-first:** OS-native sandbox (AppContainer, Seatbelt, bubblewrap/Landlock), command policy with
  human approvals, workspace jail, prompt-injection guard, append-only audit log, budgets and a kill switch.
- **Headless mode:** `kartrix run --spec task.yaml` for automation, with a JSON report and exit codes.
- **Eval suite:** `kartrix eval` measures the agent and retrieval quality (see below).

## Evals

`kartrix eval` (built with DeepEval as the LLM judge) measures Kartrix itself — see [evals/README.md](evals/README.md):

- **RAG:** 100 hand-labelled questions over 3 real repositories (this one, the FastAPI full-stack template, an
  Express + Prisma API); hit rate, recall@k, precision@k, MRR and nDCG for dense, lexical, hybrid and
  search-first retrieval; answers judged for faithfulness, relevancy, context precision and recall.
- **Agent:** 26 coding tasks (bug fixes, features, tests, refactors, explain/locate, prompt-injection and budget
  tasks) in Python and TypeScript, each run headless in the sandbox and checked by hidden tests and mutation
  testing; pass@1, pass^k, tool-call accuracy, tokens, cost, latency, safety.
- **Judge calibration** against 20 hand-labelled items (agreement, Cohen's κ).

First measured results (quick subset, 15 questions, `nemotron-3-embed-1b` embeddings):

| Retrieval mode | Hit@5 | Recall@5 | MRR |
|---|---|---|---|
| Dense (pgvector) | 80% | 80% | 0.62 |
| Search-first (no index) | 67% | 60% | 0.48 |
| Hybrid (RRF) | 53% | 53% | 0.37 |
| Lexical (full-text) | 27% | 23% | 0.15 |

```bash
uv run kartrix eval rag --quick --no-judge   # retrieval only
uv run kartrix eval all --quick              # retrieval + agent tasks, HTML report
```

## Development

```bash
git config core.hooksPath .githooks                       # once per clone: lint + types before each commit
git config blame.ignoreRevsFile .git-blame-ignore-revs    # optional: blame skips formatting commits
uv run ruff check . && uv run ruff format --check .       # lint + formatting
uv run mypy                                               # type-check
uv run pytest                                             # tests (DB/Redis tests need `docker compose up -d`)
```

Tests use a separate `<db>_test` database that is created and migrated automatically; tests that need
Postgres or Redis are skipped when those are down (CI runs them all). No test calls an LLM API.
CI (GitHub Actions): lint → type-check → migrations round-trip → tests, against the same compose services.
