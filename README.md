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
