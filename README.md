# Kartrix

An autonomous, security-first AI coding agent. Kartrix understands your codebase through
RAG, plans multi-step changes for your approval, executes them with least-privilege tools,
and verifies the results.

> Work in progress — see [docs/ROADMAP.md](docs/ROADMAP.md) and [docs/PROGRESS.md](docs/PROGRESS.md).

## Quickstart

```bash
uv sync
cp .env.example .env   # add your keys and set the Postgres/Redis passwords
docker compose up -d   # Postgres 17 + pgvector, Redis 8 (localhost only)
uv run alembic upgrade head
uv run kartrix         # run from inside the repository you want to work on
```
