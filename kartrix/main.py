import asyncio
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt

from kartrix.config import settings
from kartrix.context.indexers.pg_index import index_repo, show_index
from kartrix.llm.factory import get_llm, get_embedder
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from kartrix.agent.factory import build_agent
from kartrix.memory.short_term import get_checkpointer_db_path
from kartrix.agent.orchestrator import handle_query
from kartrix.memory.session import InvalidSessionIdError, get_current_session, new_session, switch_session
from kartrix.cache.semantic_cache import build_semantic_cache, get_repo_domain
from kartrix.db.engine import dispose_engine
from kartrix.observability.logger import get_logger

from kartrix.tasks.orchestrator import handle_plan_command
from kartrix.tasks.status import show_task_status
from kartrix.context.indexers.watcher import start_watcher, stop_watcher

console = Console()
logger = get_logger(__name__)


async def update_index() -> None:
    """Incrementally sync the Postgres code index with the current directory."""
    repo_path = str(Path.cwd())
    logger.info(f"Checking index for: {repo_path}")
    console.print(f"[dim]Checking index for {repo_path}...[/dim]")
    stats = await index_repo(repo_path)
    console.print(f"[dim]Index: {stats}[/dim]")


async def initialize(checkpointer):
    """Bootstrap LLM, embedder, index, watcher, MCP tools, cache, and session before the REPL starts."""
    llm      = get_llm()
    embedder = get_embedder()
    console.print(f"[dim]LLM: {settings.llm.provider} / {settings.llm.model}[/dim]")
    console.print(f"[dim]Embedder: {settings.embeddings.provider} / {settings.embeddings.model}[/dim]")

    repo_path = str(Path.cwd())
    await update_index()

    semantic_cache = await build_semantic_cache()
    cache_domain = get_repo_domain(repo_path) if semantic_cache else None
    if semantic_cache is not None:
        console.print(f"[dim]Semantic cache: enabled (threshold={semantic_cache.threshold})[/dim]")
    else:
        console.print("[dim]Semantic cache: disabled[/dim]")

    loop = asyncio.get_running_loop()

    async def _invalidate_cache_on_change() -> None:
        if semantic_cache is not None:
            await semantic_cache.invalidate_domain(cache_domain)

    observer   = start_watcher(repo_path, loop, on_change=_invalidate_cache_on_change)
    agent      = await build_agent(checkpointer)
    session_id = get_current_session()
    console.print(f"[dim]Session: {session_id}[/dim]")
    console.print(f"[green]✓ Ready[/green]\n")
    return llm, embedder, agent, session_id, observer, semantic_cache, cache_domain


async def _run_async():
    logger.info("Starting Kartrix")
    console.print("\n[bold blue]Kartrix[/bold blue] — RAG-powered code assistant")

    async with AsyncSqliteSaver.from_conn_string(get_checkpointer_db_path()) as checkpointer:
        llm, embedder, agent, session_id, observer, semantic_cache, cache_domain = await initialize(checkpointer)
        console.print("Type [bold]'/exit'[/bold] to quit\n")

        try:
            while True:
                user_input = Prompt.ask("[bold green]>[/bold green]")

                if not user_input.strip():
                    continue
                if user_input.lower() in ("/exit", "/quit"):
                    logger.info("Shutting down")
                    console.print("[dim]Goodbye![/dim]")
                    break
                elif user_input.startswith("/ask "):
                    question = user_input.removeprefix("/ask ").strip()
                    logger.info(f"Ask command received: {question}")
                    console.print(f"[dim]Searching for: {question}...[/dim]")
                    response = await handle_query(
                        agent, question, session_id,
                        semantic_cache=semantic_cache, cache_domain=cache_domain,
                    )
                    console.print(response)
                elif user_input == "/reindex":
                    console.print("[dim]Re-indexing current directory...[/dim]")
                    await update_index()
                    if semantic_cache is not None:
                        await semantic_cache.invalidate_domain(cache_domain)
                        console.print("[dim]Semantic cache invalidated for this repo.[/dim]")
                    console.print("[green]✓ Re-index complete.[/green]")
                elif user_input == "/new_session":
                    session_id = new_session()
                    console.print(f"[green]New session started: {session_id}[/green]")
                elif user_input.startswith("/switch "):
                    target = user_input.removeprefix("/switch ").strip()
                    try:
                        session_id = switch_session(target)
                    except InvalidSessionIdError:
                        console.print("[red]Invalid session id — expected a UUID like the one shown by /session.[/red]")
                    else:
                        console.print(f"[green]Switched to session: {session_id}[/green]")
                elif user_input == "/session":
                    console.print(f"[dim]Current session: {session_id}[/dim]")
                elif user_input == "/show_index":
                    logger.info("Showing index")
                    await show_index(Path.cwd())
                elif user_input.startswith("/plan "):
                    goal = user_input.removeprefix("/plan ").strip()
                    logger.info(f"Plan command received: {goal}")
                    await handle_plan_command(goal)
                elif user_input == "/task_status":
                    show_task_status()
                else:
                    logger.warning(f"Unknown command received: {user_input}")
                    console.print("[yellow]Unknown command. Try:[/yellow]")
                    console.print("  [bold]/ask <question>[/bold]          — ask a question about the codebase")
                    console.print("  [bold]/show_index[/bold]              — show all chunks in the index")
                    console.print("  [bold]/reindex[/bold]                 — manually re-index current directory")
                    console.print("  [bold]/new_session[/bold]             — start a fresh conversation")
                    console.print("  [bold]/switch <session_id>[/bold]     — resume a past session")
                    console.print("  [bold]/session[/bold]                 — show current session id")
                    console.print("  [bold]/plan <goal>[/bold]             — generate and execute a plan")
                    console.print("  [bold]/task_status[/bold]             — show task progress for active project")
        finally:
            stop_watcher(observer)
            await dispose_engine()

def run():
    asyncio.run(_run_async())

if __name__ == "__main__":
    run()