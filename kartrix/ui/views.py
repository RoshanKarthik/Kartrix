"""REPL views of stored state: /task_status and /show_index."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from kartrix.config import settings
from kartrix.context.indexers.pg_index import indexed_files
from kartrix.db.models import EMBEDDING_DIMS
from kartrix.tasks.task_store import TaskStore

_STATUS_STYLE = {
    "completed": "green",
    "failed": "red",
    "in_progress": "yellow",
    "pending": "dim",
    "blocked": "red",
    "skipped": "dim",
}


async def show_task_status(console: Console) -> None:
    """A table of all tasks of this repo's most recent project."""
    store = TaskStore()
    project_id = await store.get_latest_project(str(Path.cwd().resolve()))
    if not project_id:
        console.print("[yellow]No active project found. Run /plan <goal> first.[/yellow]")
        return

    tasks = await store.get_all_tasks(project_id)
    progress = await store.get_progress(project_id)
    console.print(f"\n[bold]Project:[/bold] {project_id}")
    console.print(f"[dim]Progress: {progress.get('completed', 0)}/{sum(progress.values())} completed[/dim]\n")

    table = Table(show_header=True, header_style="bold")
    for name, width in (("#", 4), ("ID", 12), ("Type", 10), ("Title", 35), ("Status", 12), ("Retries", 8),
                        ("Error", 40)):  # fmt: skip
        table.add_column(name, width=width)
    for i, task in enumerate(tasks, 1):
        style = _STATUS_STYLE.get(task["status"], "white")
        table.add_row(
            str(i),
            escape(task["id"]),
            task["task_type"],
            escape(task["title"]),
            f"[{style}]{task['status']}[/{style}]",
            f"{task['retry_count']}/{task['max_retries']}",
            escape((task.get("error") or "")[:60]),
        )
    console.print(table)


async def show_index(console: Console, repo_root: str | Path, limit: int = 30) -> None:
    """A summary of what is indexed for ``repo_root``."""
    files = await indexed_files(repo_root)
    total = sum(f.chunk_count for f in files)
    console.print(
        f"\n[bold]Code index — {len(files)} files, {total} chunks[/bold] "
        f"[dim]({settings.embeddings.model}, halfvec({EMBEDDING_DIMS}))[/dim]\n"
    )
    table = Table("File", "Chunks", "Indexed at")
    for f in files[:limit]:
        table.add_row(escape(f.path), str(f.chunk_count), f.indexed_at.strftime("%Y-%m-%d %H:%M"))
    console.print(table)
    if len(files) > limit:
        console.print(f"[dim]… and {len(files) - limit} more files[/dim]")
