from __future__ import annotations

from rich.console import Console

from kartrix.observability.logger import get_logger
from kartrix.tasks.task_store import TaskStore

logger = get_logger(__name__)
console = Console()


class RecoveryManager:
    """
    Detects and resets tasks left IN_PROGRESS when the process crashed.

    In single-process serial execution, any IN_PROGRESS task at startup is
    unconditionally orphaned — the process that was running it is gone.
    No heartbeat or timing check is needed.
    """

    def __init__(self, store: TaskStore) -> None:
        self.store = store

    async def recover(self, project_id: str) -> int:
        """
        Reset every task still marked IN_PROGRESS for this project.
          - retries left  → reset to PENDING
          - no retries    → mark FAILED
        Returns number of tasks processed.
        """
        crashed = await self.store.recover_crashed(project_id)
        for task in crashed:
            if task["status"] == "pending":
                console.print(
                    f"[yellow]🔄 Recovered:[/yellow] {task['key']} ({task['title']}) "
                    f"→ PENDING (retry {task['retry_count']}/{task['max_retries']})"
                )
            else:
                console.print(f"[red]❌ Max retries exhausted:[/red] {task['key']} → FAILED")

        if crashed:
            console.print(f"[dim]Recovery complete: {len(crashed)} task(s) processed.[/dim]\n")
        return len(crashed)
