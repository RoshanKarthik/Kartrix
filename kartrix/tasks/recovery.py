from __future__ import annotations

from kartrix.core.events import notice
from kartrix.observability.logger import get_logger
from kartrix.tasks.task_store import TaskStore

logger = get_logger(__name__)


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
                notice(
                    f"Recovered {task['key']} ({task['title']}) → pending "
                    f"(retry {task['retry_count']}/{task['max_retries']})",
                    "warning",
                )
            else:
                notice(f"Max retries used up: {task['key']} → failed", "error")
        return len(crashed)
