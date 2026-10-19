"""Bring the database schema up to date from inside Kartrix (installed copies have no ``alembic.ini``).

The migrations ship in the package (``kartrix/migrations``). Alembic's environment runs its own event loop, so
the upgrade runs in a separate thread: callers may already be inside one.
"""

from __future__ import annotations

import threading
from pathlib import Path

MIGRATIONS = Path(__file__).resolve().parent.parent / "migrations"


def upgrade_to_head() -> None:
    """``alembic upgrade head`` against ``DATABASE_URL``; a no-op when the schema is current."""
    from alembic import command
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    errors: list[BaseException] = []

    def run() -> None:
        try:
            command.upgrade(cfg, "head")
        except BaseException as e:  # re-raised in the caller's thread
            errors.append(e)

    worker = threading.Thread(target=run, name="kartrix-migrate")
    worker.start()
    worker.join()
    if errors:
        raise errors[0]
