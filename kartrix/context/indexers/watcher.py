from __future__ import annotations

import asyncio
import platform
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path

from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from kartrix.context.discovery import RepoFilter
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# How the watcher works
# ------------------------------------------------------------------
# A watchdog Observer runs in a background daemon thread watching the
# project directory recursively. When a file changes, the OS fires an
# event → _CodebaseEventHandler receives it → debounces it (editors
# emit multiple events per save) → schedules index_file() / remove_file()
# from pg_index.py on the app's asyncio loop (the DB engine lives there).
#
# Only paths that pass the repo's .gitignore rules are handled. Editing a
# .gitignore reloads the rules and runs a full incremental index_repo(),
# so newly ignored files drop out and newly included ones get indexed.
#
# This keeps the Postgres index up-to-date in real time so /ask queries
# reflect changes made by /plan tasks or manual edits without a restart.
# ------------------------------------------------------------------

# Editors often fire several events for a single save (write + chmod +
# rename swap). Wait this long after the last event before acting —
# any new event for the same file resets the timer.
_DEBOUNCE_SECONDS = 1.5


def _get_observer() -> Observer:
    """
    Return the best available watchdog observer for the current OS.

    macOS  → FSEvents   (native, event-driven, instant)
    Linux  → inotify    (native, event-driven, instant)
    Windows → PollingObserver — ReadDirectoryChangesW has edge cases
              under heavy load and on network drives, so we poll every
              2 seconds instead. Slightly delayed but reliable.
    """
    if platform.system() == "Windows":
        return PollingObserver(timeout=2)
    return Observer()


class _CodebaseEventHandler(FileSystemEventHandler):
    """
    Translates raw watchdog filesystem events into debounced indexer calls.

    Only reacts to files the RepoFilter accepts (.gitignore + config excludes +
    known extensions). Deletes only need to pass the rules — the file is gone, so
    size/binary checks can't run, and removing a never-indexed path is a no-op.

    Debounce pattern:
      Each file gets its own threading.Timer. If a new event arrives for the
      same file before the timer fires, the old timer is cancelled and a new
      one starts. The indexer is only called once the file has been stable for
      _DEBOUNCE_SECONDS.
    """

    def __init__(
        self, repo_path: str, loop: asyncio.AbstractEventLoop, on_change: Callable[[], Awaitable[None]] | None = None
    ) -> None:
        self._root = repo_path
        self._loop = loop
        self._on_change = on_change
        self._filter = RepoFilter.load(repo_path)
        # {filepath: threading.Timer} — one pending debounced call per file.
        # Timers are daemon threads so they don't block process exit.
        self._timers: dict[str, threading.Timer] = {}

    def _schedule(self, action: str, filepath: str) -> None:
        """
        Schedule a debounced indexer call for filepath.
        Cancels any already-pending call for the same file first.
        action is "upsert" (create/modify), "delete" or "rescan" (.gitignore changed).
        """
        existing = self._timers.pop(filepath, None)
        if existing:
            existing.cancel()

        timer = threading.Timer(_DEBOUNCE_SECONDS, self._run, args=(action, filepath))
        timer.daemon = True
        timer.start()
        self._timers[filepath] = timer

    def _run(self, action: str, filepath: str) -> None:
        """Runs on the timer thread: hand the work to the asyncio loop and wait for it."""
        self._timers.pop(filepath, None)
        future = asyncio.run_coroutine_threadsafe(self._apply(action, filepath), self._loop)
        try:
            future.result()
        except Exception as e:  # never kill the watcher over one file
            logger.error("Watcher reindex failed", extra={"path": filepath, "action": action, "error": str(e)})

    async def _apply(self, action: str, filepath: str) -> None:
        # Imported here so importing the watcher doesn't pull in the DB/LLM stack.
        from kartrix.context.indexers.pg_index import index_file, index_repo, remove_file

        if action == "rescan":
            self._filter = RepoFilter.load(self._root)
            stats = await index_repo(self._root)
            logger.info("Rules changed, repo re-scanned", extra={"stats": str(stats)})
        elif action == "delete":
            await remove_file(self._root, filepath)
        else:
            await index_file(self._root, filepath, self._filter)
        if self._on_change is not None:
            await self._on_change()

    def _handle(self, path: str, deleted: bool = False) -> None:
        if Path(path).name == ".gitignore":
            self._schedule("rescan", self._root)  # keyed on the root: many edits → one rescan
        elif deleted:
            if self._filter.passes_rules(path):
                self._schedule("delete", path)
        elif self._filter.is_indexable(path):
            self._schedule("upsert", path)

    # ── watchdog event hooks ──────────────────────────────────────────

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle(event.src_path, deleted=True)

    def on_moved(self, event: FileSystemEvent) -> None:
        # A rename/move = delete the old path + upsert the new path.
        if not event.is_directory:
            self._handle(event.src_path, deleted=True)
            self._handle(event.dest_path)


def start_watcher(
    repo_path: str, loop: asyncio.AbstractEventLoop, on_change: Callable[[], Awaitable[None]] | None = None
) -> Observer:
    """
    Start a filesystem observer on repo_path in a background daemon thread.

    The observer watches recursively — all subdirectories are covered.
    Daemon=True means the thread won't prevent the process from exiting
    if the user hits Ctrl+C without going through /exit.

    Index updates run as coroutines on ``loop`` (the app's event loop, where the
    database engine lives). on_change, if given, is awaited after every update -
    used to invalidate the semantic cache when the codebase changes.

    Returns the Observer so the caller can call stop_watcher() on shutdown.
    """
    handler = _CodebaseEventHandler(repo_path, loop, on_change=on_change)
    observer = _get_observer()
    observer.schedule(handler, repo_path, recursive=True)
    observer.daemon = True
    observer.start()
    logger.info(f"Watchdog started on {repo_path} (backend: {type(observer).__name__})")
    return observer


def stop_watcher(observer: Observer) -> None:
    """
    Cleanly stop the observer and wait for its thread to finish.
    Called in the finally block of _run_async() in main.py so it
    always runs regardless of how the REPL exits.
    """
    observer.stop()
    observer.join()
    logger.info("Watchdog stopped")
