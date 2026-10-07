"""Sandboxed command execution (B8).

Commands that run code (tests, builds, installs, scripts) execute inside an OS-native sandbox:
macOS Seatbelt, Linux bubblewrap (or Landlock where bubblewrap isn't installed), Windows
AppContainer — or Docker when configured. Inside it a command can write only the workspace
(minus ``workspace.deny_write``) and its own temp/cache folder, can't read the workspace's
secrets or credentials in the home folder, and has no network unless the run needs it.

``manager`` picks the backend; nothing here imports the package eagerly, so the in-sandbox
helper (``_helper.py``, standard library only) stays independent of Kartrix.
"""
