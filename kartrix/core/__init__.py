"""Kartrix's core, independent of any terminal (C1).

- ``events`` — the event stream every front end consumes;
- ``interaction`` — the callbacks the core uses to ask for decisions (approvals, plan review);
- ``session`` — :class:`~kartrix.core.session.CoreSession`: start up in a workspace and run chat
  requests and plans, budgeted, stoppable and checkpointed.

Front ends: the REPL (``kartrix.main`` + ``kartrix.ui``) and headless mode (``kartrix.headless``).
"""
