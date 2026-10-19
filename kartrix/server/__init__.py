"""The Kartrix server (``kartrix serve``): accounts, workspaces and runs behind an HTTP API with a
Server-Sent Events feed. The CLI, the web app and the VS Code extension are its clients; the agent
itself runs in one worker process per workspace (:mod:`kartrix.server.worker`)."""
