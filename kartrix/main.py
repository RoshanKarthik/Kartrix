import asyncio
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.prompt import Prompt
from rich.table import Table

from kartrix.agent.factory import NATIVE_TOOLS, build_agent
from kartrix.agent.orchestrator import handle_query
from kartrix.cache.semantic_cache import build_semantic_cache, get_repo_domain
from kartrix.config import settings
from kartrix.context.indexers.pg_index import index_repo, show_index
from kartrix.context.indexers.watcher import start_watcher, stop_watcher
from kartrix.db.engine import dispose_engine
from kartrix.llm.factory import get_embedder, get_llm
from kartrix.mcp.mcp_client import McpError, McpManager
from kartrix.memory.session import (
    InvalidSessionIdError,
    get_current_session,
    new_session,
    record_session,
    switch_session,
)
from kartrix.memory.short_term import get_checkpointer
from kartrix.observability.logger import get_logger
from kartrix.security import audit, external_tools
from kartrix.security.approval_prompt import ConsoleApprover
from kartrix.security.approvals import resume_pending
from kartrix.security.permissions import MODES, clear_session_allowances, get_mode, set_mode
from kartrix.security.workspace import set_workspace
from kartrix.skills.registry import SkillNotFoundError
from kartrix.skills.skill_tools import get_registry
from kartrix.tasks.orchestrator import handle_plan_command
from kartrix.tasks.status import show_task_status

console = Console()
logger = get_logger(__name__)
approver = ConsoleApprover(console)


async def update_index() -> None:
    """Incrementally sync the Postgres code index with the current directory."""
    repo_path = str(Path.cwd())
    logger.info(f"Checking index for: {repo_path}")
    console.print(f"[dim]Checking index for {repo_path}...[/dim]")
    stats = await index_repo(repo_path)
    console.print(f"[dim]Index: {stats}[/dim]")


async def show_audit(session_id: str, arg: str) -> None:
    """Print the newest audit rows of the current session."""
    limit = int(arg) if arg.isdigit() else 20
    rows = await audit.recent(session_id, limit)
    if not rows:
        console.print("[dim]No audit entries for this session yet.[/dim]")
        return
    table = Table(title=f"Audit log — last {len(rows)} entries")
    for col in ("time", "action", "target", "outcome", "ms"):
        table.add_column(col)
    colors = {"ok": "green", "denied": "red", "error": "red", "needs_approval": "yellow", "declined": "yellow"}
    for row in reversed(rows):
        color = colors.get(row.outcome, "white")
        table.add_row(
            row.ts.astimezone().strftime("%H:%M:%S"),
            row.action,
            (row.target or "")[:60],
            f"[{color}]{row.outcome}[/{color}]",
            str(row.details.get("duration_ms", "")),
        )
    console.print(table)


async def finish_pending_approval(agent, session_id: str) -> None:
    """A session closed while asking for approval is still paused there: ask again and finish it."""
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = await agent.aget_state(config)
        if not state.interrupts:
            return
        console.print("[yellow]This session stopped while waiting for your approval:[/yellow]")
        result = await resume_pending(agent, config, approver)
    except Exception as e:
        logger.error(f"Could not resume the pending approval: {e}")
        console.print(f"[red]Could not resume the pending approval: {e}[/red]")
        return
    if result and result.get("messages"):
        console.print(result["messages"][-1].content)


async def handle_mcp_command(command: str, arg: str, mcp: McpManager, session_id: str) -> bool:
    """/mcp, /connect <server>, /disconnect <server>. True if the agent's tools changed."""
    if command == "/mcp" or not arg:
        try:
            servers = McpManager.available()
        except McpError as e:
            console.print(f"[red]{e}[/red]")
            return False
        table = Table(title="MCP servers")
        for col in ("server", "status", "tools", "description"):
            table.add_column(col)
        for name, server in servers.items():
            connected = name in mcp.connected
            count = sum(1 for t in mcp.tools if (ext := external_tools.get(t.name)) and ext.server == name)
            status = "[green]connected[/green]" if connected else "off"
            table.add_row(name, status, str(count) if connected else "", server.description)
        console.print(table)
        console.print("[dim]/connect <server> to turn one on; it stays on for later sessions until /disconnect.[/dim]")
        return False
    if command == "/connect":
        console.print(f"[dim]Connecting {arg} (on first use the server is downloaded and verified)...[/dim]")
        try:
            tools = await mcp.connect(arg)
        except McpError as e:
            console.print(f"[red]{e}[/red]")
            await audit.record(actor="user", action="mcp.connect", target=arg, outcome="error", session_id=session_id)
            return False
        approve = sum(1 for t in tools if (ext := external_tools.get(t.name)) and ext.needs_approval)
        console.print(f"[green]✓ {arg} connected: {len(tools)} tools ({approve} need your approval to run).[/green]")
        await audit.record(
            actor="user", action="mcp.connect", target=arg, outcome="ok", session_id=session_id,
            details={"tools": [t.name for t in tools]},
        )  # fmt: skip
        return True
    await mcp.disconnect(arg)
    console.print(f"[dim]{arg} disconnected.[/dim]")
    await audit.record(actor="user", action="mcp.disconnect", target=arg, outcome="ok", session_id=session_id)
    return True


async def handle_skills_command(arg: str, session_id: str) -> bool:
    """/skills, /skills trust <name>, /skills untrust <name>. True if the trusted set changed."""
    registry = get_registry()
    action, _, name = arg.partition(" ")
    if action in ("trust", "untrust") and name.strip():
        name = name.strip()
        try:
            if action == "untrust":
                registry.untrust(name)
                console.print(f"[dim]Skill {name} is no longer approved.[/dim]")
            else:
                skill = registry.trust(name)
                console.print(f"[green]✓ Skill {name} approved[/green] [dim](sha256 {skill.digest[:16]}…)[/dim]")
        except SkillNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return False
        await audit.record(actor="user", action=f"skills.{action}", target=name, outcome="ok", session_id=session_id)
        return True
    if arg:
        console.print("[yellow]Usage: /skills, /skills trust <name>, /skills untrust <name>[/yellow]")
        return False

    skills = registry.skills()
    if not skills:
        console.print("[dim]This repo has no skills (.kartrix/skills/).[/dim]")
        return False
    colors = {"trusted": "green", "untrusted": "yellow", "changed": "red"}
    for skill in skills:
        status = registry.status(skill.name)
        files = [p for p in sorted(skill.skill_dir.rglob("*")) if p.is_file()]
        console.print(
            f"[bold]{skill.name}[/bold] [{colors[status]}]{status}[/{colors[status]}] — {escape(skill.description)}"
        )
        console.print(f"  [dim]{len(files)} files in {skill.skill_dir.relative_to(Path.cwd()).as_posix()}[/dim]")
        if skill.findings:
            rules = ", ".join(sorted({f.rule for f in skill.findings}))
            console.print(f"  [red]⚠ contains text that looks like prompt injection ({rules})[/red]")
    console.print("[dim]Review a skill's files before approving it: /skills trust <name>[/dim]")
    return False


async def initialize(checkpointer):
    """Bootstrap LLM, embedder, index, watcher, MCP tools, cache, and session before the REPL starts."""
    # Built once up front so a missing API key or bad provider config fails at startup.
    get_llm()
    get_embedder()
    console.print(f"[dim]LLM: {settings.llm.provider} / {settings.llm.model}[/dim]")
    console.print(f"[dim]Embedder: {settings.embeddings.provider} / {settings.embeddings.model}[/dim]")

    repo_path = str(Path.cwd())
    workspace = set_workspace(repo_path)  # file tools may only touch this tree
    console.print(f"[dim]Workspace: {workspace.root} · permission mode: {get_mode()}[/dim]")
    await update_index()

    semantic_cache = await build_semantic_cache()
    cache_domain = get_repo_domain(repo_path) if semantic_cache else None
    if semantic_cache is not None:
        console.print(f"[dim]Semantic cache: enabled (threshold={semantic_cache.threshold})[/dim]")
    else:
        console.print("[dim]Semantic cache: disabled[/dim]")

    loop = asyncio.get_running_loop()

    async def _invalidate_cache_on_change() -> None:
        if semantic_cache is not None and cache_domain is not None:
            await semantic_cache.invalidate_domain(cache_domain)

    observer = start_watcher(repo_path, loop, on_change=_invalidate_cache_on_change)

    mcp = McpManager(t.name for t in NATIVE_TOOLS)
    for name, error in (await mcp.connect_remembered()).items():
        console.print(f"[yellow]MCP server {name} not connected: {error}[/yellow]")
    if mcp.connected:
        console.print(f"[dim]MCP: {', '.join(mcp.connected)} ({len(mcp.tools)} tools)[/dim]")
    pending_skills = [s.name for s in get_registry().skills() if get_registry().status(s.name) != "trusted"]
    if pending_skills:
        console.print(
            f"[yellow]This repo has skills you haven't approved: {', '.join(pending_skills)} — see /skills[/yellow]"
        )

    agent = build_agent(checkpointer, mcp.tools)
    session_id = get_current_session()
    await record_session(session_id, repo_path)
    console.print(f"[dim]Session: {session_id}[/dim]")
    console.print("[green]✓ Ready[/green]\n")
    await finish_pending_approval(agent, session_id)
    return agent, session_id, observer, semantic_cache, cache_domain, mcp


async def _run_async():
    logger.info("Starting Kartrix")
    console.print("\n[bold blue]Kartrix[/bold blue] — RAG-powered code assistant")

    checkpointer = get_checkpointer()
    agent, session_id, observer, semantic_cache, cache_domain, mcp = await initialize(checkpointer)
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
                    agent,
                    question,
                    session_id,
                    approver,
                    semantic_cache=semantic_cache,
                    cache_domain=cache_domain,
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
                clear_session_allowances()
                await record_session(session_id, str(Path.cwd()))
                console.print(f"[green]New session started: {session_id}[/green]")
            elif user_input.startswith("/switch "):
                target = user_input.removeprefix("/switch ").strip()
                try:
                    session_id = switch_session(target)
                    clear_session_allowances()
                    await record_session(session_id, str(Path.cwd()))
                except InvalidSessionIdError:
                    console.print("[red]Invalid session id — expected a UUID like the one shown by /session.[/red]")
                else:
                    console.print(f"[green]Switched to session: {session_id}[/green]")
                    await finish_pending_approval(agent, session_id)
            elif user_input == "/session":
                console.print(f"[dim]Current session: {session_id}[/dim]")
            elif user_input == "/show_index":
                logger.info("Showing index")
                await show_index(Path.cwd())
            elif user_input.startswith("/plan "):
                goal = user_input.removeprefix("/plan ").strip()
                logger.info(f"Plan command received: {goal}")
                await handle_plan_command(goal, session_id, approver)
            elif user_input == "/task_status":
                await show_task_status()
            elif user_input == "/audit" or user_input.startswith("/audit "):
                await show_audit(session_id, user_input.removeprefix("/audit").strip())
            elif user_input == "/mode" or user_input.startswith("/mode "):
                target = user_input.removeprefix("/mode").strip()
                if target:
                    try:
                        previous = get_mode()
                        set_mode(target)
                    except ValueError as e:
                        console.print(f"[red]{e}[/red]")
                    else:
                        await audit.record(
                            actor="user",
                            action="permissions.mode",
                            target=target,
                            outcome="ok",
                            session_id=session_id,
                            details={"previous": previous},
                        )
                console.print(f"[dim]Permission mode: {get_mode()} (available: {', '.join(MODES)})[/dim]")
            elif user_input.split()[0] in ("/mcp", "/connect", "/disconnect"):
                command, _, arg = user_input.partition(" ")
                if await handle_mcp_command(command, arg.strip(), mcp, session_id):
                    agent = build_agent(checkpointer, mcp.tools)
            elif user_input == "/skills" or user_input.startswith("/skills "):
                if await handle_skills_command(user_input.removeprefix("/skills").strip(), session_id):
                    agent = build_agent(checkpointer, mcp.tools)
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
                console.print("  [bold]/mode [read_only|default|auto][/bold] — show or change the permission mode")
                console.print("  [bold]/audit [n][/bold]                — show this session's last n tool calls")
                console.print("  [bold]/mcp[/bold]                     — list MCP servers")
                console.print("  [bold]/connect <server>[/bold]        — connect an MCP server (e.g. github)")
                console.print("  [bold]/disconnect <server>[/bold]     — disconnect it")
                console.print("  [bold]/skills [trust|untrust <name>][/bold] — review this repo's skills")
    finally:
        await mcp.close()  # same task that opened the sessions
        stop_watcher(observer)
        await dispose_engine()


def run():
    asyncio.run(_run_async())


if __name__ == "__main__":
    run()
