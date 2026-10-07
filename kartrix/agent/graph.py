"""The chat agent as an explicit LangGraph ``StateGraph`` with focused subagents (step 3.1/3.3).

::

    START → compress → assemble → route ─┬─ chat ────────────────────────────────┐
                                         ├─ question → explore ──────────────────┤
                                         └─ change ──→ explore → code ⇄ review ──┴→ respond → END

- **compress** — once the conversation passes ``memory.summarize_at_tokens``, old turns become one
  summary (``kartrix.agent.context.compress``).
- **assemble** — ``KARTRIX.md`` and the long-term memories relevant to the request, as the ``context``
  the subagents get (``kartrix.agent.context.assemble``).

- **route** — a small structured call on the cheap model (``llm.router_model``, model routing):
  is the request small talk, a question about the code, or a change?
- **explore** — the *explorer* subagent: read-only tools (search, read, grep, glob, skills, remember). It
  investigates in a clean context and reports findings with file:line references.
- **code** — the *coder* subagent: read/write tools and ``run_command`` (with the approval
  middleware: a command that needs approval pauses the whole graph until the user answers).
- **review** — the *reviewer* subagent: an independent check of the change against the request
  in a fresh context (it can read files and run the tests), with a structured verdict. A rejected
  change goes back to the coder with the reviewer's notes (at most ``agents.max_review_rounds``).
- **respond** — the main model answers the user from what the subagents reported.

Each subagent is a ``create_agent`` graph with the shared middleware stack (fallback/retry, audit,
budget, content guard), so budgets, the kill switch, auditing and injection defences cover every
step. The conversation (``messages``) lives in the parent graph's checkpointer; subagents get only
the assembled context (project instructions, memories, recent turns — each within its token budget),
the previous step's report and the request, last — not each other's tool noise. A reached budget skips straight to the end.
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from kartrix.agent import context as agent_context
from kartrix.agent.reliability import CompletionGuardMiddleware, final_text
from kartrix.agent.streaming import AnswerStream
from kartrix.agent.tools import remember, search_codebase, symbol_graph
from kartrix.config import settings
from kartrix.core.events import AgentStep, Event, ToolCallFinished, ToolCallStarted, collecting, emit
from kartrix.llm.factory import get_chat_model, get_model_middleware
from kartrix.memory import long_term
from kartrix.observability.logger import get_logger
from kartrix.security.budget import BudgetMiddleware
from kartrix.security.budget import current as current_budget
from kartrix.security.injection import SECURITY_RULES
from kartrix.tools.filesystem_tools import READ_TOOLS, WRITE_TOOLS
from kartrix.tools.terminal_tools import REVIEW_REFUSED, review_command, run_command

logger = get_logger(__name__)

Route = Literal["chat", "question", "change"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    route: Route
    context: str  # assembled context for this turn (memory, project notes) — see kartrix.agent.context
    findings: str
    change_report: str
    changed_files: list[str]  # what the coder's write tools changed this turn (from its tool events)
    review_notes: str
    approved: bool
    rounds: int
    steps: Annotated[list[str], operator.add]  # which agents ran this turn, in order


class RouteDecision(BaseModel):
    """How to handle the user's latest message."""

    route: Route = Field(description="chat: greeting/small talk/meta; question: about the code; change: modify files")
    reason: str = Field(description="one short sentence")


class ReviewVerdict(BaseModel):
    approved: bool = Field(description="true if the change fully and correctly does what was asked")
    issues: list[str] = Field(default_factory=list, description="concrete problems to fix, empty if approved")


ROUTER_PROMPT = """You route requests for a coding assistant working inside a code repository.
Decide how to handle the user's latest message:
- chat: greetings, thanks, questions about you, anything not about this repository's code
- question: understanding, locating or explaining code — nothing must change
- change: create, edit, fix, refactor, test or run something in the repository

Examples:
- "Add a --json flag to the stats command" → change
- "slugify returns X but should return Y" → change (a bug report asks for a fix)
- "Write tests for the parser" / "Rename foo to bar" / "Run the tests" → change
- "Where is the config loaded?" / "Why does login fail?" / "What does paginate do?" → question
- "Thanks!" / "Which model are you?" → chat
When unsure between question and change, choose change if the user expects files to be different afterwards."""

EXPLORER_PROMPT = f"""You are the explorer: a read-only code investigator.
Find what the request needs in this repository: use search_codebase first, then grep, glob and read_file
for exact code, and symbol_graph to see who calls a function and what it calls. Do not change anything. Report concise findings: the relevant files, functions and
line numbers, how they work, and anything a developer must know to answer or implement the request.
If something can't be found, say so. If you learn a lasting convention about this repository, save it
with remember.
{SECURITY_RULES}"""

WORKING_RULES = """Working rules:
- run_command runs one program directly — no shell, no pipes, no "&&", no "cmd /c", no "python -c".
  Never write helper scripts to work around a command; put checks into the project's tests instead.
- If a command fails because of the environment (a missing tool, a permission or sandbox error) rather
  than the code, don't fight it: stop and say so in your report.
- A rename (or any changed name or message) means every occurrence: code, tests and docs such as the README.
  Search the whole repository for the old name with grep, without a file-type filter, and check it is gone.
- Only create the files the task needs. Delete any temporary file you created with delete_file.
- Don't re-read a file you already read unless you changed it since."""

CODER_PROMPT = f"""You are the coder: you implement changes in this repository.
Make the smallest correct change that does what was asked, matching the existing code style. Read a
file before editing it; use edit_file for small changes. Add or update tests when behaviour changes and
run the project's tests with run_command (e.g. "pytest -q", "node --test", "npm test").

{WORKING_RULES}

Finish with a short report: the files you changed, what you changed and the result of the tests. Follow
the project instructions and remembered preferences and lessons you are given; save a new lasting fact
or user preference with remember.
{SECURITY_RULES}"""

REVIEWER_PROMPT = f"""You are the reviewer: you independently check a change another agent made.
Read the changed files and run the project's tests with run_command (one program, no shell) in every review,
including after a fix — never approve, or say the tests can't run, without having run them. Approve
only if the change does everything that was asked, is correct, keeps the existing behaviour and style,
the tests pass, and no unrelated files (scratch scripts, notes, debug output) were added. For a rename, grep the
whole repository (docs and README included) for the old name. Otherwise list
concrete issues to fix. Do not edit files yourself, and don't retry a command that fails because of the
environment — report it as an issue. A step the user declined or the permissions blocked (a download, an install)
can't be done in this run: don't ask for it again — judge the rest, and approve when the rest is right and the
coder's report says plainly what was not done.
{SECURITY_RULES}"""

RESPOND_PROMPT = """You are Kartrix, a senior software engineer helping the user with their repository.
Answer the user's latest message using the reports from your team (explorer, coder, reviewer) below.
Reference specific files, functions and line numbers. Be concise.
When the team made a change, it is already done — the files listed under "Files changed" contain it. Say what
was changed and the test result; never tell the user to make the change themselves. The explorer's notes are
from before the change. If the work was stopped, a step was declined or blocked, or the reviewer still had
issues, say so plainly and say what is left."""


def _text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def _request(state: AgentState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return _text(m.content)
    return ""


def _brief(state: AgentState, *parts: str) -> str:
    """A subagent's prompt: the assembled context first (stable → volatile, for prompt caching), then this
    step's inputs, the request last. Subagents never see the whole history, only the budgeted context."""
    context = state.get("context", "")
    if not context:  # compiled without an assemble node: the recent turns only
        section = agent_context.conversation_section(state.get("messages", []), settings.context.conversation_tokens)
        context = section.render() if section else ""
    blocks = [context, *parts, f"## Request (the user's latest message)\n{_request(state)}"]
    return "\n\n".join(b for b in blocks if b)


WRITE_TOOL_NAMES = frozenset({"write_file", "edit_file", "append_file", "delete_file"})


def _written_files(events: list[Event], before: list[str]) -> list[str]:
    """Paths the coder's write tools changed successfully, in order, added to ``before``."""
    targets = {e.call_id: e.target for e in events if isinstance(e, ToolCallStarted) and e.tool in WRITE_TOOL_NAMES}
    out = list(before)
    for e in events:
        if (
            isinstance(e, ToolCallFinished)
            and e.outcome == "ok"
            and (path := targets.get(e.call_id))
            and path not in out
        ):
            out.append(path)
    return out


_CHANGE_VERBS = re.compile(
    r"^\s*(?:please\s+|can you\s+|could you\s+)?(?:add|fix|implement|create|make|change|update|remove|delete|"
    r"rename|refactor|write|build|replace|move|convert|support|handle|improve|optimi[sz]e|migrate|upgrade|bump|"
    r"edit|set up|setup|install|extend|split|merge|introduce|drop|clean up|reformat|format|run)\b",
    re.IGNORECASE,
)


def looks_like_change(request: str) -> bool:
    """An imperative request ("Add …", "Fix …", "Please rename …") asks for files to change, whatever the router
    model thought: a small router occasionally calls those questions, and then nothing gets done."""
    return bool(_CHANGE_VERBS.match(request))


def _stopped() -> str | None:
    budget = current_budget()
    return budget.exhausted() if budget is not None else None


class AgentGraph:
    """Builds the subagents and the graph. ``middleware`` makes the shared stack for one subagent
    (``kartrix.agent.factory.agent_middleware``); ``approval`` builds the approval middleware."""

    def __init__(
        self,
        middleware: Any,
        approval: Any,
        mcp_tools: list[Any] | None = None,
        skills: Any = (),
        skills_prompt: str = "",
    ) -> None:
        mcp_tools = mcp_tools or []
        extra = f"\n\n{skills_prompt}" if skills_prompt else ""
        llm = get_chat_model("main")
        explorer_tools = [search_codebase, symbol_graph, *READ_TOOLS, remember, *skills, *mcp_tools]
        coder_tools = [
            search_codebase,
            symbol_graph,
            *READ_TOOLS,
            *WRITE_TOOLS,
            run_command,
            remember,
            *skills,
            *mcp_tools,
        ]
        reviewer_tools = [*READ_TOOLS, review_command]
        self.explorer = create_agent(
            llm, tools=explorer_tools, system_prompt=EXPLORER_PROMPT + extra, middleware=middleware(None)
        )
        self.coder = create_agent(
            llm, tools=coder_tools, system_prompt=CODER_PROMPT + extra, middleware=middleware(approval())
        )
        self.reviewer = create_agent(
            llm, tools=reviewer_tools, system_prompt=REVIEWER_PROMPT, middleware=middleware(approval(refuse=REVIEW_REFUSED)),
            response_format=ReviewVerdict,
        )  # fmt: skip
        self.router = create_agent(
            get_chat_model("router", temperature=0), tools=[], system_prompt=ROUTER_PROMPT,
            response_format=RouteDecision, middleware=[*get_model_middleware(temperature=0), BudgetMiddleware()],
        )  # fmt: skip
        self.responder = create_agent(
            llm, tools=[], system_prompt=RESPOND_PROMPT,
            middleware=[*get_model_middleware(), BudgetMiddleware(), CompletionGuardMiddleware()],
        )  # fmt: skip

    # ── nodes ─────────────────────────────────────────────────────────

    @staticmethod
    async def compress(state: AgentState) -> dict[str, Any]:
        return await agent_context.compress(state)

    async def route(self, state: AgentState) -> dict[str, Any]:
        request = _request(state)
        emit(AgentStep(agent="router", status="started"))
        try:
            out = await self.router.ainvoke({"messages": [HumanMessage(request)]})
            decision: RouteDecision = out["structured_response"]
            route: Route = decision.route
        except Exception as e:  # a router failure must not lose the request: treat it as a question
            logger.warning("Router failed; treating the request as a question", extra={"error": repr(e)})
            route = "question"
        if route == "question" and looks_like_change(request):
            logger.info("Router said question for an imperative request; treating it as a change")
            route = "change"
        emit(AgentStep(agent="router", status="finished", summary=route))
        return {
            "route": route,
            "rounds": 0,
            "findings": "",
            "change_report": "",
            "changed_files": [],
            "review_notes": "",
            "steps": ["router"],
        }

    async def explore(self, state: AgentState) -> dict[str, Any]:
        findings = await self._run("explorer", self.explorer, _brief(state))
        return {"findings": findings, "steps": ["explorer"]}

    async def code(self, state: AgentState) -> dict[str, Any]:
        parts = [f"## Explorer findings\n{state.get('findings', '')}"]
        if state.get("review_notes"):
            parts.append(f"## The reviewer rejected your previous attempt. Fix these issues\n{state['review_notes']}")
        with collecting() as seen:
            report = await self._run("coder", self.coder, _brief(state, *parts))
        return {
            "change_report": report,
            "changed_files": _written_files(seen, state.get("changed_files", [])),
            "rounds": state.get("rounds", 0) + 1,
            "steps": ["coder"],
        }

    async def review(self, state: AgentState) -> dict[str, Any]:
        files = "\n".join(f"- {f}" for f in state.get("changed_files", [])) or "(none)"
        prompt = (
            f"Request: {_request(state)}\n\nThe coder's report:\n{state.get('change_report', '')}\n\n"
            f"Files the coder changed (from its tool calls):\n{files}\n\n"
            "Check the change and give your verdict."
        )
        emit(AgentStep(agent="reviewer", status="started"))
        out = await self.reviewer.ainvoke({"messages": [HumanMessage(prompt)]})
        verdict: ReviewVerdict | None = out.get("structured_response")
        approved = verdict.approved if verdict is not None else False
        notes = "\n".join(f"- {i}" for i in verdict.issues) if verdict is not None else "no verdict (stopped)"
        emit(AgentStep(agent="reviewer", status="finished", summary="approved" if approved else f"issues:\n{notes}"))
        if approved and state.get("review_notes"):
            await self._learn(_request(state), state["review_notes"])
        return {"approved": approved, "review_notes": "" if approved else notes, "steps": ["reviewer"]}

    async def respond(self, state: AgentState) -> dict[str, Any]:
        if reason := _stopped():
            return {"messages": [AIMessage(f"Stopped: {reason}.")]}
        parts = []
        if state.get("findings"):
            when = " (from before the change)" if state.get("route") == "change" else ""
            parts.append(f"## Explorer's findings{when}\n{state['findings']}")
        if state.get("changed_files"):
            parts.append("## Files changed\n" + "\n".join(f"- {f}" for f in state["changed_files"]))
        if state.get("change_report"):
            parts.append(f"## Coder's report\n{state['change_report']}")
        if state.get("route") == "change":
            verdict = (
                "approved the change" if state.get("approved") else f"still has issues:\n{state.get('review_notes')}"
            )
            parts.append(f"## Reviewer\nThe reviewer {verdict}")
        emit(AgentStep(agent="responder", status="started"))
        out = await self.responder.ainvoke(
            {"messages": [HumanMessage(_brief(state, *parts))]}, config={"callbacks": [AnswerStream()]}
        )
        answer = final_text(out.get("messages", []))
        if not answer:  # the model gave nothing back even after the nudges: hand over the reports themselves
            logger.warning("Responder returned no answer; falling back to the subagents' reports")
            answer = "\n\n".join(parts) or "Sorry — the model returned no answer. Please try again."
        emit(AgentStep(agent="responder", status="finished"))
        return {"messages": [AIMessage(answer)], "steps": ["responder"]}

    @staticmethod
    async def _learn(request: str, notes: str) -> None:
        """Episodic memory: a change the reviewer rejected and a later round fixed becomes a lesson."""
        lesson = f"For a request like '{request[:200]}', the reviewer first rejected the change for:\n{notes}"
        try:
            await long_term.remember(lesson, "lesson", source="review")
        except Exception as e:  # never fail the turn over a memory
            logger.warning("Could not store the review lesson", extra={"error": repr(e)})

    async def _run(self, name: str, agent: Any, prompt: str) -> str:
        emit(AgentStep(agent=name, status="started"))
        out = await agent.ainvoke({"messages": [HumanMessage(prompt)]})
        report = final_text(out.get("messages", []))
        if not report and not _stopped():
            logger.warning("Subagent returned no report", extra={"agent": name})
            report = f"(the {name} finished without a report)"
        emit(AgentStep(agent=name, status="finished", summary=report[:300]))
        return report

    # ── edges ─────────────────────────────────────────────────────────

    @staticmethod
    def after_route(state: AgentState) -> str:
        if _stopped():
            return "respond"
        return "respond" if state.get("route") == "chat" else "explore"

    @staticmethod
    def after_explore(state: AgentState) -> str:
        return "code" if state.get("route") == "change" and not _stopped() else "respond"

    @staticmethod
    def after_review(state: AgentState) -> str:
        if state.get("approved") or _stopped() or state.get("rounds", 0) >= settings.agents.max_review_rounds:
            return "respond"
        return "code"

    def compile(self, checkpointer: Any, assemble: Any = None) -> Any:
        graph = StateGraph(AgentState)
        if assemble is not None:
            graph.add_node("assemble", assemble)
        graph.add_node("route", self.route)
        graph.add_node("explore", self.explore)
        graph.add_node("code", self.code)
        graph.add_node("review", self.review)
        graph.add_node("respond", self.respond)
        graph.add_node("compress", self.compress)
        graph.add_edge(START, "compress")
        if assemble is not None:
            graph.add_edge("compress", "assemble")
            graph.add_edge("assemble", "route")
        else:
            graph.add_edge("compress", "route")
        graph.add_conditional_edges("route", self.after_route, ["explore", "respond"])
        graph.add_conditional_edges("explore", self.after_explore, ["code", "respond"])
        graph.add_edge("code", "review")
        graph.add_conditional_edges("review", self.after_review, ["code", "respond"])
        graph.add_edge("respond", END)
        return graph.compile(checkpointer=checkpointer, name="kartrix")
