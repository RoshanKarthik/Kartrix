"""The chat agent as an explicit LangGraph ``StateGraph`` with focused subagents (step 3.1/3.3).

::

    START → route ─┬─ chat ────────────────────────────────┐
                   ├─ question → explore ──────────────────┤
                   └─ change ──→ explore → code ⇄ review ──┴→ respond → END

- **route** — a small structured call on the cheap model (``llm.router_model``, model routing):
  is the request small talk, a question about the code, or a change?
- **explore** — the *explorer* subagent: read-only tools (search, read, grep, glob, skills). It
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
the request, a short excerpt of the conversation and the previous step's report — not each other's
tool noise. A reached budget skips straight to the end.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from kartrix.agent.tools import search_codebase
from kartrix.config import settings
from kartrix.core.events import AgentStep, emit
from kartrix.llm.factory import get_chat_model, get_model_middleware
from kartrix.observability.logger import get_logger
from kartrix.security.budget import BudgetMiddleware
from kartrix.security.budget import current as current_budget
from kartrix.security.injection import SECURITY_RULES
from kartrix.tools.filesystem_tools import READ_TOOLS, WRITE_TOOLS
from kartrix.tools.terminal_tools import run_command

logger = get_logger(__name__)

Route = Literal["chat", "question", "change"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    route: Route
    context: str  # assembled context for this turn (memory, project notes) — see kartrix.agent.context
    findings: str
    change_report: str
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
- change: create, edit, fix, refactor, test or run something in the repository"""

EXPLORER_PROMPT = f"""You are the explorer: a read-only code investigator.
Find what the request needs in this repository: use search_codebase first, then grep, glob and read_file
for exact code. Do not change anything. Report concise findings: the relevant files, functions and
line numbers, how they work, and anything a developer must know to answer or implement the request.
If something can't be found, say so.
{SECURITY_RULES}"""

CODER_PROMPT = f"""You are the coder: you implement changes in this repository.
Make the smallest correct change that does what was asked, matching the existing code style. Read a
file before editing it; use edit_file for small changes. Add or update tests when behaviour changes and
run them with run_command. Finish with a short report: the files you changed, what you changed and
the result of the tests.
{SECURITY_RULES}"""

REVIEWER_PROMPT = f"""You are the reviewer: you independently check a change another agent made.
Read the changed files and run the relevant tests. Approve only if the change does everything that was
asked, is correct, keeps the existing behaviour and style, and the tests pass. Otherwise list concrete
issues to fix. Do not edit files yourself.
{SECURITY_RULES}"""

RESPOND_PROMPT = """You are Kartrix, a senior software engineer helping the user with their repository.
Answer the user's latest message using the reports from your team (explorer, coder, reviewer) below.
Reference specific files, functions and line numbers. If the work was stopped or the reviewer still
had issues, say so plainly. Be concise."""


def _text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def _request(state: AgentState) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return _text(m.content)
    return ""


def _excerpt(state: AgentState, turns: int = 6, chars: int = 4000) -> str:
    """The last few turns before the current request (subagents don't see the whole history)."""
    msgs = [m for m in state.get("messages", [])[:-1] if isinstance(m, HumanMessage | AIMessage) and _text(m.content)]
    lines = [f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {_text(m.content)}" for m in msgs[-turns:]]
    text = "\n".join(lines)
    return text[-chars:]


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
        explorer_tools = [search_codebase, *READ_TOOLS, *skills, *mcp_tools]
        coder_tools = [search_codebase, *READ_TOOLS, *WRITE_TOOLS, run_command, *skills, *mcp_tools]
        reviewer_tools = [*READ_TOOLS, run_command]
        self.explorer = create_agent(
            llm, tools=explorer_tools, system_prompt=EXPLORER_PROMPT + extra, middleware=middleware(None)
        )
        self.coder = create_agent(
            llm, tools=coder_tools, system_prompt=CODER_PROMPT + extra, middleware=middleware(approval())
        )
        self.reviewer = create_agent(
            llm, tools=reviewer_tools, system_prompt=REVIEWER_PROMPT, middleware=middleware(approval()),
            response_format=ReviewVerdict,
        )  # fmt: skip
        self.router = create_agent(
            get_chat_model("router", temperature=0), tools=[], system_prompt=ROUTER_PROMPT,
            response_format=RouteDecision, middleware=[*get_model_middleware(temperature=0), BudgetMiddleware()],
        )  # fmt: skip
        self.responder = create_agent(
            llm, tools=[], system_prompt=RESPOND_PROMPT,
            middleware=[*get_model_middleware(), BudgetMiddleware()],
        )  # fmt: skip

    # ── nodes ─────────────────────────────────────────────────────────

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
        emit(AgentStep(agent="router", status="finished", summary=route))
        return {
            "route": route,
            "rounds": 0,
            "findings": "",
            "change_report": "",
            "review_notes": "",
            "steps": ["router"],
        }

    async def explore(self, state: AgentState) -> dict[str, Any]:
        prompt = f"Request: {_request(state)}"
        if excerpt := _excerpt(state):
            prompt += f"\n\nEarlier conversation:\n{excerpt}"
        if state.get("context"):
            prompt += f"\n\nProject context:\n{state['context']}"
        findings = await self._run("explorer", self.explorer, prompt)
        return {"findings": findings, "steps": ["explorer"]}

    async def code(self, state: AgentState) -> dict[str, Any]:
        prompt = f"Request: {_request(state)}\n\nExplorer findings:\n{state.get('findings', '')}"
        if state.get("context"):
            prompt += f"\n\nProject context:\n{state['context']}"
        if state.get("review_notes"):
            prompt += f"\n\nThe reviewer rejected your previous attempt. Fix these issues:\n{state['review_notes']}"
        report = await self._run("coder", self.coder, prompt)
        return {"change_report": report, "rounds": state.get("rounds", 0) + 1, "steps": ["coder"]}

    async def review(self, state: AgentState) -> dict[str, Any]:
        prompt = (
            f"Request: {_request(state)}\n\nThe coder's report:\n{state.get('change_report', '')}\n\n"
            "Check the change and give your verdict."
        )
        emit(AgentStep(agent="reviewer", status="started"))
        out = await self.reviewer.ainvoke({"messages": [HumanMessage(prompt)]})
        verdict: ReviewVerdict | None = out.get("structured_response")
        approved = verdict.approved if verdict is not None else False
        notes = "\n".join(f"- {i}" for i in verdict.issues) if verdict is not None else "no verdict (stopped)"
        emit(AgentStep(agent="reviewer", status="finished", summary="approved" if approved else f"issues:\n{notes}"))
        return {"approved": approved, "review_notes": "" if approved else notes, "steps": ["reviewer"]}

    async def respond(self, state: AgentState) -> dict[str, Any]:
        if reason := _stopped():
            return {"messages": [AIMessage(f"Stopped: {reason}.")]}
        parts = [f"User's message: {_request(state)}"]
        if excerpt := _excerpt(state):
            parts.append(f"Earlier conversation:\n{excerpt}")
        if state.get("context"):
            parts.append(f"Project context:\n{state['context']}")
        if state.get("findings"):
            parts.append(f"Explorer's findings:\n{state['findings']}")
        if state.get("change_report"):
            parts.append(f"Coder's report:\n{state['change_report']}")
        if state.get("route") == "change":
            verdict = (
                "approved the change" if state.get("approved") else f"still has issues:\n{state.get('review_notes')}"
            )
            parts.append(f"Reviewer {verdict}")
        emit(AgentStep(agent="responder", status="started"))
        out = await self.responder.ainvoke({"messages": [HumanMessage("\n\n".join(parts))]})
        answer = _text(out["messages"][-1].content) if out.get("messages") else ""
        emit(AgentStep(agent="responder", status="finished"))
        return {"messages": [AIMessage(answer)], "steps": ["responder"]}

    async def _run(self, name: str, agent: Any, prompt: str) -> str:
        emit(AgentStep(agent=name, status="started"))
        out = await agent.ainvoke({"messages": [HumanMessage(prompt)]})
        report = _text(out["messages"][-1].content) if out.get("messages") else ""
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
        if assemble is not None:
            graph.add_edge(START, "assemble")
            graph.add_edge("assemble", "route")
        else:
            graph.add_edge(START, "route")
        graph.add_conditional_edges("route", self.after_route, ["explore", "respond"])
        graph.add_conditional_edges("explore", self.after_explore, ["code", "respond"])
        graph.add_edge("code", "review")
        graph.add_conditional_edges("review", self.after_review, ["code", "respond"])
        graph.add_edge("respond", END)
        return graph.compile(checkpointer=checkpointer, name="kartrix")
