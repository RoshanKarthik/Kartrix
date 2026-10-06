"""PgCheckpointSaver must behave exactly like LangGraph's reference InMemorySaver."""

import asyncio
import operator
import uuid
from typing import Annotated, Any, TypedDict

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from kartrix.memory.checkpointer import PgCheckpointSaver

pytestmark = pytest.mark.usefixtures("db")


class State(TypedDict, total=False):
    log: Annotated[list[str], operator.add]
    approved: str


def _a(s: State) -> State:
    return {"log": ["a"]}


def _ask(s: State) -> State:
    return {"approved": interrupt({"question": "ok?", "nul": "x\x00y"})}  # NUL must survive JSONB


def _b(s: State) -> State:
    return {"log": [f"b:{s['approved']}"]}


def _graph(saver: BaseCheckpointSaver[Any]) -> Any:
    g = StateGraph(State)
    g.add_node("a", _a)
    g.add_node("ask", _ask)
    g.add_node("b", _b)
    g.add_edge(START, "a")
    g.add_edge("a", "ask")
    g.add_edge("ask", "b")
    g.add_edge("b", END)
    return g.compile(checkpointer=saver)


async def _scenario(saver: BaseCheckpointSaver[Any]) -> dict[str, Any]:
    g = _graph(saver)
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    r1 = await g.ainvoke({"log": ["start"]}, cfg)  # stops at the interrupt
    paused = await g.aget_state(cfg)
    r2 = await g.ainvoke(Command(resume="yes"), cfg)
    r3 = await g.ainvoke({"log": ["again"]}, cfg)  # second turn: state accumulates
    hist = [c async for c in saver.alist(cfg)]
    return {
        "interrupt": [i.value for i in r1["__interrupt__"]],
        "next": paused.next,
        "r2": r2["log"],
        "r3": r3["log"],
        "history": len(hist),
        "limit": len([c async for c in saver.alist(cfg, limit=2)]),
        "before": len([c async for c in saver.alist(cfg, before=hist[1].config)]),
        "filter": sorted(c.metadata["step"] for c in [c async for c in saver.alist(cfg, filter={"source": "input"})]),
    }


async def test_matches_in_memory_saver() -> None:
    assert await _scenario(PgCheckpointSaver()) == await _scenario(InMemorySaver())


async def test_sync_api_from_other_thread_only() -> None:
    saver = PgCheckpointSaver()
    g = _graph(saver)
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    await g.ainvoke({"log": ["start"]}, cfg)
    expected = await saver.aget_tuple(cfg)
    assert expected is not None
    from_thread = await asyncio.to_thread(saver.get_tuple, cfg)
    assert from_thread is not None and from_thread.config == expected.config
    assert len(await asyncio.to_thread(lambda: list(saver.list(cfg, limit=1)))) == 1
    with pytest.raises(asyncio.InvalidStateError):
        saver.get_tuple(cfg)  # on the loop thread this would deadlock, so it must refuse


async def test_delete_thread_removes_everything() -> None:
    saver = PgCheckpointSaver()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    other = {"configurable": {"thread_id": str(uuid.uuid4())}}
    await _graph(saver).ainvoke({"log": ["x"]}, cfg)
    await _graph(saver).ainvoke({"log": ["y"]}, other)
    await saver.adelete_thread(cfg["configurable"]["thread_id"])
    assert await saver.aget_tuple(cfg) is None
    assert [c async for c in saver.alist(cfg)] == []
    assert await saver.aget_tuple(other) is not None


async def test_versions_increase() -> None:
    saver = PgCheckpointSaver()
    v1 = saver.get_next_version(None, None)
    v2 = saver.get_next_version(v1, None)
    assert v1 < v2 and v2.startswith("0" * 31 + "2.")
    assert saver.get_next_version(7, None).startswith("0" * 31 + "8.")  # int versions from old checkpoints
