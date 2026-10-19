"""The server's pieces that need no database: password hashing, the worker protocol, run feeds,
the worker's remote approver/reviewer and the client's SSE parser."""

from __future__ import annotations

import asyncio
import io
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest

from kartrix.client.api import _parse_sse
from kartrix.security.approvals import ApprovalRequest
from kartrix.server import auth
from kartrix.server.protocol import OP_ADAPTER, ApprovalOp, Decision, PlanReviewOp, RunOp, decode, encode
from kartrix.server.worker import Channel, RemoteFrontEnd
from kartrix.server.workers import RunFeed
from kartrix.tasks.planner import ExecutionPlan

# ── accounts ──────────────────────────────────────────────────────────


def test_password_hash_round_trip() -> None:
    stored = auth.hash_password("correct horse battery")
    assert stored.startswith("scrypt$")
    assert "correct horse" not in stored
    assert auth.verify_password("correct horse battery", stored)
    assert not auth.verify_password("correct horse batterY", stored)
    assert auth.hash_password("correct horse battery") != stored  # salted


@pytest.mark.parametrize("stored", ["", "bcrypt$x", "scrypt$1$2$3$zz$00", "scrypt$a$b$c$d$e"])
def test_verify_rejects_malformed_hashes(stored: str) -> None:
    assert not auth.verify_password("anything", stored)


def test_email_and_password_rules() -> None:
    assert auth.normalize_email("  Ada@Example.COM ") == "ada@example.com"
    for bad in ("ada", "ada@", "@example.com", "a b@example.com", "ada@example"):
        with pytest.raises(ValueError):
            auth.normalize_email(bad)
    auth.check_password_rules("8 chars!")
    with pytest.raises(ValueError):
        auth.check_password_rules("short")


def test_tokens_are_stored_hashed() -> None:
    assert auth.token_hash("kx_abc") != "kx_abc"
    assert len(auth.token_hash("kx_abc")) == 64


def test_bearer_token_header() -> None:
    assert auth.bearer_token("Bearer kx_abc") == "kx_abc"
    for bad in (None, "", "Basic abc", "Bearer "):
        with pytest.raises(Exception, match="missing bearer token"):
            auth.bearer_token(bad)


# ── protocol ──────────────────────────────────────────────────────────


def test_ops_round_trip() -> None:
    run = RunOp(run_id="r1", kind="plan", input="build it", session_id=str(uuid.uuid4()))
    assert OP_ADAPTER.validate_python(decode(encode(run))) == run
    approval = ApprovalOp(request_id="q", decisions=[Decision(type="edit", command="pytest -q")])
    assert OP_ADAPTER.validate_python(decode(encode(approval))) == approval
    with pytest.raises(ValueError):
        decode("[1, 2]")


# ── run feeds ─────────────────────────────────────────────────────────


async def test_feed_backlog_and_live_items_without_gaps_or_repeats() -> None:
    feed = RunFeed(uuid.uuid4(), uuid.uuid4())
    feed.publish("event", {"type": "notice", "text": "one"})
    feed.publish("event", {"type": "assistant_delta", "text": "tok"}, persist=False)
    feed.publish("event", {"type": "notice", "text": "two"})
    backlog, queue = feed.subscribe(after=1)
    assert [i.seq for i in backlog] == [2]  # deltas have no number and are never replayed
    feed.publish("approval_needed", {"request_id": "q"})
    feed.close()
    live = [await queue.get(), await queue.get()]
    assert live[0] is not None and live[0].seq == 3
    assert live[1] is None


async def test_subscribing_to_a_finished_feed_ends_at_once() -> None:
    feed = RunFeed(uuid.uuid4(), uuid.uuid4())
    feed.publish("run_finished", {"status": "completed"})
    feed.close()
    backlog, queue = feed.subscribe(after=0)
    assert [i.kind for i in backlog] == ["run_finished"]
    assert await queue.get() is None


# ── worker front end ──────────────────────────────────────────────────


def _request() -> ApprovalRequest:
    return ApprovalRequest("call1", "run_command", "rm -rf build", ".", "destructive", "deletes files", "default", True)


async def test_remote_approver_waits_for_the_servers_answer() -> None:
    out = io.StringIO()
    front = RemoteFrontEnd(Channel(out))
    asking = asyncio.create_task(front.approve([_request()]))
    await asyncio.sleep(0)
    sent = json.loads(out.getvalue())
    assert sent["kind"] == "approval_needed" and sent["requests"][0]["command"] == "rm -rf build"
    assert not front.answer(PlanReviewOp(request_id=sent["request_id"], approved=True))  # wrong kind of answer
    assert front.answer(ApprovalOp(request_id=sent["request_id"], decisions=[Decision(type="reject", message="no")]))
    decisions = await asking
    assert decisions[0].type == "reject" and decisions[0].message == "no"
    assert not front.answer(ApprovalOp(request_id=sent["request_id"], decisions=[]))  # answered already


async def test_stop_answers_open_approvals_with_no() -> None:
    front = RemoteFrontEnd(Channel(io.StringIO()))
    asking = asyncio.create_task(front.approve([_request(), _request()]))
    await asyncio.sleep(0)
    front.cancel_all()
    assert [d.type for d in await asking] == ["reject", "reject"]


async def test_remote_plan_review() -> None:
    from kartrix.security.budget import RunStopped

    plan = ExecutionPlan(
        project_name="p", goal_summary="g", tech_stack=[], total_estimated_hours=1, tasks=[], risks=[], assumptions=[]
    )
    out = io.StringIO()
    front = RemoteFrontEnd(Channel(out))

    async def review(answer: Any) -> Any:
        task = asyncio.create_task(front.review_plan(plan))
        await asyncio.sleep(0)
        request_id = json.loads(out.getvalue().splitlines()[-1])["request_id"]
        if answer == "stop":
            front.cancel_all()
        else:
            front.answer(PlanReviewOp(request_id=request_id, **answer))
        return await task

    assert (await review({"approved": True})).plan == plan
    rejected = await review({"approved": False, "feedback": "smaller"})
    assert rejected.plan is None and rejected.feedback == "smaller"
    edited = await review({"approved": True, "plan": {**plan.model_dump(), "project_name": "q"}})
    assert edited.plan.project_name == "q"
    with pytest.raises(RunStopped):
        await review("stop")


# ── client SSE parser ─────────────────────────────────────────────────


async def test_parse_sse() -> None:
    async def lines() -> AsyncIterator[str]:
        for line in [": ping", "", "id: 3", "event: event", 'data: {"type": "notice",', 'data: "text": "hi"}', ""]:
            yield line
        for line in ["event: event", 'data: {"type": "assistant_delta"}', ""]:
            yield line

    messages = [m async for m in _parse_sse(lines())]
    assert [(m.kind, m.seq) for m in messages] == [("event", 3), ("event", None)]
    assert messages[0].data == {"type": "notice", "text": "hi"}


# ── worker pipes ──────────────────────────────────────────────────────

_CHILD_WHILE_READING = r"""
import subprocess, sys, threading, time
from kartrix.server.worker import private_protocol_pipes
source, out = private_protocol_pipes()
threading.Thread(target=lambda: [None for _ in source], daemon=True).start()  # blocked on the pipe, as in a worker
time.sleep(0.3)
subprocess.run([sys.executable, "-c", "pass"], timeout=20)  # inherits fds 0-2
out.write("child ok\n")
out.flush()
"""


def test_child_processes_do_not_inherit_the_protocol_pipe() -> None:
    """On Windows a child inheriting a pipe the worker blocks on hangs (git checkpoints did, for 300 s)."""
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", _CHILD_WHILE_READING], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        out, _ = proc.communicate(timeout=60)  # stdin stays open until the child has finished
    finally:
        proc.kill()
    assert b"child ok" in out
