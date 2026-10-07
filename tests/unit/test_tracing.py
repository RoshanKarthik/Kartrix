"""Local run traces (kartrix.observability.tracing): one file per run, summary and timeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from kartrix.core.events import (
    AgentStep,
    AssistantDelta,
    AssistantMessage,
    ModelCall,
    Notice,
    RunFinished,
    RunStarted,
    ToolCallFinished,
    ToolCallStarted,
    Usage,
    bus,
    emit,
    run_scope,
)
from kartrix.observability import tracing


def _run(recorder: tracing.TraceRecorder, run_id: str, question: str) -> None:
    unsubscribe = bus.subscribe(recorder)
    try:
        _emit_run(run_id, question)
    finally:
        unsubscribe()


def _emit_run(run_id: str, question: str) -> None:
    with run_scope(run_id):
        for event in (
            RunStarted(kind="ask", input=question, session_id="s"),
            AgentStep(agent="router", status="started"),
            ModelCall(model="small", input_tokens=100, output_tokens=10, duration_ms=500),
            AgentStep(agent="router", status="finished", summary="question"),
            AgentStep(agent="explorer", status="started"),
            ModelCall(model="big", input_tokens=2000, output_tokens=300, duration_ms=4000, finish_reason="tool_calls"),
            ToolCallStarted(call_id="c1", tool="read_file", target="src/app.py"),
            ToolCallFinished(call_id="c1", tool="read_file", outcome="ok", duration_ms=12),
            AgentStep(agent="explorer", status="finished", summary="found it"),
            AgentStep(agent="responder", status="started"),
            AssistantDelta(text="It is "),
            ModelCall(model="big", input_tokens=900, output_tokens=40, duration_ms=1000),
            AssistantMessage(text="It is in src/app.py."),
            RunFinished(kind="ask", status="completed", usage=Usage(input_tokens=3000, output_tokens=350)),
        ):
            emit(event)


def test_one_trace_file_per_run_without_deltas_or_global_events(tmp_path: Path) -> None:
    recorder = tracing.TraceRecorder(tmp_path)
    recorder(Notice(text="startup notice, not part of a run"))
    _run(recorder, "aaa111", "first?")
    _run(recorder, "bbb222", "second?")
    files = tracing.list_traces(tmp_path)
    assert len(files) == 2 and files[0].stem.endswith("bbb222")  # newest first
    events = tracing.load(files[1])
    types = [e["type"] for e in events]
    assert "assistant_delta" not in types and "notice" not in types
    assert types[0] == "run_started" and types[-1] == "run_finished"
    assert tracing.find("aaa", tmp_path) == files[1] and tracing.find("last", tmp_path) == files[0]


def test_old_traces_are_pruned(tmp_path: Path) -> None:
    recorder = tracing.TraceRecorder(tmp_path, keep=2)
    for i in range(4):
        _run(recorder, f"run{i}", "q")
    assert len(tracing.list_traces(tmp_path)) <= 3  # the one being written is kept as well


def test_summary_attributes_model_calls_and_tools_to_agents(tmp_path: Path) -> None:
    recorder = tracing.TraceRecorder(tmp_path)
    _run(recorder, "ccc333", "where?")
    summary = tracing.summarize(tracing.load(tracing.list_traces(tmp_path)[0]))
    assert summary.status == "completed" and summary.answer == "It is in src/app.py."
    spans = {s.agent: s for s in summary.spans}
    assert spans["router"].model_calls == 1 and spans["explorer"].tools == ["read_file"]
    assert spans["explorer"].tokens_in == 2000 and spans["responder"].model_calls == 1
    assert summary.models["big"]["calls"] == 2 and summary.slowest_models[0][1:] == ("big", "explorer")
    text = tracing.render(summary)
    assert "explorer" in text and "read_file src/app.py" in text and "where?" in text


def test_langsmith_only_with_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert tracing.enable_langsmith() is False
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.setattr(tracing.settings.tracing, "langsmith", "off")
    assert tracing.enable_langsmith() is False
    monkeypatch.setattr(tracing.settings.tracing, "langsmith", "auto")
    assert tracing.enable_langsmith() is True
    import os

    assert os.environ["LANGSMITH_TRACING"] == "true"
    monkeypatch.delenv("LANGSMITH_TRACING")
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
