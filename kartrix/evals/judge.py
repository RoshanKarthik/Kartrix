"""LLM-as-judge through DeepEval, with Kartrix's own judge model (``llm.judge_model``).

DeepEval supplies the metric algorithms (claim extraction, verdicts, scoring); :class:`KartrixJudge`
plugs our provider layer into it, so judging uses the same keys, retries and provider as the rest of
Kartrix and no other service is contacted: DeepEval's telemetry, ``.env`` autoloading and its
home/cache folders are switched off or moved into the eval cache before it is imported.

A judge is only trusted as far as it agrees with people: :func:`calibrate` scores the hand-labelled
items of ``evals/rag/calibration.yaml`` and reports agreement and Cohen's kappa per metric.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from kartrix.config import settings
from kartrix.evals.fixtures import cache_dir
from kartrix.llm.retry import acall_with_retry
from kartrix.observability.logger import get_logger

logger = get_logger(__name__)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def configure_deepeval() -> None:
    """Environment for DeepEval — call before importing it."""
    base = cache_dir()
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")  # Kartrix already loaded its .env
    os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_IN", "0")
    os.environ.setdefault("DEEPEVAL_HOME", str(base / "deepeval-home"))
    os.environ.setdefault("DEEPEVAL_CACHE_FOLDER", str(base / "deepeval-cache"))


configure_deepeval()

from deepeval.models import DeepEvalBaseLLM  # noqa: E402 — needs the environment above


@dataclass
class ModelUsage:
    """Tokens and time spent by one model while evaluating (not by the agent being measured)."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    failures: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "seconds": round(self.seconds, 2),
            "failures": self.failures,
        }


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else str(b.get("text", "")) for b in content if isinstance(b, str | dict)
        )
    return str(content)


def parse_json_reply(text: str, schema: type[BaseModel]) -> BaseModel | None:
    """The first JSON object in ``text`` validated against ``schema`` (None if there is none)."""
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return schema.model_validate(json.loads(cleaned[start : end + 1]))
    except (json.JSONDecodeError, ValidationError):
        return None


class KartrixJudge(DeepEvalBaseLLM):
    """A DeepEval evaluation model backed by :func:`kartrix.llm.factory.get_chat_model`.

    Structured output: the reply's JSON is parsed into DeepEval's schema; one retry with the schema
    spelled out if it doesn't parse. Calls are limited to ``concurrency`` at a time (free tiers allow
    few parallel requests) and retried on transient errors with the configured backoff."""

    def __init__(self, role: str = "judge", concurrency: int = 4, chat_model: Any = None) -> None:
        self.role = role
        self._chat_model = chat_model
        self.usage = ModelUsage()
        self._semaphore = asyncio.Semaphore(concurrency)
        name = settings.llm.effective_judge_model if role == "judge" else settings.llm.model
        super().__init__(model=name)

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        if self._chat_model is not None:
            return self._chat_model
        from kartrix.llm.factory import get_chat_model

        return get_chat_model(self.role, temperature=0)  # type: ignore[arg-type]

    @property
    def chat_model(self) -> Any:
        return self.model  # the LangChain chat model load_model() returned

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return f"{settings.llm.provider}:{self.name}"

    async def _ainvoke(self, prompt: str) -> str:
        async with self._semaphore:
            start = time.perf_counter()
            try:
                reply = await acall_with_retry(
                    lambda: self.chat_model.ainvoke(prompt), settings.llm.retry, "judge call"
                )
            except Exception:
                self.usage.failures += 1
                raise
            finally:
                self.usage.seconds += time.perf_counter() - start
        self.usage.calls += 1
        meta = getattr(reply, "usage_metadata", None) or {}
        self.usage.input_tokens += int(meta.get("input_tokens") or 0)
        self.usage.output_tokens += int(meta.get("output_tokens") or 0)
        return _text(reply.content)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: Any) -> Any:
        text = await self._ainvoke(prompt)
        if schema is None:
            return text
        parsed = parse_json_reply(text, schema)
        if parsed is not None:
            return parsed
        retry = (
            f"{prompt}\n\nReply with one JSON object only (no prose, no code fence) that matches this JSON "
            f"schema:\n{json.dumps(schema.model_json_schema())}"
        )
        text = await self._ainvoke(retry)
        return parse_json_reply(text, schema) or text  # DeepEval reports the metric error if it is still invalid

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: Any) -> Any:
        return asyncio.run(self.a_generate(prompt, schema))


# ── metrics ───────────────────────────────────────────────────────────

RAG_JUDGE_METRICS = ("faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall")


def rag_metrics(judge: KartrixJudge, names: tuple[str, ...] = RAG_JUDGE_METRICS) -> dict[str, Any]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )

    factories = {
        "faithfulness": FaithfulnessMetric,
        "answer_relevancy": AnswerRelevancyMetric,
        "contextual_precision": ContextualPrecisionMetric,
        "contextual_recall": ContextualRecallMetric,
    }
    return {n: factories[n](model=judge, threshold=0.5, include_reason=True, async_mode=True) for n in names}


async def measure(metric: Any, test_case: Any) -> dict[str, Any]:
    """Score one test case; a judge failure is recorded, not raised."""
    try:
        await metric.a_measure(test_case, _show_indicator=False)
    except Exception as e:
        logger.warning("Judge metric failed", extra={"metric": type(metric).__name__, "error": repr(e)[:300]})
        return {"score": None, "reason": None, "error": f"{type(e).__name__}: {e}"[:500]}
    return {"score": metric.score, "reason": (metric.reason or "")[:1000], "error": None}


def answer_correctness_metric(judge: KartrixJudge) -> Any:
    """G-Eval: does the agent's answer agree with the reference answer?"""
    from deepeval.metrics import GEval
    from deepeval.test_case import SingleTurnParams

    return GEval(
        name="Answer correctness",
        criteria=(
            "Judge whether the actual output answers the input correctly, using the expected output as the "
            "reference. Facts in the actual output that contradict the reference are errors; extra correct "
            "detail is fine; missing key facts from the reference lower the score."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=judge,
        threshold=0.5,
        async_mode=True,
    )


def plan_quality_metric(judge: KartrixJudge) -> Any:
    """G-Eval over the plan a /plan run proposed (DeepEval's own plan metric needs its tracing)."""
    from deepeval.metrics import GEval
    from deepeval.test_case import SingleTurnParams

    return GEval(
        name="Plan quality",
        criteria=(
            "The input is a coding goal for an existing small repository; the actual output is the plan an agent "
            "made for it (JSON). A good plan covers everything the goal asks for, breaks it into small ordered "
            "steps with clear dependencies, includes writing or running tests to verify the change, and adds no "
            "unrelated work."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=0.5,
        async_mode=True,
    )


# ── calibration ───────────────────────────────────────────────────────


async def calibrate(judge: KartrixJudge, threshold: float = 0.5) -> dict[str, Any]:
    """Run the faithfulness and answer-relevancy judges on the hand-labelled items and compare."""
    from deepeval.test_case import LLMTestCase

    from kartrix.evals.datasets import load_calibration
    from kartrix.evals.metrics import cohen_kappa

    items = load_calibration()

    async def one(item: Any) -> dict[str, Any]:
        case = LLMTestCase(input=item.question, actual_output=item.answer, retrieval_context=item.context)
        metrics = rag_metrics(judge, ("faithfulness", "answer_relevancy"))
        faithful, relevant = await asyncio.gather(
            measure(metrics["faithfulness"], case), measure(metrics["answer_relevancy"], case)
        )
        return {
            "id": item.id,
            "labels": {"faithfulness": item.faithful, "answer_relevancy": item.relevant},
            "scores": {"faithfulness": faithful, "answer_relevancy": relevant},
        }

    rows = await asyncio.gather(*(one(i) for i in items))
    summary: dict[str, Any] = {}
    for name in ("faithfulness", "answer_relevancy"):
        pairs = [
            (r["labels"][name], r["scores"][name]["score"] >= threshold)
            for r in rows
            if r["scores"][name]["score"] is not None
        ]
        labels = [a for a, _ in pairs]
        preds = [b for _, b in pairs]
        n = len(pairs)
        summary[name] = {
            "items": n,
            "errors": len(rows) - n,
            "agreement": (sum(a == b for a, b in pairs) / n) if n else None,
            "kappa": cohen_kappa(labels, preds),
            "false_pass": sum(1 for a, b in pairs if b and not a),  # judge passed what people failed
            "false_fail": sum(1 for a, b in pairs if a and not b),
        }
    return {"threshold": threshold, "summary": summary, "items": rows}
