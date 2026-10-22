"""Tool calls a model writes into the text of an answer.

The responder has no tools, yet asked to "run a command and tell me what it printed" the main model sometimes
calls one anyway, in whatever markup its chat template or training suggests. Seen live on Nemotron 3.5 Super
(2026-10-21): ``<|tool_calls_section_begin|><|tool_call_begin|>functions.exec:0…``, ``<tool_calls><tool_call
name="invoke" target="coder"><parameter name="task">…`` followed by an invented ``<tool_call_output>``,
``<tool_code><parameter name="command">…``, ``<execute_command><parameter=command>…``; Nemotron 3 Ultra writes
``[[{"name": …, "parameters": …}]]``. Nothing runs, so such a reply is no answer — and it may claim an output
nobody saw. :func:`has_tool_markup` spots one in a finished answer; :class:`MarkupFilter` keeps it out of the
streamed tokens. Code blocks and inline code are not searched, so an answer that explains such a format in code
is left alone.
"""

from __future__ import annotations

import re

from kartrix.llm.nim import parse_text_tool_calls

# Chat-template tokens (<|tool_calls_section_begin|>, <|python_tag|>, <|channel|>, <|im_start|>) never belong in
# an answer; the tags are the tool-call formats of common templates, <parameter …> the part most of them share.
_MARKERS = re.compile(
    r"<\|[\w.:-]{1,40}\|>"
    r"|</?(?:tool[_-]?(?:call|code|use|output|result)|function[_-]?(?:call|result)|function=|invoke\b"
    r"|execute_command\b|parameter\b)"
    r"|\[TOOL_CALLS\]",
    re.IGNORECASE,
)
_TOOL_FENCE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})[ \t]*(?:tool_code|tool_calls?)\b", re.MULTILINE | re.IGNORECASE)
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})[^\n]*(?:\n.*?^[ \t]*\1[ \t]*$|.*\Z)", re.MULTILINE | re.DOTALL)
_INLINE = re.compile(r"`[^`\n]*(?:`|$)", re.MULTILINE)  # unclosed: code to the end of the line (still streaming)
_JSON_START = re.compile(r"\s*(?:\{|\[\s*[\[{])")
_HOLD = 64  # an unfinished "<…" or "[…" this short may still become a marker


def _blank(match: re.Match[str]) -> str:
    return " " * len(match.group(0))


def markup_start(text: str) -> int | None:
    """Where the first tool-call marker outside code starts in ``text``, or None."""
    prose = _INLINE.sub(_blank, _FENCE.sub(_blank, text))  # same length: positions still match ``text``
    found = [m.start() for m in (_TOOL_FENCE.search(text), _MARKERS.search(prose)) if m]
    return min(found) if found else None


def is_tool_json(text: str) -> bool:
    """``text`` is nothing but tool calls as JSON (``[[{"name": "exec", "parameters": {…}}]]``), any tool name."""
    return bool(_JSON_START.match(text)) and parse_text_tool_calls(text, None) is not None


def has_tool_markup(text: str) -> bool:
    """``text`` calls a tool instead of being (only) an answer."""
    return markup_start(text) is not None or is_tool_json(text)


class MarkupFilter:
    """The streamed text of one model call without its tool-call markup: :meth:`feed` returns what can be shown
    now, :meth:`end` the rest. Text that might still turn into a marker is held back until it can be told apart;
    from the first marker on, nothing more of the call is shown. A reply that opens like JSON is held whole, since
    only the end shows whether it is a tool call."""

    def __init__(self) -> None:
        self.text = ""
        self.shown = 0
        self.blocked = False

    def feed(self, token: str) -> str:
        self.text += token
        if self.blocked:
            return ""
        start = markup_start(self.text)
        if start is not None:
            self.blocked = True
            return self._show(start)
        if _JSON_START.match(self.text):
            return ""
        return self._show(self._held_from())

    def end(self) -> str:
        if self.blocked or is_tool_json(self.text):
            self.blocked = True
            return ""
        return self._show(len(self.text))

    def _show(self, upto: int) -> str:
        upto = max(upto, self.shown)
        out, self.shown = self.text[self.shown : upto], upto
        return out

    def _held_from(self) -> int:
        text, held = self.text, len(self.text)
        for opener, closer in (("<", ">"), ("[", "]")):
            at = text.rfind(opener)
            if at != -1 and closer not in text[at:] and len(text) - at < _HOLD:
                held = min(held, at)
        line = text.rfind("\n") + 1
        start = text[line:].lstrip()
        if start and (start.startswith(("```", "~~~")) or set(start) in ({"`"}, {"~"})):  # a fence: tool_code?
            held = min(held, line)
        return held
