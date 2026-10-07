"""Validation of request bodies."""

from __future__ import annotations

from typing import Any

MAX_TITLE = 200


class ValidationError(Exception):
    """The request body is invalid; the message is shown to the client."""


def _check_tags(tags: Any) -> list[str]:
    if not isinstance(tags, list) or not all(isinstance(t, str) and t for t in tags):
        raise ValidationError("tags must be a list of non-empty strings")
    if any("," in t for t in tags):
        raise ValidationError("tags must not contain commas")
    return tags


def validate_new_todo(body: Any) -> tuple[str, list[str]]:
    """Title and tags of a todo to create."""
    if not isinstance(body, dict):
        raise ValidationError("body must be a JSON object")
    title = body.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValidationError("title must not be empty")
    if len(title) > MAX_TITLE:
        raise ValidationError(f"title must be at most {MAX_TITLE} characters")
    return title.strip(), _check_tags(body.get("tags", []))


def validate_update(body: Any) -> dict[str, Any]:
    """The fields of a PATCH body."""
    if not isinstance(body, dict) or not body:
        raise ValidationError("body must be a non-empty JSON object")
    unknown = set(body) - {"title", "done", "tags"}
    if unknown:
        raise ValidationError(f"unknown fields: {', '.join(sorted(unknown))}")
    fields: dict[str, Any] = {}
    if "title" in body:
        if not isinstance(body["title"], str) or not body["title"].strip():
            raise ValidationError("title must not be empty")
        fields["title"] = body["title"].strip()
    if "done" in body:
        if not isinstance(body["done"], bool):
            raise ValidationError("done must be true or false")
        fields["done"] = body["done"]
    if "tags" in body:
        fields["tags"] = _check_tags(body["tags"])
    return fields
