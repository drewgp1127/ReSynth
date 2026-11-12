"""A stub for the Anthropic client: records requests and replays canned responses.

Mimics only what app/stages/claude.py reads: client.beta.messages.create(**kw) returning an
object with .content (blocks with .type and fields), .stop_reason and .usage.
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS
from typing import Any, Callable


def text_block(text: str, citations: list[dict[str, str]] | None = None) -> NS:
    return NS(type="text", text=text, citations=[NS(**c) for c in citations or []])


def search_result_block(results: list[dict[str, str]]) -> NS:
    return NS(type="web_search_tool_result", content=[NS(type="web_search_result", **r) for r in results])


def response(blocks: list[NS], stop_reason: str = "end_turn") -> NS:
    return NS(content=blocks, stop_reason=stop_reason, stop_details=None,
              usage=NS(input_tokens=100, output_tokens=50))


def json_response(data: dict[str, Any]) -> NS:
    return response([text_block(json.dumps(data))])


class FakeClient:
    """`responder(kwargs) -> response` decides each reply; every request is kept in .calls."""

    def __init__(self, responder: Callable[[dict[str, Any]], NS]):
        self.calls: list[dict[str, Any]] = []
        outer = self

        class _Messages:
            def create(self, **kw: Any) -> NS:
                outer.calls.append(kw)
                return responder(kw)

        self.beta = NS(messages=_Messages())


class FailingClient(FakeClient):
    def __init__(self, exc: Exception):
        def boom(_kw: dict[str, Any]) -> NS:
            raise exc
        super().__init__(boom)
