"""Small wrapper over the Anthropic Python SDK for the research, plan and report stages.

The API key is read only from the RESYNTH_ANTHROPIC_KEY environment variable and passed to the
client explicitly. It is never logged or written to any file.

Two call shapes:
- `search_notes`: one request with the server-side web search tool. Returns Claude's notes and
  every URL the search actually returned (used to check the URLs Claude later cites).
- `ask_json`: one request with structured outputs (a JSON schema), no tools. Web-search answers
  carry citations, which structured outputs do not accept, so research is two calls: search,
  then structure the notes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

KEY_ENV = "RESYNTH_ANTHROPIC_KEY"
MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # on a policy refusal the API retries on a fallback model
WEB_SEARCH_TOOL = "web_search_20260209"
MAX_CONTINUATIONS = 3


class ClaudeUnavailable(RuntimeError):
    """No key, no SDK, or the API call failed. The message is safe to show to the user."""


def key_present() -> bool:
    return bool(os.environ.get(KEY_ENV, "").strip())


def get_client() -> Any:
    """An Anthropic client using RESYNTH_ANTHROPIC_KEY, or ClaudeUnavailable explaining why not."""
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise ClaudeUnavailable(f"no API key: the {KEY_ENV} environment variable is not set")
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover - dependency is in pyproject
        raise ClaudeUnavailable("the anthropic package is not installed") from e
    return anthropic.Anthropic(api_key=key, max_retries=2, timeout=600.0)


def _describe_error(e: Exception) -> str:
    """A short user-safe description of an SDK error (never includes the key)."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return type(e).__name__
    if isinstance(e, anthropic.AuthenticationError):
        return f"the API rejected the key in {KEY_ENV} (authentication error)"
    if isinstance(e, anthropic.PermissionDeniedError):
        return "the API key lacks permission for this request"
    if isinstance(e, anthropic.RateLimitError):
        return "rate limited by the API; try again in a minute"
    if isinstance(e, anthropic.BadRequestError):
        return f"the API rejected the request: {getattr(e, 'message', e)}"
    if isinstance(e, anthropic.APIStatusError):
        return f"API error {e.status_code}"
    if isinstance(e, anthropic.APIConnectionError):
        return "could not reach the Anthropic API (network)"
    return f"{type(e).__name__}: {e}"


def _create(client: Any, **kw: Any) -> Any:
    try:
        return client.beta.messages.create(model=MODEL, betas=[FALLBACK_BETA], fallbacks="default", **kw)
    except ClaudeUnavailable:
        raise
    except Exception as e:
        raise ClaudeUnavailable(_describe_error(e)) from e


def _check_stop(resp: Any) -> None:
    if getattr(resp, "stop_reason", None) == "refusal":
        details = getattr(resp, "stop_details", None)
        cat = getattr(details, "category", None) if details else None
        raise ClaudeUnavailable(f"Claude declined the request{f' ({cat})' if cat else ''}")


@dataclass
class SearchNotes:
    text: str
    urls: dict[str, str] = field(default_factory=dict)  # url -> title, from search results and citations
    searches: int = 0
    usage: dict[str, int] = field(default_factory=dict)


def _add_usage(total: dict[str, int], resp: Any) -> None:
    u = getattr(resp, "usage", None)
    for k in ("input_tokens", "output_tokens"):
        v = getattr(u, k, None) if u is not None else None
        if isinstance(v, int):
            total[k] = total.get(k, 0) + v


def search_notes(client: Any, system: str, prompt: str, max_searches: int = 6, max_tokens: int = 16000,
                 effort: str = "medium") -> SearchNotes:
    """Let Claude research with web search; collect its final text and the URLs it saw."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tools = [{"type": WEB_SEARCH_TOOL, "name": "web_search", "max_uses": max_searches}]
    out = SearchNotes(text="")
    texts: list[str] = []
    for _ in range(MAX_CONTINUATIONS + 1):
        resp = _create(client, max_tokens=max_tokens, system=system, messages=messages, tools=tools,
                       output_config={"effort": effort})
        _check_stop(resp)
        _add_usage(out.usage, resp)
        for block in resp.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                texts.append(block.text)
                for c in getattr(block, "citations", None) or []:
                    url = getattr(c, "url", None)
                    if url:
                        out.urls.setdefault(url, getattr(c, "title", "") or "")
            elif btype == "server_tool_use":
                out.searches += 1
            elif btype == "web_search_tool_result":
                content = getattr(block, "content", None)
                if isinstance(content, list):  # an error result is a single object, not a list
                    for r in content:
                        url = getattr(r, "url", None)
                        if url:
                            out.urls.setdefault(url, getattr(r, "title", "") or "")
        if resp.stop_reason != "pause_turn":
            break
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": resp.content}]
    out.text = "\n".join(t for t in texts if t.strip())
    return out


def ask_json(client: Any, system: str, prompt: str, schema: dict[str, Any], max_tokens: int = 16000,
             effort: str = "medium") -> dict[str, Any]:
    """One structured-output request; returns the parsed JSON object."""
    resp = _create(client, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": prompt}],
                   output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}})
    _check_stop(resp)
    if resp.stop_reason == "max_tokens":
        raise ClaudeUnavailable("Claude's answer was cut off (max_tokens)")
    text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ClaudeUnavailable("Claude returned malformed JSON") from e
    if not isinstance(data, dict):
        raise ClaudeUnavailable("Claude returned JSON that is not an object")
    return data
