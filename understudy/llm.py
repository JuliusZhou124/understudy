"""The LLM boundary.

Two implementations behind one protocol:

  * `StubLLM` — a deterministic rule-based negotiator that honours
    `PersonaParams` exactly. It is why the entire pipeline, test suite and
    demo run with **no API keys**, and it doubles as an executable
    specification of what "correct persona behaviour" means.
  * `AnthropicLLM` — Claude. Used for real runs and behind the voice agent.

Keeping the protocol this narrow (system + messages + tools in, text +
tool calls out) is what lets the simulator treat both interchangeably.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from understudy.models import PersonaParams

DEFAULT_CLAUDE_MODEL = "claude-opus-5"
# Deliberately read from the environment rather than pinned: OpenAI's catalogue
# moves faster than this repo will. `understudy models` lists what a key can
# actually reach.
FALLBACK_OPENAI_MODEL = "gpt-4o-mini"


def default_openai_model() -> str:
    """Resolved at call time, not import time.

    `or` rather than a default argument: `.env.example` ships a bare
    `OPENAI_MODEL=`, which yields "" rather than a missing key, and
    `os.environ.get(name, default)` does not fall back on an empty string.
    """
    return os.environ.get("OPENAI_MODEL") or FALLBACK_OPENAI_MODEL

Role = Literal["buyer", "seller"]


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    def complete(self, system: str, messages: list[Message],
                 tools: list[dict]) -> LLMResponse: ...


@dataclass
class Usage:
    """Running total for a process. A live sweep is thousands of calls, and an
    unlogged bill is a surprise waiting to happen."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, resp) -> None:
        u = getattr(resp, "usage", None)
        if u is None:
            return
        self.calls += 1
        self.input_tokens += getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", 0) or 0
        self.output_tokens += getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", 0) or 0

    def __str__(self) -> str:
        return (f"{self.calls} calls, {self.input_tokens:,} in / "
                f"{self.output_tokens:,} out tokens")


# Process-wide, so a sweep's total is available wherever it finishes.
USAGE = Usage()


_MONEY = re.compile(r"\$\s?([\d,]+(?:\.\d{2})?)")


def last_price(messages: list[Message]) -> float | None:
    """The most recent dollar figure **the other side** said.

    Only `role="user"` turns count. Scanning every message lets a negotiator
    read back its own last quote and treat it as the counterparty's offer —
    which makes a seller accept at its own asking price the moment the buyer
    says something without a number in it.
    """
    for m in reversed(messages):
        if m.role != "user":
            continue
        found = _MONEY.findall(m.content or "")
        if found:
            return float(found[-1].replace(",", ""))
    return None


class StubLLM:
    """Deterministic rule-based negotiator. No network, no keys."""

    WALK_AFTER_TURNS = 12

    def __init__(self, role: Role, *, ask_price: float,
                 params: PersonaParams | None = None,
                 target: float | None = None, seed: int = 0,
                 strategy=None):
        self.role = role
        self.ask_price = ask_price
        self.params = params
        self.target = target if target is not None else ask_price * 0.85
        self.strategy = strategy
        self.rng = random.Random(seed)
        self.turn = 0
        # Per-seed heterogeneity in how this particular buyer concedes. Without
        # it every run walks an identical price ladder, the settle becomes a
        # step function of the seller's floor, and the predicted settle
        # distribution collapses to a point mass (measured: $0.00 spread,
        # coverage@80 = 3%). Real buyers are not identical; neither are these.
        self._step_jitter = 1.0 + (self.rng.random() - 0.5) * 0.5   # +/-25%
        self._open_jitter = 1.0 + (self.rng.random() - 0.5) * 0.08  # +/-4%

        if role == "seller":
            self.current = ask_price
        elif strategy is not None:
            self.current = min(ask_price * strategy.opening_ratio * self._open_jitter, self.target)
        else:
            self.current = self.target * 0.9

    @property
    def floor(self) -> float:
        ratio = self.params.reservation_ratio if self.params else 0.8
        return self.ask_price * ratio

    def complete(self, system: str, messages: list[Message], tools: list[dict]) -> LLMResponse:
        self.turn += 1
        return self._seller(messages) if self.role == "seller" else self._buyer(messages)

    # -- seller -------------------------------------------------------------
    def _seller(self, messages: list[Message]) -> LLMResponse:
        p = self.params
        offer = last_price(messages)

        if offer is not None and offer >= self.floor - 1e-6:
            return LLMResponse(
                text=f"Alright — ${offer:.0f} works. Let's do it.",
                tool_calls=[ToolCall("accept", {"price": round(offer, 2)})],
            )

        if offer is not None and self.turn > 1 and self.turn <= p.patience:
            # Concede a share of the gap down to (but never through) the floor.
            gap = max(0.0, self.current - max(offer, self.floor))
            self.current = max(self.floor, self.current - gap * p.concession_rate)

        if self.turn == 1:
            text = f"Yeah, still available. It's listed at ${self.current:.0f}."
        elif self.turn > p.patience:
            text = f"${self.current:.0f} is where I'm at. Take it or leave it."
        else:
            text = f"I could do ${self.current:.0f}."

        return LLMResponse(text=text,
                           tool_calls=[ToolCall("quote_price", {"price": round(self.current, 2)})])

    # -- buyer --------------------------------------------------------------
    def _buyer(self, messages: list[Message]) -> LLMResponse:
        quoted = last_price(messages)
        calls: list[ToolCall] = []

        if quoted is not None:
            calls.append(ToolCall("log_offer", {"price": quoted}))
            if quoted <= self.target + 1e-6:
                calls.append(ToolCall("accept_offer", {"price": quoted}))
                return LLMResponse(text=f"${quoted:.0f} works for me — let's do it.",
                                   tool_calls=calls)

        walk_turns = self.strategy.walk_turns if self.strategy else self.WALK_AFTER_TURNS
        if self.turn >= walk_turns:
            calls.append(ToolCall("walk_away", {"reason": "seller would not reach my number"}))
            return LLMResponse(text="That's above what I can do. I'll pass — thanks for your time.",
                               tool_calls=calls)

        # deflect_first: refuse to put a number on the table on the opening turn.
        # No figure in the text means the seller has nothing to accept.
        if self.strategy is not None and self.strategy.deflect_first and self.turn == 1:
            return LLMResponse(
                text="You know this card better than I do — where can you be on price?",
                tool_calls=calls)

        if quoted is not None:
            # Close a fraction of the distance — but never bid past our own
            # ceiling. Without this cap the buyer walks itself above its
            # walk-away and the seller happily accepts.
            # Diminishing concessions: each step is a fraction of the last.
            # A constant step makes the buyer leap from below the seller's floor
            # to well above it, so the settle lands half a step high every time
            # (measured bias: +$24.77). Finer steps near the close remove it,
            # and matches how people actually concede.
            base = (self.strategy.concession_step if self.strategy else
                    (0.5 if self.turn == 1 else 0.25))
            step = max(0.03, base * self._step_jitter * (0.7 ** (self.turn - 1)))
            self.current = min(quoted, self.current + (quoted - self.current) * step)
            self.current = min(self.current, self.target)

        return LLMResponse(text=f"I could do ${self.current:.0f} today, cash.", tool_calls=calls)


# GPT-5.x and the o-series reject `max_tokens` outright and require
# `max_completion_tokens`; older models accept only the former.
_REASONING_MODEL = re.compile(r"^(?:gpt-5|o\d)", re.I)
_OTHER_TOKEN_PARAM = {"max_tokens": "max_completion_tokens",
                      "max_completion_tokens": "max_tokens"}


def token_limit_param(model: str) -> str:
    """Which output-limit parameter this model accepts."""
    name = model.split("/")[-1]  # gateways prefix with 'provider/'
    return "max_completion_tokens" if _REASONING_MODEL.match(name) else "max_tokens"


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Canonical (Anthropic-shaped) tool defs -> OpenAI's nested form."""
    return [
        {"type": "function",
         "function": {"name": t["name"], "description": t["description"],
                      "parameters": t["input_schema"]}}
        for t in tools
    ]


def openai_tool_calls(raw) -> list[ToolCall]:
    """OpenAI tool calls -> ToolCall. Arguments arrive as a JSON string.

    Malformed arguments are dropped rather than raised: a live model can emit
    invalid JSON, and one bad call should not end a negotiation.
    """
    import json

    out: list[ToolCall] = []
    for call in raw or []:
        try:
            out.append(ToolCall(call.function.name, json.loads(call.function.arguments or "{}")))
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    return out


class OpenAILLM:  # pragma: no cover - requires network + key
    """OpenAI-compatible negotiator.

    `base_url` makes this provider-agnostic: the same class reaches OpenAI,
    a local Ollama, OpenRouter, Together, or any OpenAI-compatible gateway by
    changing one environment variable, with no code change here.
    """

    def __init__(self, model: str | None = None, max_tokens: int = 512,
                 base_url: str | None = None, needs_text: bool = True):
        import openai

        self.model = model or default_openai_model()
        self.max_tokens = max_tokens
        # A negotiator must speak, so a tool-only reply needs a follow-up call.
        # A judge only ever reports through a tool, and that follow-up would
        # double its cost for output nobody reads.
        self.needs_text = needs_text
        self.client = openai.OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None)
        self._token_param = token_limit_param(self.model)

    def complete(self, system: str, messages: list[Message], tools: list[dict]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "tools": to_openai_tools(tools) or None,
        }
        try:
            resp = self.client.chat.completions.create(
                **payload, **{self._token_param: self.max_tokens})
        except Exception as e:
            # The name of the token-limit parameter is the one thing that
            # reliably differs across OpenAI-compatible endpoints. Swap once
            # and remember, rather than failing a whole sweep on it.
            other = _OTHER_TOKEN_PARAM[self._token_param]
            if self._token_param not in str(e):
                raise
            self._token_param = other
            resp = self.client.chat.completions.create(**payload, **{other: self.max_tokens})

        USAGE.add(resp)
        msg = resp.choices[0].message
        raw_calls = getattr(msg, "tool_calls", None) or []
        calls = openai_tool_calls(raw_calls)
        text = (msg.content or "").strip()

        # A model that calls a tool usually returns no content alongside it.
        # On a phone call that means the agent logs the offer and then says
        # nothing. Close the loop: hand back the tool results and ask for the
        # spoken turn. One follow-up only — this is a conversation, not an
        # agent loop, and the tools here are all fire-and-forget.
        if raw_calls and not text and self.needs_text:
            payload["messages"] = payload["messages"] + [
                msg.model_dump(exclude_none=True),
                *[{"role": "tool", "tool_call_id": c.id, "content": "ok"} for c in raw_calls],
            ]
            follow = self.client.chat.completions.create(
                **{**payload, "tools": None}, **{self._token_param: self.max_tokens})
            USAGE.add(follow)
            text = (follow.choices[0].message.content or "").strip()

        return LLMResponse(text=text, tool_calls=calls)


class AnthropicLLM:  # pragma: no cover - requires network + key
    """Claude-backed negotiator.

    Effort is pinned low and thinking left adaptive: negotiation turns are one
    or two spoken sentences, and on a voice call latency is the constraint.
    """

    def __init__(self, model: str = DEFAULT_CLAUDE_MODEL, max_tokens: int = 512,
                 effort: str = "low"):
        import anthropic

        # Honours ANTHROPIC_BASE_URL, so this also routes through an
        # Anthropic-compatible gateway without a code change.
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort

    def complete(self, system: str, messages: list[Message], tools: list[dict]) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools or [],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        USAGE.add(resp)
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(block.name, dict(block.input)))
        return LLMResponse(text=" ".join(text_parts).strip(), tool_calls=calls)
