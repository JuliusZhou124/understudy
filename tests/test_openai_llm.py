"""Tool-schema translation between our canonical shape and OpenAI's.

The project's canonical tool shape is Anthropic's (`input_schema`). OpenAI
nests the same JSON Schema under `function.parameters` and returns arguments
as a JSON *string*. That translation is the only real difference between the
two providers, so it lives in one tested function rather than in the caller.
"""

import json

import pytest

from understudy.llm import openai_tool_calls, to_openai_tools
from understudy.sim import BUYER_TOOLS


def test_translation_preserves_names_and_schemas():
    out = to_openai_tools(BUYER_TOOLS)
    assert len(out) == len(BUYER_TOOLS)
    for src, dst in zip(BUYER_TOOLS, out):
        assert dst["type"] == "function"
        assert dst["function"]["name"] == src["name"]
        assert dst["function"]["description"] == src["description"]
        assert dst["function"]["parameters"] == src["input_schema"]


def test_translation_of_no_tools_is_empty():
    assert to_openai_tools([]) == []


class _Fn:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _Call:
    def __init__(self, name, arguments):
        self.function = _Fn(name, arguments)


def test_arguments_are_parsed_from_json_strings():
    calls = openai_tool_calls([_Call("log_offer", json.dumps({"price": 480}))])
    assert calls[0].name == "log_offer"
    assert calls[0].args == {"price": 480}


def test_malformed_arguments_are_dropped_not_raised():
    """A model can emit invalid JSON; a negotiation must not die on it."""
    assert openai_tool_calls([_Call("log_offer", "{not json")]) == []


def test_missing_tool_calls_is_empty():
    assert openai_tool_calls(None) == []


@pytest.mark.parametrize("model,expected", [
    ("gpt-5.4", "max_completion_tokens"),
    ("gpt-5", "max_completion_tokens"),
    ("gpt-5.4-mini", "max_completion_tokens"),
    ("o3-mini", "max_completion_tokens"),
    ("openai/gpt-5.4", "max_completion_tokens"),   # gateway-prefixed slug
    ("gpt-4.1-mini", "max_tokens"),
    ("gpt-4o", "max_tokens"),
    ("llama3", "max_tokens"),
])
def test_token_limit_parameter_per_model(model, expected):
    """GPT-5.x rejects max_tokens outright; older models accept only it."""
    from understudy.llm import token_limit_param
    assert token_limit_param(model) == expected


def test_usage_accumulates_and_ignores_responses_without_it():
    from understudy.llm import Usage

    class U:
        prompt_tokens, completion_tokens = 100, 20

    class Resp:
        usage = U()

    u = Usage()
    u.add(Resp())
    u.add(Resp())
    u.add(object())          # no usage attribute — must not count or raise
    assert (u.calls, u.input_tokens, u.output_tokens) == (2, 200, 40)
    assert "2 calls" in str(u)


def test_usage_reads_anthropic_field_names():
    from understudy.llm import Usage

    class U:
        input_tokens, output_tokens = 50, 10

    class Resp:
        usage = U()

    u = Usage()
    u.add(Resp())
    assert (u.input_tokens, u.output_tokens) == (50, 10)
