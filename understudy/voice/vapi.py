"""Vapi telephony.

Vapi is used for the same reason lowball used it: the interesting part of this
project is the negotiation policy, not building a Twilio-to-realtime-audio
bridge. Vapi dials, does speech-to-speech, and posts events back; this module
turns those events into the dashboard's vocabulary.

`create_call` cannot dial anything `voice.safety` has not approved — the
customer number comes from `resolve_call_number` and nowhere else.
"""

from __future__ import annotations

import json
import os
from typing import Any

from understudy.models import Listing, Packet
from understudy.sim import BUYER_TOOLS
from understudy.strategies import Strategy
from understudy.voice.assistant import voice_system_prompt
from understudy.voice.safety import resolve_call_number

VAPI_API = "https://api.vapi.ai/call"


def _args(raw: Any) -> dict:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw or {}


def handle_webhook(body: dict) -> list[dict]:
    """Translate one Vapi webhook payload into dashboard events."""
    message = (body or {}).get("message") or {}
    kind = message.get("type")

    if kind == "transcript":
        # Vapi's "assistant" is our buyer agent; its "user" is the seller.
        return [{
            "type": "transcript",
            "speaker": "buyer" if message.get("role") == "assistant" else "seller",
            "text": message.get("transcript", ""),
            "final": message.get("transcriptType") == "final",
        }]

    if kind == "tool-calls":
        events: list[dict] = []
        for call in message.get("toolCallList") or message.get("toolCalls") or []:
            name = call.get("name") or (call.get("function") or {}).get("name")
            args = _args(call.get("arguments") or (call.get("function") or {}).get("arguments"))
            if name == "log_offer" and "price" in args:
                events.append({"type": "offer", "price": float(args["price"])})
            elif name == "accept_offer" and "price" in args:
                events.append({"type": "deal", "price": float(args["price"])})
            elif name == "walk_away":
                events.append({"type": "call-ended", "outcome": "no_deal",
                               "reason": args.get("reason", "")})
        return events

    if kind == "status-update":
        return [{"type": "status", "status": message.get("status")}]

    if kind == "end-of-call-report":
        return [{"type": "report", "summary": message.get("summary"),
                 "ended_reason": message.get("endedReason")}]

    return []


def build_assistant_config(packet: Packet, strategy: Strategy) -> dict:
    return {
        "firstMessage": f"Hi — I'm calling about the {packet.headline}. Is it still available?",
        "model": {
            "provider": "anthropic",
            "model": os.environ.get("VOICE_MODEL", "claude-opus-5"),
            "messages": [{"role": "system", "content": voice_system_prompt(packet, strategy)}],
            "tools": [{"type": "function",
                       "function": {"name": t["name"], "description": t["description"],
                                    "parameters": t["input_schema"]}}
                      for t in BUYER_TOOLS],
        },
        "serverUrl": f"{os.environ['PUBLIC_URL']}/vapi-webhook" if os.environ.get("PUBLIC_URL") else None,
    }


def create_call(listing: Listing, packet: Packet, strategy: Strategy) -> dict:  # pragma: no cover - live
    """Place the call. Raises CallRefused before any network call if not permitted."""
    import httpx

    number = resolve_call_number(listing)  # the gate — must come first
    token = os.environ.get("VAPI_API_KEY")
    phone_number_id = os.environ.get("VAPI_PHONE_NUMBER_ID")
    if not token or not phone_number_id:
        raise RuntimeError("VAPI_API_KEY and VAPI_PHONE_NUMBER_ID must be set to place calls")

    payload = {
        "phoneNumberId": phone_number_id,
        "customer": {"number": number},
        "assistant": build_assistant_config(packet, strategy),
    }
    resp = httpx.post(VAPI_API, json=payload,
                      headers={"Authorization": f"Bearer {token}"}, timeout=30.0)
    resp.raise_for_status()
    return resp.json()
