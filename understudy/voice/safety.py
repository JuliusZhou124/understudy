"""The call-permission gate.

Every outbound call in this project goes through `resolve_call_number`, and it
is the only function that may produce a number to dial. It ignores whatever
phone the listing data carries and returns the operator's own demo number.

The rules, in order:
  1. Calling is off unless CALLS_ENABLED=true.
  2. Private individuals are never dialable — cold AI-voice calls to consumers
     are regulated (TCPA) and this project does not make them. Only sellers
     positively identified as businesses may be called at all.
  3. The number dialled is always DEMO_CALL_NUMBER, never the seller's.

Inherited from lowball's `negotiate.ts`, which pinned its demo calls the same
way for the same reasons.
"""

from __future__ import annotations

import os
import re

from understudy.models import Listing

_E164 = re.compile(r"^\+[1-9]\d{7,14}$")


class CallRefused(RuntimeError):
    """Raised whenever an outbound call is not permitted."""


def calls_enabled() -> bool:
    return os.environ.get("CALLS_ENABLED") == "true"


def resolve_call_number(listing: Listing) -> str:
    if not calls_enabled():
        raise CallRefused("Outbound calling is disabled. Set CALLS_ENABLED=true to enable it.")
    if listing.seller_type != "business":
        raise CallRefused(
            f"Refusing to call a '{listing.seller_type}' seller — only sellers positively "
            "identified as businesses may be called, never private individuals."
        )
    demo = os.environ.get("DEMO_CALL_NUMBER", "")
    if not demo:
        raise CallRefused("DEMO_CALL_NUMBER is not set — refusing to dial anything.")
    if not _E164.match(demo):
        raise CallRefused(f"DEMO_CALL_NUMBER {demo!r} is not E.164 (e.g. +15551234567).")
    return demo
