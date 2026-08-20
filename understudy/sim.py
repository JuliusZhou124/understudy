"""The negotiation simulator.

A buyer LLM and a seller LLM alternate turns until someone accepts, someone
walks, or the turn cap is hit. Both sides are held to a tool contract — every
price named must come through a tool call — so the outcome is read from
structured data rather than parsed out of prose.

Because both sides are behind the `LLM` protocol, the same loop runs entirely
offline with `StubLLM` (fast, deterministic, free) and against Claude
(realistic, slow, costs money). Every test uses the former.
"""

from __future__ import annotations

from datetime import date

from understudy.llm import LLM, Message
from understudy.models import (
    Listing,
    Packet,
    PersonaParams,
    SimResult,
    SkuStats,
    TranscriptTurn,
)
from understudy.persona import persona_system_prompt, sample_persona
from understudy.strategies import Strategy, buyer_system_prompt

MAX_TURNS = 24


def _tool(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": props, "required": required},
    }


BUYER_TOOLS = [
    _tool("log_offer",
          "Log a price the seller just named. Call this the moment any price is said.",
          {"price": {"type": "number", "description": "Price in USD the seller named"}},
          ["price"]),
    _tool("accept_offer", "Accept the deal at this price.",
          {"price": {"type": "number", "description": "Agreed price in USD"}},
          ["price"]),
    _tool("walk_away", "End the negotiation with no deal.",
          {"reason": {"type": "string", "description": "Why you are walking"}},
          ["reason"]),
]

SELLER_TOOLS = [
    _tool("quote_price", "Name the price you are willing to sell at right now.",
          {"price": {"type": "number", "description": "Your price in USD"}},
          ["price"]),
    _tool("accept", "Accept the buyer's offer at this price.",
          {"price": {"type": "number", "description": "Agreed price in USD"}},
          ["price"]),
    _tool("end_conversation", "Hang up with no deal.",
          {"reason": {"type": "string", "description": "Why you are ending it"}},
          ["reason"]),
]


def _fallback_seller_prompt(packet: Packet, persona: PersonaParams) -> str:
    floor = packet.ask_price * persona.reservation_ratio
    return (f"You are selling {packet.headline}, listed at ${packet.ask_price:.0f}. "
            f"Your hidden floor is ${floor:.0f} — NEVER REVEAL it. Speak in short phone "
            f"sentences. Call quote_price every time you name a price.")


def negotiate(
    packet: Packet,
    persona: PersonaParams,
    buyer_llm: LLM,
    seller_llm: LLM,
    strategy: Strategy,
    seed: int,
    stats: SkuStats | None = None,
    listing: Listing | None = None,
    max_turns: int = MAX_TURNS,
) -> SimResult:
    buyer_system = buyer_system_prompt(packet, strategy)
    seller_system = (
        persona_system_prompt(listing, persona, stats)
        if listing is not None and stats is not None
        else _fallback_seller_prompt(packet, persona)
    )

    transcript: list[TranscriptTurn] = []
    buyer_msgs: list[Message] = []
    seller_msgs: list[Message] = []
    offers: list[float] = []
    outcome: str = "timeout"
    final_price: float | None = None
    turns = 0

    opening_line = f"Hi — I'm calling about the {packet.headline}. Is it still available?"
    transcript.append(TranscriptTurn(speaker="buyer", text=opening_line))
    seller_msgs.append(Message(role="user", content=opening_line))

    for turns in range(1, max_turns + 1):
        # --- seller ---
        s = seller_llm.complete(seller_system, seller_msgs, SELLER_TOOLS)
        for tc in s.tool_calls:
            if tc.name == "accept":
                outcome, final_price = "deal", float(tc.args["price"])
            elif tc.name == "end_conversation":
                outcome = "no_deal"
        seller_text = s.text or "(silence)"
        transcript.append(TranscriptTurn(speaker="seller", text=seller_text))
        seller_msgs.append(Message(role="assistant", content=seller_text))
        buyer_msgs.append(Message(role="user", content=seller_text))
        if outcome in {"deal", "no_deal"}:
            break

        # --- buyer ---
        b = buyer_llm.complete(buyer_system, buyer_msgs, BUYER_TOOLS)
        for tc in b.tool_calls:
            if tc.name == "log_offer":
                offers.append(float(tc.args["price"]))
            elif tc.name == "accept_offer":
                outcome, final_price = "deal", float(tc.args["price"])
            elif tc.name == "walk_away":
                outcome = "walk"
        buyer_text = b.text or "(silence)"
        transcript.append(TranscriptTurn(speaker="buyer", text=buyer_text))
        buyer_msgs.append(Message(role="assistant", content=buyer_text))
        seller_msgs.append(Message(role="user", content=buyer_text))
        if outcome in {"deal", "walk"}:
            break

    return SimResult(
        listing_id=packet.listing_id, strategy=strategy.name, seed=seed,
        outcome=outcome, final_price=final_price, offers=offers,
        turns=turns, transcript=transcript, persona=persona,
    )


def simulate_listing(
    listing: Listing,
    stats: SkuStats,
    model,
    strategy: Strategy,
    llm_factory,
    seed: int = 0,
    n: int = 1,
    today: date | None = None,
) -> list[SimResult]:
    """Run `n` seeded simulations of one listing.

    `llm_factory(role, ask_price, persona, target, seed, strategy) -> LLM` —
    the single signature used by every caller (recommend, calibrate, ab_test,
    the CLI).

    `today` is the as-of date for time-dependent features. Leave it None for a
    live listing (age so far). Calibration MUST pass the listing's age at sale,
    or `days_listed` is 0 at predict time and 60 at fit time — a train/serve
    skew that biased predicted settles by +$24 before it was caught.
    """
    from understudy.decide import build_packet

    packet = build_packet(listing, stats, model, today=today)
    results: list[SimResult] = []
    for i in range(n):
        s = seed + i
        persona = sample_persona(listing, stats, model, seed=s, today=today)
        results.append(negotiate(
            packet, persona,
            buyer_llm=llm_factory("buyer", packet.ask_price, None, packet.walk_away, s, strategy),
            seller_llm=llm_factory("seller", packet.ask_price, persona, None, s, strategy),
            strategy=strategy, seed=s, stats=stats, listing=listing,
        ))
    return results
