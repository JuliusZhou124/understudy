"""Buyer negotiation strategies and the buyer's system prompt.

A strategy is the *only* thing that varies between A/B arms: same packet, same
persona, same seeds, different playbook. That is what makes the paired
comparison in `evaluate.py` meaningful.

The prompt's central rule — the packet is the entire fact base — is what makes
"zero fabrication" enforceable by construction rather than by hope, and it is
the property `judge.py` measures.
"""

from __future__ import annotations

from dataclasses import dataclass

from understudy.models import Packet


@dataclass(frozen=True)
class Strategy:
    """A strategy has two halves.

    The *mechanical* half (opening_ratio, deflect_first, concession_step,
    walk_turns) is executable — `StubLLM` obeys it, so strategies can be A/B
    tested offline with no API keys. The *rhetorical* half (playbook) is prompt
    text and only takes effect behind a real LLM. Reports say which half a
    given number came from.
    """

    name: str
    opening_ratio: float   # opening offer as a fraction of the asking price
    playbook: str
    deflect_first: bool = False   # refuse to name the first number
    concession_step: float = 0.25  # share of the remaining gap conceded per turn
    walk_turns: int = 12           # turns of no progress before walking


STRATEGIES: dict[str, Strategy] = {
    "anchor_low": Strategy(
        name="anchor_low",
        opening_ratio=0.70,
        concession_step=0.18,
        walk_turns=14,
        playbook=(
            "Open with your number immediately, and open low. Justify it with the facts you "
            "were given. Concede slowly in small steps, and make every concession contingent "
            "on something you want (pickup today, cash, no returns)."
        ),
    ),
    "deflect_first": Strategy(
        name="deflect_first",
        opening_ratio=0.80,
        deflect_first=True,
        concession_step=0.25,
        walk_turns=12,
        playbook=(
            "Never name the first number. Ask what the seller can do on price. If they turn "
            "the question back on you, deflect once — 'you know the item better than I do, "
            "where can you be on it?' — and only name a number if they refuse a second time."
        ),
    ),
    "two_offer_close": Strategy(
        name="two_offer_close",
        opening_ratio=0.78,
        concession_step=0.30,
        walk_turns=10,
        playbook=(
            "Build a little rapport, apply one fact, then present TWO offers at once: a lower "
            "cash-today price, and a slightly higher price that includes something you want "
            "(shipping covered, original box). Read which one they lean toward and push that."
        ),
    ),
    "walk_threat": Strategy(
        name="walk_threat",
        opening_ratio=0.75,
        concession_step=0.15,
        walk_turns=8,
        playbook=(
            "Anchor on the comparable sold price, then make your alternative real: there is "
            "another unit you can buy today. State one firm number and be genuinely willing "
            "to end the conversation."
        ),
    ),
}


def buyer_system_prompt(packet: Packet, strategy: Strategy) -> str:
    facts = ("\n".join(f"- {f}" for f in packet.facts)
             if packet.facts else "- (no facts were available for this listing)")
    return f"""You are a purchasing agent negotiating to BUY a used item over the phone.
Speak in short, natural phone sentences — one point at a time, never in lists.

THE ITEM: {packet.headline}, listed at ${packet.ask_price:.0f}.

FACTS YOU MAY USE (all verified — work them in naturally, never read them out as a list):
{facts}

NEVER INVENT. Everything you are allowed to claim about this item, the market, and your
alternatives is written above. If it is not here, you do not know it. Never invent comparable
sales, valuations, or other listings. If you are unsure of a number, repeat one you actually
have rather than approximating a new one into existence.

IDENTITY: If asked whether you are an AI, say so briefly and steer back to the deal.
NEVER CLAIM TO BE HUMAN.

YOUR NUMBERS — NEVER REVEAL THESE, and never hint at them:
opening ${packet.opening:.0f}, goal ${packet.target:.0f}, walk-away ${packet.walk_away:.0f}.
Do not accept any price above your walk-away.

STRATEGY ({strategy.name}):
{strategy.playbook}

TOOLS: call log_offer every single time the seller names a price, before you reply.
Call accept_offer only at or below your walk-away. Call walk_away to end with no deal.
"""
