"""Negotiability: which listings are worth calling about, and why.

Two outputs, deliberately separate:

  * `negotiability_score` orders the list.
  * `why_reasons` explains that order in the seller's own observable terms.

The reasons are terse UI chips, not the agent's briefing — the packet in
`decide.py` remains the agent's only fact source. Both draw on the same
observations, so the interface never claims something the agent could not.

Mirrors lowball's `server/src/rank.ts`.
"""

from __future__ import annotations

from datetime import date

from understudy.models import Listing, SkuStats

MIN_GAP_USD = 10.0
MIN_DAYS = 7


def why_reasons(listing: Listing, stats: SkuStats, today: date | None = None) -> list[str]:
    reasons: list[str] = []

    gap = listing.ask_price - stats.median
    if abs(gap) >= MIN_GAP_USD:
        reasons.append(f"${abs(gap):,.0f} {'over' if gap > 0 else 'under'} median")

    if listing.price_cuts:
        reasons.append(f"{listing.price_cuts} price cut{'s' if listing.price_cuts > 1 else ''}")

    days = listing.days_listed(today)
    if days >= MIN_DAYS:
        reasons.append(f"listed {days} days")

    if listing.accepts_offers:
        reasons.append("accepts offers")

    reasons.extend(f for f in listing.condition_flags if f != "lot")
    return reasons


def negotiability_score(listing: Listing, stats: SkuStats, today: date | None = None) -> float:
    """Higher means more room to move. Units are arbitrary; only order matters."""
    over = max(0.0, listing.ask_price - stats.median) / max(stats.median, 1.0)
    return (
        3.0 * over
        + 1.5 * listing.price_cuts
        + min(listing.days_listed(today), 90) / 30.0
        + (0.75 if listing.accepts_offers else 0.0)
        + 0.4 * len([f for f in listing.condition_flags if f != "lot"])
    )
