"""Turning a listing into a negotiating position.

`build_packet` is the only place facts are written, and the packet is the only
thing the buyer agent is told. Everything the agent can truthfully say is
derived here from the listing and the SKU's sold-price statistics; anything
else it says is, by construction, a fabrication — which is what the judge looks
for.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from understudy.models import Listing, Packet, Recommendation, SkuStats
from understudy.resolve import condition_flags
from understudy.truth import ReservationModel, features


def _facts(listing: Listing, stats: SkuStats, today: date | None = None) -> list[str]:
    facts = [
        f"Comparable units sold recently for a median of ${stats.median:.0f} "
        f"(middle half ${stats.q25:.0f}-${stats.q75:.0f}, across {stats.n_sold} sales)."
    ]
    delta = listing.ask_price - stats.median
    if abs(delta) >= 10:
        facts.append(f"This listing is ${abs(delta):.0f} "
                     f"{'above' if delta > 0 else 'below'} that median.")
    if listing.price_cuts:
        facts.append(f"The seller has cut the price {listing.price_cuts} time(s), "
                     f"down {listing.total_drop_pct:.0%} from its high.")
    days = listing.days_listed(today)
    if days >= 7:
        facts.append(f"It has been listed for {days} days.")
    flags = sorted(set(listing.condition_flags)
                   | set(condition_flags(f"{listing.title} {listing.description or ''}")))
    if flags:
        facts.append("Condition notes in the listing itself: " + ", ".join(flags) + ".")
    if listing.accepts_offers:
        facts.append("The listing explicitly accepts offers.")
    return facts


def build_packet(listing: Listing, stats: SkuStats, model: ReservationModel,
                 today: date | None = None) -> Packet:
    q = model.predict_quantiles(features(listing, stats, today=today))
    ask = listing.ask_price

    target = min(ask * q[0.25], ask - 2.0)
    walk_away = min(ask * q[0.75], stats.median, ask - 1.0)
    walk_away = max(walk_away, target)
    opening = min(target * 0.92, ask * q[0.1], target - 1.0)

    return Packet(
        listing_id=listing.id,
        headline=listing.title,
        facts=_facts(listing, stats, today),
        ask_price=ask,
        target=round(target, 2),
        opening=round(opening, 2),
        walk_away=round(walk_away, 2),
    )


def recommend(listing: Listing, stats: SkuStats, model: ReservationModel,
              llm_factory, strategy, n: int = 20, seed: int = 0) -> Recommendation:
    """Simulate this listing `n` times and turn the outcomes into advice."""
    from understudy.sim import simulate_listing

    packet = build_packet(listing, stats, model)
    results = simulate_listing(listing, stats, model, strategy, llm_factory, seed=seed, n=n)
    deals = [r.final_price for r in results if r.outcome == "deal" and r.final_price is not None]
    arr = np.asarray(deals if deals else [listing.ask_price], dtype=float)
    return Recommendation(
        listing_id=listing.id,
        p_deal=len(deals) / len(results) if results else 0.0,
        expected_settle=float(np.median(arr)),
        settle_p10=float(np.percentile(arr, 10)),
        settle_p90=float(np.percentile(arr, 90)),
        opening_offer=packet.opening,
        packet=packet,
    )
