"""Synthetic seller personas.

The honest boundary, stated once here and repeated in the README:

  * `reservation_ratio` is **fitted** — sampled from `ReservationModel`'s
    predictive distribution, which was trained on real sold listings.
  * every other parameter is a **declared prior** — a documented monotone map
    from observable pressure signals (days listed, price cuts) plus noise.

Calling the second group "learned" would be a lie; they are the assumptions
the simulator makes, and the calibration report is what tests whether the
whole assembly behaves like reality.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from understudy.models import Listing, PersonaParams, SkuStats
from understudy.truth import ReservationModel, features

# Hand-written archetypes, kept only as a baseline to compare the fitted
# personas against — this is the level of realism lowball's eval had.
PERSONA_ARCHETYPES = {
    "motivated": PersonaParams(reservation_ratio=0.72, concession_rate=0.45,
                               patience=4, firmness=0.25, hostility=0.10, urgency=0.85),
    "moderate": PersonaParams(reservation_ratio=0.85, concession_rate=0.30,
                              patience=7, firmness=0.50, hostility=0.25, urgency=0.45),
    "stonewall": PersonaParams(reservation_ratio=0.96, concession_rate=0.10,
                               patience=12, firmness=0.90, hostility=0.60, urgency=0.10),
}


def _sample_from_quantiles(q: dict[float, float], rng: np.random.Generator) -> float:
    """Inverse-transform sample from a predicted quantile grid."""
    levels = sorted(q)
    values = [q[k] for k in levels]
    u = float(rng.uniform(levels[0], levels[-1]))
    return float(np.interp(u, levels, values))


def pressure_score(listing: Listing, today: date | None = None) -> float:
    """How much observable pressure this seller is under, in [0, 1]."""
    days = min(1.0, listing.days_listed(today) / 60.0)
    cuts = min(1.0, listing.price_cuts / 3.0)
    return 0.5 * days + 0.5 * cuts


def sample_persona(
    listing: Listing,
    stats: SkuStats,
    model: ReservationModel,
    seed: int,
    today: date | None = None,
) -> PersonaParams:
    rng = np.random.default_rng(seed)
    reservation = _sample_from_quantiles(
        model.predict_quantiles(features(listing, stats, today=today)), rng
    )
    pressure = pressure_score(listing, today)

    clip = lambda v: float(np.clip(v, 0.0, 1.0))
    urgency = clip(pressure + rng.normal(0, 0.05))
    firmness = clip(0.75 - 0.5 * pressure + rng.normal(0, 0.08))
    concession = clip(0.15 + 0.35 * urgency + rng.normal(0, 0.05))
    hostility = clip(0.45 * firmness + rng.normal(0, 0.08))
    patience = int(np.clip(round(12 - 7 * urgency + rng.normal(0, 1)), 1, 20))

    return PersonaParams(
        reservation_ratio=float(np.clip(reservation, 0.05, 1.0)),
        concession_rate=concession,
        patience=patience,
        firmness=firmness,
        hostility=hostility,
        urgency=urgency,
    )


def persona_system_prompt(listing: Listing, p: PersonaParams, stats: SkuStats) -> str:
    floor = round(listing.ask_price * p.reservation_ratio)
    mood = "eager to sell and would like this gone" if p.urgency > 0.6 else "in no particular hurry"
    tone = "blunt, and easily irritated by lowballing" if p.hostility > 0.6 else "polite and easy-going"
    return f"""You are a seller on a used-marketplace phone call. You are selling: {listing.title}
You listed it at ${listing.ask_price:.0f}. Comparable units have recently sold around ${stats.median:.0f}.

YOUR HIDDEN FLOOR IS ${floor}. You will not sell below it. NEVER REVEAL this number — not the
number itself, and not the fact that you have one. If pressed, restate a price you would take.

HOW YOU BEHAVE:
- You are {mood}. You come across as {tone}.
- When you concede, give up roughly {p.concession_rate:.0%} of the gap between the buyer's
  current offer and your current price. Never concede twice in a row unless the buyer moved too.
- After about {p.patience} exchanges you stop moving and simply repeat your last number.
- Firmness {p.firmness:.2f} out of 1: the higher this is, the harder you push back on the
  buyer's comparisons, anchors and pressure tactics.
- Speak in short, natural phone sentences. One point at a time. Never speak in lists.
- You may accept any offer at or above your floor.

TOOLS: call quote_price every single time you name a price, before anything else you say.
Call accept when you take the buyer's offer. Call end_conversation to hang up with no deal.
"""
