"""A labelled stand-in for sold-listing data.

READ THIS BEFORE TRUSTING ANY NUMBER DERIVED FROM IT.

eBay CAPTCHA-walls sold/completed search (measured: 0 rows across 5 queries and
many attempts; see the README's Data section). Until `ingest/snapshot.py` has
accumulated enough real delisting observations, this module manufactures a
sold-price history from the **real** active listings so that the fitting,
simulation and calibration machinery can be exercised end to end.

What calibrating against this data does and does not show:

  * IT DOES show whether `ReservationModel` + the persona sampler + the
    simulator can **recover a known generative process** from observable
    features alone. That is a genuine estimator-recovery test, and a failure
    here is a real bug.
  * IT DOES NOT show anything about the real used-GPU market. The process
    below is an assumption, and calibrating a model against your own
    assumptions is circular. Every report separates synthetic rows from real
    ones on `Listing.synthetic`, and the README states which numbers are which.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from understudy.models import Listing, PricePoint

GENERATIVE_PROCESS = """
For each real active listing, holding its real title, SKU, seller type and asking price:

1. Time on market: days ~ Geometric(mean 28), capped at 120.
2. Price cuts: the longer it sits, the more likely a cut. Each cut removes 3-8% of the
   current price, at most 3 cuts, spaced evenly across the observed window.
3. Latent reservation ratio (the seller's true floor as a share of the final ask):
       r = 0.94 - 0.16 * urgency - 0.03 * condition_penalty + 0.04 * is_business + noise
   where urgency rises with days on market and number of cuts, exactly as
   persona.pressure_score defines it, and noise ~ Normal(0, 0.04).
4. Settle price = final ask * clip(r, 0.55, 1.0).

Consequence, by construction: listings that sat longer and cut more settle at a lower
share of their ask. Any model fitted here should recover that relationship; if it does
not, the estimator is broken.
"""

_MEAN_DAYS = 28
_MAX_DAYS = 120
_MAX_CUTS = 3


def synthesize_sold_history(actives: list[Listing], seed: int = 0) -> list[Listing]:
    """Produce synthetic sold counterparts for real active listings."""
    rng = np.random.default_rng(seed)
    out: list[Listing] = []

    for src in actives:
        days = int(min(_MAX_DAYS, max(1, rng.geometric(1 / _MEAN_DAYS))))
        n_cuts = int(min(_MAX_CUTS, rng.binomial(_MAX_CUTS, min(0.9, days / _MAX_DAYS))))

        price = src.ask_price
        history = [PricePoint(date=src.first_seen, price=round(price, 2))]
        for k in range(n_cuts):
            price *= 1.0 - float(rng.uniform(0.03, 0.08))
            when = src.first_seen + timedelta(days=int(days * (k + 1) / (n_cuts + 1)))
            history.append(PricePoint(date=when, price=round(price, 2)))

        urgency = 0.5 * min(1.0, days / 60.0) + 0.5 * min(1.0, n_cuts / 3.0)
        penalty = len(src.condition_flags)
        r = (0.94
             - 0.16 * urgency
             - 0.03 * penalty
             + (0.04 if src.seller_type == "business" else 0.0)
             + float(rng.normal(0, 0.04)))
        r = float(np.clip(r, 0.55, 1.0))

        final_ask = round(price, 2)
        out.append(src.model_copy(update={
            "id": f"{src.id}-syn",
            "synthetic": True,
            "ask_price": final_ask,
            "price_history": history,
            "sold_price": round(final_ask * r, 2),
            "sold_at": src.first_seen + timedelta(days=days),
        }))

    return out
