"""Longitudinal collection: turning repeated search snapshots into history.

eBay serves active search results but CAPTCHA-walls sold/completed search, so
the settle prices this project needs cannot simply be scraped. They can,
however, be *observed over time*:

  * A listing seen at a new price gains a price-history point — which is where
    `price_cuts` and `total_drop_pct` come from. From a single snapshot both
    features are identically zero, so this module is what makes them real.
  * A listing that was present and is now absent has left the market. Its last
    observed asking price is recorded as the settle price.

The honest caveat, repeated in the README: **disappearance is not proof of
sale.** A seller who gives up and delists looks identical to one who sold. This
biases inferred settle prices upward (unsold withdrawals keep their full ask),
and the bias is reported rather than hidden.
"""

from __future__ import annotations

from datetime import date

from understudy.models import Listing, PricePoint
from understudy.store import Store


def record_snapshot(store: Store, observed: list[Listing],
                    today: date | None = None) -> list[Listing]:
    """Merge one scrape into the store, accumulating price history."""
    today = today or date.today()
    merged: list[Listing] = []

    for fresh in observed:
        prior = store.get(fresh.id)
        if prior is None:
            row = fresh.model_copy(update={
                "first_seen": fresh.first_seen or today,
                "price_history": [PricePoint(date=today, price=fresh.ask_price)],
            })
        else:
            history = list(prior.price_history)
            if not history or history[-1].price != fresh.ask_price:
                history.append(PricePoint(date=today, price=fresh.ask_price))
            row = prior.model_copy(update={
                "ask_price": fresh.ask_price,
                "title": fresh.title,
                "description": fresh.description,
                "accepts_offers": fresh.accepts_offers,
                "seller_feedback": fresh.seller_feedback,
                "price_history": history,
                "first_seen": prior.first_seen,  # never moves forward
            })
        merged.append(row)

    store.upsert(merged)
    return merged


def infer_sales(store: Store, seen_ids: set[str], today: date | None = None) -> list[Listing]:
    """Mark previously-active listings that have vanished as sold at last price.

    `seen_ids` must be every id observed in the snapshot that just completed,
    across *all* queries — otherwise listings merely absent from one query get
    falsely marked sold.
    """
    today = today or date.today()
    gone: list[Listing] = []

    for l in store.all():
        if l.sold_price is not None or l.id in seen_ids or l.synthetic:
            continue
        gone.append(l.model_copy(update={"sold_price": l.ask_price, "sold_at": today}))

    store.upsert(gone)
    return gone
