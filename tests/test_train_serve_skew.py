"""Regression test for a train/serve skew that cost +$24 of bias.

`days_listed` is computed against "today". At fit time a sold listing's today
is its sale date (age 60 days); at predict time, hiding `sold_at` made today
default to the real clock and the age collapsed to 0. The model then read a
brand-new listing, predicted a high reservation ratio, and every predicted
settle came out too high.
"""

from datetime import date, timedelta

from understudy.models import Listing, PricePoint
from understudy.truth import ReservationModel, features, sku_stats


def a_listing(lid, ask, sold_price=None, first_seen=date(2026, 1, 1), sold_at=None):
    return Listing(id=lid, url=f"u{lid}", title="RTX 3080", ask_price=ask,
                   sold_price=sold_price, sold_at=sold_at, condition="Used",
                   seller_hash="h", seller_type="private", first_seen=first_seen,
                   sku_id="rtx3080",
                   price_history=[PricePoint(date=first_seen, price=ask)])


def test_days_listed_uses_the_supplied_as_of_date():
    stats = sku_stats([a_listing(str(i), 600, 500) for i in range(5)])
    l = a_listing("x", 600, first_seen=date(2026, 1, 1))
    assert features(l, stats, today=date(2026, 3, 2))["days_listed"] == 60


def test_simulating_a_sold_listing_as_of_its_sale_date_preserves_its_age():
    """The price is hidden during calibration; the elapsed time is not."""
    from understudy import pipeline
    from understudy.sim import simulate_listing
    from understudy.strategies import STRATEGIES

    first_seen = date(2026, 1, 1)
    sold_at = first_seen + timedelta(days=60)
    train = [a_listing(f"t{i}", 600 + (i % 5) * 30, 470 + (i % 6) * 4,
                       first_seen=first_seen, sold_at=sold_at) for i in range(40)]
    stats = sku_stats(train)
    model = ReservationModel().fit(train, {"rtx3080": stats})

    target = a_listing("x", 600, first_seen=first_seen)
    fit_time = features(train[0], stats, today=sold_at)["days_listed"]

    captured = {}
    real_sample = __import__("understudy.persona", fromlist=["sample_persona"]).sample_persona

    def spy(listing, stats_, model_, seed, today=None):
        captured["days"] = features(listing, stats_, today=today)["days_listed"]
        return real_sample(listing, stats_, model_, seed=seed, today=today)

    import understudy.sim as sim
    sim.sample_persona = spy
    try:
        simulate_listing(target, stats, model, STRATEGIES["anchor_low"],
                         pipeline.make_llm_factory("stub"), seed=0, n=1, today=sold_at)
    finally:
        sim.sample_persona = real_sample

    assert captured["days"] == fit_time == 60
