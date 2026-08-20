from datetime import date

from understudy.decide import build_packet
from understudy.models import Listing, PricePoint
from understudy.truth import ReservationModel, sku_stats


def listing(lid, ask, sold_price=None, **kw):
    base = dict(id=lid, url=f"u{lid}", title="RTX 3080 10GB", ask_price=ask,
                sold_price=sold_price, condition="Used", seller_hash="h",
                seller_type="private", first_seen=date(2026, 1, 1), sku_id="rtx3080")
    base.update(kw)
    return Listing(**base)


def fitted():
    stats = sku_stats([listing(str(i), 600, 460 + i) for i in range(30)])
    train = [listing(f"t{i}", 600 + (i % 5) * 30, 470 + (i % 6) * 4) for i in range(60)]
    return stats, ReservationModel().fit(train, {"rtx3080": stats})


def test_packet_prices_are_ordered():
    stats, model = fitted()
    for ask in (400.0, 600.0, 900.0):
        p = build_packet(listing("a", ask), stats, model)
        assert p.opening < p.target <= p.walk_away < p.ask_price


def test_packet_cites_the_sold_median():
    stats, model = fitted()
    p = build_packet(listing("a", 600), stats, model)
    assert p.facts
    assert any(str(round(stats.median)) in f for f in p.facts)


def test_packet_mentions_price_cuts_when_present():
    stats, model = fitted()
    l = listing("a", 600, price_history=[
        PricePoint(date=date(2026, 1, 1), price=750.0),
        PricePoint(date=date(2026, 2, 1), price=600.0),
    ])
    p = build_packet(l, stats, model, today=date(2026, 3, 1))
    assert any("cut" in f.lower() for f in p.facts)


def test_packet_omits_price_cuts_when_absent():
    stats, model = fitted()
    p = build_packet(listing("a", 600), stats, model)
    assert not any("cut" in f.lower() for f in p.facts)


def test_packet_contains_no_pii():
    stats, model = fitted()
    l = listing("a", 600, description="call 415-555-0134 or bob@x.com")
    blob = build_packet(l, stats, model).model_dump_json()
    assert "415-555-0134" not in blob and "bob@x.com" not in blob
