import pytest
from datetime import date
from understudy.models import Listing, PricePoint, PersonaParams


def a_listing(**kw):
    base = dict(
        id="l1", source="ebay", url="https://x/1", title="RTX 3080 FE 10GB",
        ask_price=500.0, condition="Used", seller_hash="abc123",
        seller_type="private", first_seen=date(2026, 1, 1),
    )
    base.update(kw)
    return Listing(**base)


def test_listing_defaults():
    l = a_listing()
    assert l.sold_price is None
    assert l.price_history == []
    assert l.condition_flags == []


def test_days_listed_computed_from_price_history():
    l = a_listing(price_history=[
        PricePoint(date=date(2026, 1, 1), price=600.0),
        PricePoint(date=date(2026, 1, 21), price=500.0),
    ])
    assert l.price_cuts == 1
    assert l.total_drop_pct == pytest.approx(100 / 600)


def test_persona_params_reject_out_of_range_reservation():
    with pytest.raises(ValueError):
        PersonaParams(reservation_ratio=1.5, concession_rate=0.3,
                      patience=5, firmness=0.5, hostility=0.2, urgency=0.4)
