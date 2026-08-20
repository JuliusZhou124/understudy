from datetime import date

from understudy.models import Listing, PricePoint, SkuStats
from understudy.rank import negotiability_score, why_reasons


def listing(lid="a", ask=500.0, **kw):
    base = dict(id=lid, url=f"u{lid}", title="RTX 3080", ask_price=ask, condition="Used",
                seller_hash="h", seller_type="private", first_seen=date(2026, 1, 1),
                sku_id="rtx3080")
    base.update(kw)
    return Listing(**base)


STATS = SkuStats(sku_id="rtx3080", n_sold=100, median=400.0, q25=350.0, q75=450.0,
                 prices=[400.0] * 100)
TODAY = date(2026, 1, 1)


def test_reasons_lead_with_the_gap_to_median():
    reasons = why_reasons(listing(ask=500.0), STATS, today=TODAY)
    assert reasons[0] == "$100 over median"


def test_under_median_is_stated_as_under():
    assert why_reasons(listing(ask=350.0), STATS, today=TODAY)[0] == "$50 under median"


def test_reasons_include_price_cuts_and_offers():
    l = listing(accepts_offers=True, price_history=[
        PricePoint(date=date(2026, 1, 1), price=600.0),
        PricePoint(date=date(2026, 1, 8), price=500.0),
    ])
    reasons = why_reasons(l, STATS, today=TODAY)
    assert "1 price cut" in reasons
    assert "accepts offers" in reasons


def test_reasons_pluralise_cuts():
    l = listing(price_history=[PricePoint(date=date(2026, 1, 1), price=700.0),
                               PricePoint(date=date(2026, 1, 5), price=600.0),
                               PricePoint(date=date(2026, 1, 9), price=500.0)])
    assert "2 price cuts" in why_reasons(l, STATS, today=TODAY)


def test_reasons_include_days_listed_when_meaningful():
    l = listing(first_seen=date(2025, 11, 1))
    assert "listed 61 days" in why_reasons(l, STATS, today=TODAY)


def test_reasons_surface_condition_flags():
    assert "mining" in why_reasons(listing(condition_flags=["mining"]), STATS, today=TODAY)


def test_score_rises_with_overpricing_cuts_and_age():
    plain = listing()
    juicy = listing(ask=700.0, accepts_offers=True, first_seen=date(2025, 11, 1),
                    price_history=[PricePoint(date=date(2025, 11, 1), price=900.0),
                                   PricePoint(date=date(2025, 12, 1), price=700.0)])
    assert negotiability_score(juicy, STATS, today=TODAY) > negotiability_score(plain, STATS, today=TODAY)


def test_score_is_zero_or_more():
    assert negotiability_score(listing(ask=100.0), STATS, today=TODAY) >= 0.0
