from datetime import date

import pytest

from understudy.models import Listing, PricePoint
from understudy.truth import FEATURE_ORDER, ReservationModel, features, sku_stats


def sold(lid, ask, sold_price=None, **kw):
    base = dict(id=lid, url=f"u{lid}", title="RTX 3080", ask_price=ask,
                sold_price=sold_price, condition="Used", seller_hash="h",
                seller_type="private", first_seen=date(2026, 1, 1), sku_id="rtx3080")
    base.update(kw)
    return Listing(**base)


def test_sku_stats_computes_median_and_quartiles():
    s = sku_stats([sold(str(i), 600, p) for i, p in enumerate([400, 450, 500, 550, 600])])
    assert s.n_sold == 5
    assert s.median == 500
    assert s.q25 == 450 and s.q75 == 550


def test_sku_stats_rejects_empty_input():
    with pytest.raises(ValueError):
        sku_stats([])


def test_features_are_stable_and_complete():
    s = sku_stats([sold(str(i), 600, 500) for i in range(5)])
    l = sold("x", 600, None, price_history=[
        PricePoint(date=date(2026, 1, 1), price=700.0),
        PricePoint(date=date(2026, 2, 1), price=600.0),
    ])
    f = features(l, s, today=date(2026, 3, 1))
    assert set(f) == set(FEATURE_ORDER)
    assert f["ask_ratio"] == pytest.approx(600 / 500)
    assert f["price_cuts"] == 1
    assert f["days_listed"] == 59


def test_model_learns_that_high_ask_ratio_means_bigger_discount():
    stats = sku_stats([sold(str(i), 600, 500) for i in range(30)])
    train = []
    for i in range(60):
        overpriced = i % 2 == 0
        ask = 800.0 if overpriced else 520.0
        settle = 500.0 if overpriced else 495.0
        train.append(sold(f"t{i}", ask, settle))
    m = ReservationModel().fit(train, {"rtx3080": stats})
    hi = m.predict_quantiles(features(sold("a", 800), stats, today=date(2026, 3, 1)))
    lo = m.predict_quantiles(features(sold("b", 520), stats, today=date(2026, 3, 1)))
    assert hi[0.5] < lo[0.5]
    assert 0 < hi[0.5] <= 1.0


def test_quantiles_are_monotone():
    stats = sku_stats([sold(str(i), 600, 500) for i in range(30)])
    train = [sold(f"t{i}", 600 + i, 500 + (i % 7)) for i in range(60)]
    m = ReservationModel().fit(train, {"rtx3080": stats})
    q = m.predict_quantiles(features(sold("a", 620), stats, today=date(2026, 3, 1)))
    assert q[0.1] <= q[0.25] <= q[0.5] <= q[0.75] <= q[0.9]


def test_fit_refuses_too_little_data():
    stats = sku_stats([sold(str(i), 600, 500) for i in range(5)])
    with pytest.raises(ValueError, match="20"):
        ReservationModel().fit([sold("a", 600, 500)], {"rtx3080": stats})


def test_model_roundtrips_to_disk(tmp_path):
    stats = sku_stats([sold(str(i), 600, 500) for i in range(30)])
    train = [sold(f"t{i}", 600 + i, 500 + (i % 7)) for i in range(60)]
    m = ReservationModel().fit(train, {"rtx3080": stats})
    f = features(sold("a", 620), stats, today=date(2026, 3, 1))
    before = m.predict_quantiles(f)
    m.save(tmp_path / "m.pkl")
    assert ReservationModel.load(tmp_path / "m.pkl").predict_quantiles(f) == before
