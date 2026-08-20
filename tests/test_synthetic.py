from datetime import date

import pytest

from understudy.models import Listing
from understudy.synthetic import GENERATIVE_PROCESS, synthesize_sold_history


def listing(lid, ask, **kw):
    base = dict(id=lid, url=f"u{lid}", title="RTX 3080", ask_price=ask, condition="Used",
                seller_hash="h", seller_type="private", first_seen=date(2026, 1, 1),
                sku_id="rtx3080")
    base.update(kw)
    return Listing(**base)


@pytest.fixture
def actives():
    return [listing(f"a{i}", 400.0 + i * 10) for i in range(50)]


def test_every_generated_row_is_flagged_synthetic(actives):
    for l in synthesize_sold_history(actives, seed=1):
        assert l.synthetic is True


def test_generated_rows_do_not_collide_with_real_ids(actives):
    real = {l.id for l in actives}
    assert not (real & {l.id for l in synthesize_sold_history(actives, seed=1)})


def test_every_row_has_a_settle_price_below_its_ask(actives):
    for l in synthesize_sold_history(actives, seed=1):
        assert l.sold_price is not None
        assert 0 < l.sold_price <= l.ask_price


def test_generation_is_deterministic(actives):
    a = synthesize_sold_history(actives, seed=7)
    b = synthesize_sold_history(actives, seed=7)
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]


def test_price_cuts_and_history_are_produced(actives):
    out = synthesize_sold_history(actives, seed=3)
    assert any(l.price_cuts > 0 for l in out)
    assert all(l.price_history for l in out)


def test_listings_that_sat_longer_settle_at_a_lower_ratio(actives):
    out = synthesize_sold_history(actives, seed=5)
    ratios = [(l.days_listed(l.sold_at), l.sold_price / l.ask_price) for l in out]
    quick = [r for d, r in ratios if d <= 14]
    slow = [r for d, r in ratios if d >= 45]
    assert quick and slow
    assert sum(slow) / len(slow) < sum(quick) / len(quick)


def test_the_generative_process_is_documented():
    assert len(GENERATIVE_PROCESS.strip()) > 200
