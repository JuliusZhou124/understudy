from datetime import date

import pytest

from understudy.models import Listing, PricePoint
from understudy.persona import PERSONA_ARCHETYPES, persona_system_prompt, sample_persona
from understudy.truth import ReservationModel, sku_stats


def listing(lid, ask, sold_price=None, **kw):
    base = dict(id=lid, url=f"u{lid}", title="RTX 3080", ask_price=ask,
                sold_price=sold_price, condition="Used", seller_hash="h",
                seller_type="private", first_seen=date(2026, 1, 1), sku_id="rtx3080")
    base.update(kw)
    return Listing(**base)


@pytest.fixture(scope="module")
def fitted():
    stats = sku_stats([listing(str(i), 600, 460 + i) for i in range(30)])
    train = [listing(f"t{i}", 600 + (i % 5) * 40, 480 + (i % 7) * 5) for i in range(60)]
    return stats, ReservationModel().fit(train, {"rtx3080": stats})


def test_sampling_is_deterministic_for_a_seed(fitted):
    stats, model = fitted
    l = listing("a", 620)
    assert sample_persona(l, stats, model, seed=7) == sample_persona(l, stats, model, seed=7)


def test_different_seeds_give_different_personas(fitted):
    stats, model = fitted
    l = listing("a", 620)
    assert sample_persona(l, stats, model, seed=1) != sample_persona(l, stats, model, seed=2)


def test_reservation_ratio_stays_in_range(fitted):
    stats, model = fitted
    l = listing("a", 620)
    for seed in range(50):
        assert 0.0 < sample_persona(l, stats, model, seed=seed).reservation_ratio <= 1.0


def test_urgency_rises_with_days_listed_and_price_cuts(fitted):
    stats, model = fitted
    today = date(2026, 3, 1)
    fresh = listing("a", 620, first_seen=today)
    stale = listing("b", 620, first_seen=date(2025, 9, 1), price_history=[
        PricePoint(date=date(2025, 9, 1), price=800.0),
        PricePoint(date=date(2025, 11, 1), price=700.0),
        PricePoint(date=date(2026, 1, 1), price=620.0),
    ])
    mean = lambda l: sum(sample_persona(l, stats, model, seed=s, today=today).urgency
                         for s in range(20)) / 20
    assert mean(stale) > mean(fresh)


def test_firmness_falls_as_urgency_rises(fitted):
    stats, model = fitted
    today = date(2026, 3, 1)
    fresh = listing("a", 620, first_seen=today)
    stale = listing("b", 620, first_seen=date(2025, 9, 1))
    mean = lambda l: sum(sample_persona(l, stats, model, seed=s, today=today).firmness
                         for s in range(20)) / 20
    assert mean(stale) < mean(fresh)


def test_prompt_states_the_floor_and_forbids_revealing_it(fitted):
    stats, model = fitted
    l = listing("a", 620)
    p = sample_persona(l, stats, model, seed=3)
    prompt = persona_system_prompt(l, p, stats)
    assert str(round(l.ask_price * p.reservation_ratio)) in prompt
    assert "never reveal" in prompt.lower()


def test_prompt_contains_no_pii(fitted):
    stats, model = fitted
    l = listing("a", 620, description="call 415-555-0134")
    p = sample_persona(l, stats, model, seed=3)
    assert "415-555-0134" not in persona_system_prompt(l, p, stats)


def test_archetypes_are_valid_and_ordered():
    a = PERSONA_ARCHETYPES
    assert set(a) == {"motivated", "moderate", "stonewall"}
    assert a["motivated"].reservation_ratio < a["moderate"].reservation_ratio < a["stonewall"].reservation_ratio
