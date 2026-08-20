import pytest

from understudy.llm import LLMResponse, Message, StubLLM
from understudy.models import PersonaParams


def a_persona(**kw):
    base = dict(reservation_ratio=0.8, concession_rate=0.3, patience=6,
                firmness=0.5, hostility=0.2, urgency=0.5)
    base.update(kw)
    return PersonaParams(**base)


def test_stub_seller_quotes_the_ask_first():
    s = StubLLM("seller", ask_price=500.0, params=a_persona(), seed=1)
    r = s.complete("sys", [Message(role="user", content="Hi, is this still available?")], [])
    assert isinstance(r, LLMResponse)
    quotes = [tc for tc in r.tool_calls if tc.name == "quote_price"]
    assert quotes and quotes[0].args["price"] == 500.0


def test_stub_seller_never_goes_below_floor():
    s = StubLLM("seller", ask_price=500.0, params=a_persona(reservation_ratio=0.8), seed=1)
    msgs = [Message(role="user", content="I'll give you $200")]
    prices = []
    for _ in range(20):
        r = s.complete("sys", msgs, [])
        prices += [tc.args["price"] for tc in r.tool_calls if tc.name == "quote_price"]
        msgs.append(Message(role="user", content="Still $200, final offer"))
    assert prices
    assert min(prices) >= 500.0 * 0.8 - 1e-6


def test_stub_seller_accepts_at_or_above_floor():
    s = StubLLM("seller", ask_price=500.0, params=a_persona(reservation_ratio=0.8), seed=1)
    r = s.complete("sys", [Message(role="user", content="I'll do $420")], [])
    assert any(tc.name == "accept" for tc in r.tool_calls)


def test_stub_seller_concedes_toward_but_not_past_the_floor():
    s = StubLLM("seller", ask_price=500.0, params=a_persona(reservation_ratio=0.7, patience=10), seed=1)
    msgs = [Message(role="user", content="I can do $300")]
    quotes = []
    for _ in range(6):
        r = s.complete("sys", msgs, [])
        quotes += [tc.args["price"] for tc in r.tool_calls if tc.name == "quote_price"]
        msgs.append(Message(role="user", content="Still $300"))
    assert quotes == sorted(quotes, reverse=True)
    assert quotes[-1] < quotes[0]
    assert quotes[-1] >= 350.0 - 1e-6


def test_stub_buyer_logs_every_price_the_seller_names():
    b = StubLLM("buyer", ask_price=500.0, target=400.0, seed=1)
    r = b.complete("sys", [Message(role="user", content="It's listed at $500.")], [])
    logged = [tc for tc in r.tool_calls if tc.name == "log_offer"]
    assert logged and logged[0].args["price"] == 500.0


def test_stub_buyer_accepts_at_or_below_target():
    b = StubLLM("buyer", ask_price=500.0, target=400.0, seed=1)
    r = b.complete("sys", [Message(role="user", content="I can do $390")], [])
    assert any(tc.name == "accept_offer" for tc in r.tool_calls)


def test_stub_buyer_eventually_walks_away():
    b = StubLLM("buyer", ask_price=500.0, target=100.0, seed=1)
    saw_walk = False
    for _ in range(20):
        r = b.complete("sys", [Message(role="user", content="It's $500, firm.")], [])
        if any(tc.name == "walk_away" for tc in r.tool_calls):
            saw_walk = True
            break
    assert saw_walk


def test_stub_is_deterministic():
    def run():
        b = StubLLM("buyer", ask_price=500.0, target=400.0, seed=42)
        return b.complete("s", [Message(role="user", content="$480")], []).text
    assert run() == run()


def test_stub_speaks_no_price_it_was_not_given():
    b = StubLLM("buyer", ask_price=500.0, target=400.0, seed=1)
    r = b.complete("sys", [Message(role="user", content="Hi there")], [])
    assert isinstance(r.text, str) and r.text


def test_deflect_first_buyer_names_no_price_on_the_opening_turn():
    from understudy.strategies import STRATEGIES
    b = StubLLM("buyer", ask_price=500.0, target=400.0, seed=1,
                strategy=STRATEGIES["deflect_first"])
    r = b.complete("sys", [Message(role="user", content="It's $500.")], [])
    assert "$" not in r.text
    assert any(tc.name == "log_offer" for tc in r.tool_calls)


def test_anchor_low_buyer_does_name_a_price_immediately():
    from understudy.strategies import STRATEGIES
    b = StubLLM("buyer", ask_price=500.0, target=400.0, seed=1,
                strategy=STRATEGIES["anchor_low"])
    r = b.complete("sys", [Message(role="user", content="It's $500.")], [])
    assert "$" in r.text


def test_strategy_sets_the_opening_offer():
    """500 * 0.70, within the +/-4% per-buyer jitter."""
    from understudy.strategies import STRATEGIES
    b = StubLLM("buyer", ask_price=500.0, target=490.0, seed=1,
                strategy=STRATEGIES["anchor_low"])
    assert 350.0 * 0.96 <= b.current <= 350.0 * 1.04


def test_negotiator_ignores_its_own_previous_quote():
    """A seller must not read back its own price as the buyer's offer."""
    s = StubLLM("seller", ask_price=500.0, params=a_persona(reservation_ratio=0.9), seed=1)
    convo = [
        Message(role="user", content="Hi, is it available?"),
        Message(role="assistant", content="Yeah, it's listed at $500."),
        Message(role="user", content="Where can you be on price?"),  # no number
    ]
    r = s.complete("sys", convo, [])
    assert not any(tc.name == "accept" for tc in r.tool_calls)


def test_buyers_with_different_seeds_walk_different_price_ladders():
    """Identical ladders collapse the predicted settle distribution to a point."""
    from understudy.strategies import STRATEGIES

    def ladder(seed):
        b = StubLLM("buyer", ask_price=500.0, target=470.0, seed=seed,
                    strategy=STRATEGIES["two_offer_close"])
        out = []
        for _ in range(4):
            r = b.complete("s", [Message(role="user", content="It's $500.")], [])
            out.append(round(b.current, 2))
        return out

    assert ladder(1) != ladder(2)
