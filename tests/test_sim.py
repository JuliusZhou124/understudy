from understudy.llm import StubLLM
from understudy.models import Packet, PersonaParams
from understudy.sim import BUYER_TOOLS, SELLER_TOOLS, negotiate
from understudy.strategies import STRATEGIES


def a_packet(ask=500.0, target=430.0, walk=470.0):
    return Packet(listing_id="a", headline="RTX 3080", facts=["Sold comps around $460"],
                  ask_price=ask, target=target, opening=ask * 0.78, walk_away=walk)


def a_persona(**kw):
    base = dict(reservation_ratio=0.85, concession_rate=0.35, patience=8,
                firmness=0.4, hostility=0.2, urgency=0.6)
    base.update(kw)
    return PersonaParams(**base)


def run(packet, persona, seed=1, strategy="two_offer_close", max_turns=24):
    return negotiate(
        packet, persona,
        buyer_llm=StubLLM("buyer", ask_price=packet.ask_price, target=packet.walk_away, seed=seed),
        seller_llm=StubLLM("seller", ask_price=packet.ask_price, params=persona, seed=seed),
        strategy=STRATEGIES[strategy], seed=seed, max_turns=max_turns,
    )


def test_tool_schemas_are_well_formed():
    for tool in BUYER_TOOLS + SELLER_TOOLS:
        assert tool["name"] and tool["description"]
        assert tool["input_schema"]["type"] == "object"
        for req in tool["input_schema"]["required"]:
            assert req in tool["input_schema"]["properties"]


def test_deal_when_floor_is_below_buyer_walkaway():
    r = run(a_packet(walk=470.0), a_persona(reservation_ratio=0.85))  # floor 425
    assert r.outcome == "deal"
    assert r.final_price is not None and r.final_price <= 470.0


def test_no_deal_when_floor_exceeds_buyer_walkaway():
    r = run(a_packet(walk=400.0), a_persona(reservation_ratio=0.98, patience=2))  # floor 490
    assert r.outcome in {"walk", "no_deal", "timeout"}
    assert r.final_price is None


def test_never_settles_below_the_seller_floor():
    r = run(a_packet(walk=490.0), a_persona(reservation_ratio=0.9))
    if r.final_price is not None:
        assert r.final_price >= 500.0 * 0.9 - 1e-6


def test_never_settles_above_the_buyer_walkaway():
    for seed in range(10):
        r = run(a_packet(walk=460.0), a_persona(reservation_ratio=0.8), seed=seed)
        if r.final_price is not None:
            assert r.final_price <= 460.0 + 1e-6


def test_simulation_is_deterministic_for_a_seed():
    assert run(a_packet(), a_persona(), seed=5).model_dump() == \
           run(a_packet(), a_persona(), seed=5).model_dump()


def test_transcript_alternates_and_starts_with_the_buyer():
    r = run(a_packet(), a_persona())
    speakers = [t.speaker for t in r.transcript]
    assert len(speakers) >= 2
    assert speakers[0] == "buyer"
    assert all(a != b for a, b in zip(speakers, speakers[1:]))


def test_offers_are_logged():
    r = run(a_packet(), a_persona())
    assert r.offers
    assert all(o > 0 for o in r.offers)


def test_terminates_at_the_turn_cap():
    r = run(a_packet(walk=100.0), a_persona(reservation_ratio=1.0, patience=20), max_turns=6)
    assert r.turns <= 6


def test_result_records_the_persona_and_strategy():
    r = run(a_packet(), a_persona(), strategy="walk_threat")
    assert r.strategy == "walk_threat"
    assert r.persona is not None
