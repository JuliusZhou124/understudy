from understudy.models import Packet
from understudy.strategies import STRATEGIES, buyer_system_prompt


def a_packet():
    return Packet(listing_id="a", headline="RTX 3080 10GB",
                  facts=["Comparable units sold around $460", "Listed 42 days ago"],
                  ask_price=500.0, target=430.0, opening=400.0, walk_away=470.0)


def test_all_strategies_have_distinct_prompts():
    prompts = {n: buyer_system_prompt(a_packet(), s) for n, s in STRATEGIES.items()}
    assert len(set(prompts.values())) == len(STRATEGIES)


def test_prompt_includes_every_fact():
    p = buyer_system_prompt(a_packet(), STRATEGIES["two_offer_close"])
    for fact in a_packet().facts:
        assert fact in p


def test_prompt_forbids_inventing_facts():
    p = buyer_system_prompt(a_packet(), STRATEGIES["two_offer_close"])
    assert "invent" in p.lower()


def test_prompt_marks_the_numbers_as_secret():
    p = buyer_system_prompt(a_packet(), STRATEGIES["anchor_low"])
    assert "never reveal" in p.lower()


def test_prompt_forbids_claiming_to_be_human():
    p = buyer_system_prompt(a_packet(), STRATEGIES["anchor_low"])
    assert "never claim to be human" in p.lower()


def test_strategy_opening_ratios_are_ordered():
    assert STRATEGIES["anchor_low"].opening_ratio < STRATEGIES["deflect_first"].opening_ratio


def test_every_strategy_has_a_playbook():
    assert all(s.playbook.strip() and s.name == n for n, s in STRATEGIES.items())


def test_packet_with_no_facts_still_renders():
    p = Packet(listing_id="a", headline="RTX 3080", facts=[], ask_price=500.0,
               target=430.0, opening=400.0, walk_away=470.0)
    assert "no facts" in buyer_system_prompt(p, STRATEGIES["anchor_low"]).lower()


def test_mechanical_fields_differ_between_strategies():
    assert len({s.concession_step for s in STRATEGIES.values()}) > 1
    assert len({s.walk_turns for s in STRATEGIES.values()}) > 1
    assert STRATEGIES["deflect_first"].deflect_first is True
    assert STRATEGIES["anchor_low"].deflect_first is False
