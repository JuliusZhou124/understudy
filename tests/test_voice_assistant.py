import re

from understudy.models import Packet
from understudy.strategies import STRATEGIES
from understudy.voice.assistant import spoken_usd, voice_system_prompt


def a_packet():
    return Packet(listing_id="a", headline="RTX 3080 10GB",
                  facts=["Comparable units sold around $460"],
                  ask_price=500.0, target=430.0, opening=390.0, walk_away=470.0)


def test_spoken_usd_writes_numbers_as_words():
    assert spoken_usd(12500) == "twelve thousand five hundred dollars"
    assert spoken_usd(460) == "four hundred sixty dollars"
    assert spoken_usd(0) == "zero dollars"
    assert spoken_usd(1) == "one dollar"
    assert spoken_usd(19) == "nineteen dollars"
    assert spoken_usd(1000) == "one thousand dollars"


def test_voice_prompt_speaks_prices_as_words():
    p = voice_system_prompt(a_packet(), STRATEGIES["two_offer_close"])
    assert "five hundred dollars" in p


def test_voice_prompt_contains_no_dollar_figures():
    p = voice_system_prompt(a_packet(), STRATEGIES["two_offer_close"])
    assert not re.search(r"\$\s?\d", p)


def test_voice_prompt_forbids_claiming_to_be_human():
    p = voice_system_prompt(a_packet(), STRATEGIES["anchor_low"])
    assert "never claim to be human" in p.lower()


def test_voice_prompt_keeps_the_facts():
    p = voice_system_prompt(a_packet(), STRATEGIES["anchor_low"])
    assert "Comparable units sold around four hundred sixty dollars" in p
