import gzip
from datetime import date
from pathlib import Path

import pytest

from understudy.ingest.ebay import parse_search_html, search_url

FIX = Path(__file__).parent / "fixtures"
# Stored gzipped: the captured page is 3 MB uncompressed, which would be most
# of the repository. The parser is the thing under test, not the disk format.
ACTIVE = FIX / "ebay_search.html.gz"
SOLD = FIX / "ebay_sold.html.gz"


def read_fixture(path: Path) -> str:
    return gzip.decompress(path.read_bytes()).decode()


def test_search_url_sets_sold_filters():
    u = search_url("rtx 3080", sold=True)
    assert "LH_Sold=1" in u and "LH_Complete=1" in u
    assert "rtx+3080" in u


def test_search_url_omits_pgn_on_page_one():
    assert "_pgn" not in search_url("rtx 3080", sold=False)
    assert "_pgn=2" in search_url("rtx 3080", sold=False, page=2)


@pytest.fixture(scope="module")
def active():
    if not ACTIVE.exists():
        pytest.skip("active fixture not captured")
    return parse_search_html(read_fixture(ACTIVE), sold=False, today=date(2026, 8, 20))


def test_parses_many_real_listings(active):
    assert len(active) >= 40


def test_every_listing_has_the_fields_downstream_needs(active):
    for l in active:
        assert l.title and l.ask_price > 0
        assert l.url.startswith("https://")
        assert len(l.seller_hash) == 16
        assert l.sold_price is None


def test_placeholder_shop_on_ebay_card_is_dropped(active):
    assert not any(l.title.lower().startswith("shop on ebay") for l in active)


def test_ids_are_stable_and_unique(active):
    ids = [l.id for l in active]
    assert len(set(ids)) == len(ids)
    again = parse_search_html(read_fixture(ACTIVE), sold=False, today=date(2026, 8, 20))
    assert [l.id for l in again] == ids


def test_best_offer_listings_are_flagged(active):
    assert any(l.accepts_offers for l in active)


def test_conditions_are_captured(active):
    conds = {l.condition for l in active}
    assert any("Pre-Owned" in c or "New" in c or "Refurbished" in c or "Open" in c for c in conds)


def test_seller_feedback_is_parsed_for_some_listings(active):
    assert any(l.seller_feedback and l.seller_feedback > 0 for l in active)


def test_no_raw_pii_survives_parsing(active):
    for l in active:
        blob = l.model_dump_json()
        assert "% positive" not in blob  # seller line is hashed, never stored raw
        assert l.phone is None


@pytest.mark.skipif(not SOLD.exists(), reason="sold fixture not captured (eBay CAPTCHA-walls sold search)")
def test_sold_listings_carry_a_sold_price():
    sold = parse_search_html(read_fixture(SOLD), sold=True, today=date(2026, 8, 20))
    assert len(sold) >= 20
    assert all(l.sold_price is not None for l in sold)
    assert all(l.sold_price == l.ask_price for l in sold)


def test_screen_reader_noise_is_stripped_from_titles(active):
    assert not any("Opens in a new window" in l.title for l in active)
