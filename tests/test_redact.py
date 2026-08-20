from datetime import date
from understudy.ingest.redact import hash_seller, strip_pii, redact_listing
from understudy.models import Listing


def test_hash_seller_is_stable_and_short():
    assert hash_seller("gpu_guy_99") == hash_seller("gpu_guy_99")
    assert len(hash_seller("gpu_guy_99")) == 16
    assert "gpu_guy_99" not in hash_seller("gpu_guy_99")


def test_strip_pii_removes_phone_email_and_address():
    t = "Call me 415-555-0134 or bob@example.com, at 123 Maple Street"
    out = strip_pii(t)
    assert "415-555-0134" not in out
    assert "bob@example.com" not in out
    assert "123 Maple Street" not in out
    assert "[phone]" in out and "[email]" in out and "[address]" in out


def test_strip_pii_handles_none():
    assert strip_pii(None) is None


def test_redact_listing_scrubs_description_and_drops_raw_phone():
    l = Listing(
        id="a", url="u", title="RTX 3080 call 415-555-0134",
        description="email me at x@y.com", ask_price=500.0, condition="Used",
        seller_hash="raw_handle", seller_type="private", first_seen=date(2026, 1, 1),
        phone="+14155550134",
    )
    r = redact_listing(l)
    assert "415-555-0134" not in r.title
    assert "x@y.com" not in r.description
    assert r.phone is None
