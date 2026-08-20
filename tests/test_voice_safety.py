from datetime import date

import pytest

from understudy.models import Listing
from understudy.voice.safety import CallRefused, resolve_call_number


def a_listing(**kw):
    base = dict(id="a", url="u", title="RTX 3080", ask_price=500.0, condition="Used",
                seller_hash="h", seller_type="business", first_seen=date(2026, 1, 1),
                phone="+15551112222")
    base.update(kw)
    return Listing(**base)


def test_refuses_when_calls_disabled(monkeypatch):
    monkeypatch.delenv("CALLS_ENABLED", raising=False)
    monkeypatch.setenv("DEMO_CALL_NUMBER", "+15550001111")
    with pytest.raises(CallRefused, match="disabled"):
        resolve_call_number(a_listing())


def test_refuses_private_sellers(monkeypatch):
    monkeypatch.setenv("CALLS_ENABLED", "true")
    monkeypatch.setenv("DEMO_CALL_NUMBER", "+15550001111")
    with pytest.raises(CallRefused, match="private"):
        resolve_call_number(a_listing(seller_type="private", phone=None))


def test_refuses_unknown_seller_type(monkeypatch):
    monkeypatch.setenv("CALLS_ENABLED", "true")
    monkeypatch.setenv("DEMO_CALL_NUMBER", "+15550001111")
    with pytest.raises(CallRefused):
        resolve_call_number(a_listing(seller_type="unknown"))


def test_always_returns_the_demo_number_never_the_listing_phone(monkeypatch):
    monkeypatch.setenv("CALLS_ENABLED", "true")
    monkeypatch.setenv("DEMO_CALL_NUMBER", "+15550001111")
    assert resolve_call_number(a_listing(phone="+15559998888")) == "+15550001111"


def test_refuses_when_demo_number_missing(monkeypatch):
    monkeypatch.setenv("CALLS_ENABLED", "true")
    monkeypatch.delenv("DEMO_CALL_NUMBER", raising=False)
    with pytest.raises(CallRefused, match="DEMO_CALL_NUMBER"):
        resolve_call_number(a_listing())


def test_refuses_a_malformed_demo_number(monkeypatch):
    monkeypatch.setenv("CALLS_ENABLED", "true")
    monkeypatch.setenv("DEMO_CALL_NUMBER", "555-hello")
    with pytest.raises(CallRefused, match="E.164"):
        resolve_call_number(a_listing())
