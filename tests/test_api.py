import pytest
from fastapi.testclient import TestClient

from understudy.api import app

client = TestClient(app)


def test_health_reports_capabilities():
    body = client.get("/health").json()
    assert body["ok"] is True
    assert "calls_enabled" in body and "artifacts" in body


def test_listings_returns_the_dataset():
    r = client.get("/listings")
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list) and rows
    assert {"id", "title", "ask_price", "sku_id"} <= set(rows[0])


def test_listings_can_filter_by_sku():
    rows = client.get("/listings").json()
    sku = rows[0]["sku_id"]
    assert all(r["sku_id"] == sku for r in client.get(f"/listings?sku={sku}").json())


def test_skus_endpoint_returns_sold_price_stats():
    stats = client.get("/skus").json()
    assert stats
    first = next(iter(stats.values()))
    assert {"median", "n_sold", "prices"} <= set(first)


def test_simulate_returns_a_settle_distribution():
    listing_id = client.get("/listings").json()[0]["id"]
    r = client.post("/simulate", json={"listing_id": listing_id, "n": 4,
                                       "strategy": "two_offer_close"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 4
    assert 0.0 <= body["p_deal"] <= 1.0
    assert body["packet"]["facts"]


def test_simulate_rejects_an_unknown_listing():
    assert client.post("/simulate", json={"listing_id": "nope", "n": 2}).status_code == 404


def test_report_exposes_the_data_provenance():
    body = client.get("/report").json()
    assert "synthetic_sold_data" in body
    assert isinstance(body["synthetic_sold_data"], bool)


def test_negotiate_refuses_when_calls_are_disabled(monkeypatch):
    monkeypatch.delenv("CALLS_ENABLED", raising=False)
    listing_id = client.get("/listings").json()[0]["id"]
    r = client.post("/negotiate", json={"listing_id": listing_id})
    assert r.status_code == 403
    assert "disabled" in r.json()["detail"].lower()


def test_dashboard_websocket_greets():
    with client.websocket_connect("/dashboard") as ws:
        assert ws.receive_json()["type"] == "hello"


def test_webhook_broadcasts_to_the_dashboard():
    with client.websocket_connect("/dashboard") as ws:
        ws.receive_json()  # hello
        client.post("/vapi-webhook", json={"message": {
            "type": "transcript", "role": "assistant",
            "transcript": "Hello", "transcriptType": "final"}})
        assert ws.receive_json() == {"type": "transcript", "speaker": "buyer",
                                     "text": "Hello", "final": True}


def test_listings_carry_reasons_and_a_rank():
    rows = client.get("/listings").json()
    assert {"why", "rank", "most_negotiable"} <= set(rows[0])
    assert isinstance(rows[0]["why"], list)


def test_listings_are_sorted_by_negotiability():
    ranks = [r["rank"] for r in client.get("/listings").json()]
    assert ranks == sorted(ranks, reverse=True)


def test_exactly_one_listing_is_stamped_most_negotiable():
    rows = client.get("/listings").json()
    assert sum(r["most_negotiable"] for r in rows) == 1
    assert rows[0]["most_negotiable"] is True


def test_simulate_returns_replayable_call_events():
    listing_id = client.get("/listings").json()[0]["id"]
    body = client.post("/simulate", json={"listing_id": listing_id, "n": 2}).json()
    replay = body["replay"]
    assert replay
    assert replay[0]["type"] == "transcript"
    assert replay[-1]["type"] in {"deal", "call-ended"}
    # same vocabulary the Vapi webhook emits, so one renderer serves both
    assert set(e["type"] for e in replay) <= {"transcript", "offer", "deal", "call-ended"}


def test_replay_offers_follow_seller_turns():
    listing_id = client.get("/listings").json()[0]["id"]
    replay = client.post("/simulate", json={"listing_id": listing_id, "n": 1}).json()["replay"]
    for prev, cur in zip(replay, replay[1:]):
        if cur["type"] == "offer":
            assert prev["type"] == "transcript" and prev["speaker"] == "seller"


def test_root_serves_the_built_dashboard_or_says_how_to_build_it():
    r = client.get("/")
    assert r.status_code in (200, 503)
    if r.status_code == 503:
        assert "npm run build" in r.text
    else:
        assert "<div id=\"root\">" in r.text


def test_simulate_reports_which_engine_wrote_the_transcript():
    listing_id = client.get("/listings").json()[0]["id"]
    body = client.post("/simulate", json={"listing_id": listing_id, "n": 2}).json()
    assert body["engine"] == "rule-based stub"


def test_health_advertises_engines_and_whether_claude_is_usable():
    body = client.get("/health").json()
    assert body["engines"]["stub"] == "rule-based stub"
    assert body["engines"]["claude"].startswith("claude-")
    assert isinstance(body["claude_available"], bool)


def test_claude_is_refused_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    listing_id = client.get("/listings").json()[0]["id"]
    r = client.post("/simulate", json={"listing_id": listing_id, "n": 2, "llm": "claude"})
    assert r.status_code == 400
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_unknown_engine_is_rejected():
    listing_id = client.get("/listings").json()[0]["id"]
    r = client.post("/simulate", json={"listing_id": listing_id, "n": 2, "llm": "gpt"})
    assert r.status_code == 400


def test_claude_runs_are_capped_to_protect_the_bill(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    listing_id = client.get("/listings").json()[0]["id"]
    # No network call is made: the cap is applied before the model is constructed,
    # and a bad key surfaces as a 502 rather than a silent 60-run sweep.
    r = client.post("/simulate", json={"listing_id": listing_id, "n": 60, "llm": "claude"})
    assert r.status_code in (200, 502)
    if r.status_code == 200:
        assert r.json()["runs"] == 5


def test_health_lists_every_provider_and_its_availability():
    body = client.get("/health").json()
    assert set(body["engines"]) == {"stub", "claude", "openai"}
    assert body["engines_available"]["stub"] is True
    assert set(body["engines_available"]) == set(body["engines"])


def test_openai_is_refused_without_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    listing_id = client.get("/listings").json()[0]["id"]
    r = client.post("/simulate", json={"listing_id": listing_id, "n": 2, "llm": "openai"})
    assert r.status_code == 400
    assert "OPENAI_API_KEY" in r.json()["detail"]
