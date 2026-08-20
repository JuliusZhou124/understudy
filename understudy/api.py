"""FastAPI app: REST for the dashboard, a WebSocket for live call events.

Artefacts (fitted model, SKU stats, listing store) are loaded once at import
and held in memory — this is a local demo, not a service.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from understudy import pipeline
from understudy.decide import build_packet
from understudy.rank import negotiability_score, why_reasons
from understudy.sim import simulate_listing
from understudy.strategies import STRATEGIES
from understudy.voice.safety import CallRefused, calls_enabled
from understudy.voice.vapi import create_call, handle_webhook

# The dashboard is a Vite + React app built to frontend/dist. FastAPI serves
# the built output; `npm run dev` proxies the API here during development.
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
DIST = FRONTEND / "dist"

app = FastAPI(title="Understudy")

try:
    ART = pipeline.load()
except SystemExit:  # artefacts not built yet — /health still answers
    ART = None

_clients: set[WebSocket] = set()


async def broadcast(event: dict) -> None:
    dead = []
    for ws in _clients:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


def _artifacts() -> pipeline.Artifacts:
    if ART is None:
        raise HTTPException(503, "artefacts not built — run `python -m understudy build`")
    return ART


def _replay_events(result) -> list[dict]:
    """Reconstruct a simulated run as the same event stream a live call emits.

    One offer is logged per seller quote, in order, so walking the transcript
    and consuming `result.offers` at each seller turn reproduces the live
    sequence exactly — which lets the call card replay a simulation without
    a phone, using identical rendering code.
    """
    events: list[dict] = []
    pending = list(result.offers)
    for turn in result.transcript:
        events.append({"type": "transcript", "speaker": turn.speaker,
                       "text": turn.text, "final": True})
        if turn.speaker == "seller" and pending:
            events.append({"type": "offer", "price": pending.pop(0)})
    if result.outcome == "deal" and result.final_price is not None:
        events.append({"type": "deal", "price": result.final_price})
    else:
        events.append({"type": "call-ended", "outcome": result.outcome, "reason": ""})
    return events


class SimulateRequest(BaseModel):
    listing_id: str
    n: int = 12
    strategy: str = "two_offer_close"
    seed: int = 0
    llm: str = "stub"


# Each run is a full multi-turn negotiation, so a live-model sweep costs real
# money and real seconds. Cap it rather than let a stray "60" in the runs box
# bill the user. Applies to every provider except the stub.
MAX_CLAUDE_RUNS = 5


class NegotiateRequest(BaseModel):
    listing_id: str
    strategy: str = "two_offer_close"


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "artifacts": ART is not None,
        "calls_enabled": calls_enabled(),
        "claude_available": pipeline.claude_available(),
        "engines": {k: pipeline.engine_label(k) for k in pipeline.PROVIDERS},
        "engines_available": {k: pipeline.provider_available(k) for k in pipeline.PROVIDERS},
        "max_claude_runs": MAX_CLAUDE_RUNS,
        "listings": len(ART.store.all()) if ART else 0,
        "skus": sorted(ART.stats) if ART else [],
        "strategies": sorted(STRATEGIES),
    }


@app.get("/listings")
async def listings(sku: str | None = None, limit: int = 200) -> list[dict]:
    art = _artifacts()
    rows = []
    for l in art.actives():
        if sku and l.sku_id != sku:
            continue
        stats = art.stats[l.sku_id]
        rows.append({
            "id": l.id, "title": l.title, "url": l.url, "ask_price": l.ask_price,
            "sku_id": l.sku_id, "condition": l.condition, "seller_type": l.seller_type,
            "accepts_offers": l.accepts_offers, "condition_flags": l.condition_flags,
            "days_listed": l.days_listed(), "price_cuts": l.price_cuts,
            "sold_median": stats.median,
            "over_median": round(l.ask_price - stats.median, 2),
            "why": why_reasons(l, stats),
            "rank": round(negotiability_score(l, stats), 4),
        })
    rows.sort(key=lambda r: -r["rank"])
    for i, r in enumerate(rows):
        r["most_negotiable"] = i == 0
    return rows[:limit]


@app.get("/skus")
async def skus() -> dict:
    art = _artifacts()
    return {k: json.loads(v.model_dump_json()) for k, v in art.stats.items()}


@app.post("/simulate")
async def simulate(req: SimulateRequest) -> dict:
    art = _artifacts()
    listing = art.store.get(req.listing_id)
    if listing is None or listing.sku_id not in art.stats:
        raise HTTPException(404, "no such listing")
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"unknown strategy {req.strategy}")
    if req.llm not in pipeline.PROVIDERS:
        raise HTTPException(400, f"unknown llm {req.llm}")
    if not pipeline.provider_available(req.llm):
        key = pipeline.PROVIDERS[req.llm]["key"]
        raise HTTPException(400, f"{key} is not set — cannot run {req.llm}.")

    n = req.n if req.llm == "stub" else min(req.n, MAX_CLAUDE_RUNS)
    stats = art.stats[listing.sku_id]
    packet = build_packet(listing, stats, art.model)
    try:
        results = simulate_listing(listing, stats, art.model, STRATEGIES[req.strategy],
                                   pipeline.make_llm_factory(req.llm), seed=req.seed, n=n)
    except Exception as e:  # a live model can fail in ways the stub never does
        raise HTTPException(502, f"{req.llm} run failed: {e}")
    settles = [r.final_price for r in results if r.final_price is not None]
    return {
        "engine": pipeline.engine_label(req.llm),
        "requested_runs": req.n,
        "runs": n,
        "listing": {"id": listing.id, "title": listing.title, "ask_price": listing.ask_price,
                    "sku_id": listing.sku_id},
        "packet": json.loads(packet.model_dump_json()),
        "sold_prices": stats.prices,
        "sold_median": stats.median,
        "p_deal": len(settles) / len(results) if results else 0.0,
        "settles": settles,
        "results": [{"outcome": r.outcome, "final_price": r.final_price, "turns": r.turns,
                     "reservation_ratio": r.persona.reservation_ratio if r.persona else None,
                     "floor": round(listing.ask_price * r.persona.reservation_ratio, 2)
                     if r.persona else None}
                    for r in results],
        "transcript": [{"speaker": t.speaker, "text": t.text} for t in results[0].transcript]
        if results else [],
        "replay": _replay_events(results[0]) if results else [],
    }


@app.get("/report")
async def report() -> dict:
    art = _artifacts()
    results_dir = Path("docs/results")

    def read(name: str) -> str:
        f = results_dir / name
        return f.read_text() if f.exists() else "(not generated yet)"

    return {
        "listings_total": len(art.store.all()),
        "listings_real": len([l for l in art.store.all() if not l.synthetic]),
        "sold_real": len(art.sold(synthetic=False)),
        "sold_synthetic": len(art.sold(synthetic=True)),
        "synthetic_sold_data": len(art.sold(synthetic=False)) == 0,
        "calibration": read("calibration.txt"),
        "eval": read("eval.txt"),
    }


@app.post("/negotiate")
async def negotiate(req: NegotiateRequest) -> dict:
    art = _artifacts()
    listing = art.store.get(req.listing_id)
    if listing is None or listing.sku_id not in art.stats:
        raise HTTPException(404, "no such listing")
    packet = build_packet(listing, art.stats[listing.sku_id], art.model)
    try:
        call = create_call(listing, packet, STRATEGIES[req.strategy])
    except CallRefused as e:
        raise HTTPException(403, str(e))
    await broadcast({"type": "call-started", "listing": listing.title,
                     "ask_price": listing.ask_price, "target": packet.target})
    return {"call_id": call.get("id"), "packet": json.loads(packet.model_dump_json())}


@app.post("/vapi-webhook")
async def vapi_webhook(body: dict) -> dict:
    for event in handle_webhook(body):
        await broadcast(event)
    return {}


@app.websocket("/dashboard")
async def dashboard(ws: WebSocket) -> None:
    await ws.accept()
    _clients.add(ws)
    await ws.send_json({"type": "hello", "clients": len(_clients)})
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)


NOT_BUILT = """<!doctype html><meta charset="utf-8">
<title>Understudy — frontend not built</title>
<body style="font:16px/1.6 ui-monospace,monospace;max-width:44rem;margin:12vh auto;padding:0 6vw">
<h1 style="font-size:1.3rem">The dashboard has not been built yet.</h1>
<p>The API is running. Build the React app once, then reload:</p>
<pre style="background:#111;color:#eee;padding:1rem;overflow-x:auto">cd frontend
npm install
npm run build</pre>
<p>Or run <code>npm run dev</code> in <code>frontend/</code> for hot reload; it proxies the
API to this server.</p>
</body>"""


@app.get("/")
async def index():
    if not (DIST / "index.html").exists():
        return HTMLResponse(NOT_BUILT, status_code=503)
    return FileResponse(DIST / "index.html")


if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")
