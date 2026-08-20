"""`python -m understudy <command>`.

Every command defaults to `--llm stub`, so the whole thing runs with no API
keys. `--llm claude` switches the negotiators to Claude and costs money.
"""

from __future__ import annotations

import argparse
import json
from datetime import date

from understudy import pipeline
from understudy.calibrate import calibrate
from understudy.decide import recommend
from understudy.evaluate import ab_test
from understudy.strategies import STRATEGIES


def _fmt(x: float | None, prefix: str = "$") -> str:
    return "-" if x is None else f"{prefix}{x:,.0f}"


def cmd_ingest(args) -> None:
    from understudy.ingest.ebay import fetch_search
    from understudy.ingest.snapshot import infer_sales, record_snapshot

    store = pipeline.Store(pipeline.DB_PATH)
    today = date.today()
    seen: set[str] = set()
    for query in [q.strip() for q in args.queries.split(",") if q.strip()]:
        got = fetch_search(query, sold=args.sold, pages=args.pages)
        seen |= {l.id for l in got}
        record_snapshot(store, got, today=today)
        print(f"{query:<20} {'sold' if args.sold else 'active':<7} {len(got):>4}")
    if args.infer_sales and seen:
        sold = infer_sales(store, seen_ids=seen, today=today)
        print(f"inferred {len(sold)} sale(s) from listings that disappeared")
    print(f"store now holds {len(store.all())} listings")


def cmd_build(args) -> None:
    report = pipeline.build(allow_synthetic=not args.no_synthetic, seed=args.seed)
    print(json.dumps(report, indent=2))
    if report["used_synthetic_sold_data"]:
        print("\n*** Sold-price data is SYNTHETIC (see understudy/synthetic.py). ***\n"
              "*** Calibration below tests estimator recovery, not the real market. ***")


def cmd_sim(args) -> None:
    art = pipeline.load()
    factory = pipeline.make_llm_factory(args.llm)
    strategy = STRATEGIES[args.strategy]
    actives = art.actives()[: args.listings]
    if not actives:
        raise SystemExit("no active listings with a modelled SKU — run ingest + build")

    from understudy.sim import simulate_listing

    for listing in actives:
        stats = art.stats[listing.sku_id]
        results = simulate_listing(listing, stats, art.model, strategy, factory,
                                   seed=args.seed, n=args.n)
        deals = [r.final_price for r in results if r.final_price is not None]
        print(f"\n{listing.title[:64]}")
        print(f"  ask {_fmt(listing.ask_price)} | sold-median {_fmt(stats.median)}")
        for r in results:
            print(f"  outcome={r.outcome:<8} price={_fmt(r.final_price):>8} "
                  f"turns={r.turns:<3} floor={_fmt(listing.ask_price * r.persona.reservation_ratio)}")
        if deals:
            print(f"  -> deal rate {len(deals)/len(results):.0%}, "
                  f"median settle {_fmt(sorted(deals)[len(deals)//2])}")

    if args.transcript:
        print("\n--- sample transcript ---")
        for turn in results[0].transcript:
            print(f"{turn.speaker.upper():>6}: {turn.text}")


def cmd_recommend(args) -> None:
    art = pipeline.load()
    listing = art.store.get(args.listing_id) or next(iter(art.actives()), None)
    if listing is None:
        raise SystemExit("no such listing")
    rec = recommend(listing, art.stats[listing.sku_id], art.model,
                    pipeline.make_llm_factory(args.llm), STRATEGIES[args.strategy], n=args.n)
    print(json.dumps(json.loads(rec.model_dump_json()), indent=2))


def cmd_calibrate(args) -> None:
    art = pipeline.load()
    sold = art.sold()
    holdout = sorted(sold, key=lambda l: (l.sold_at or date.min))[-args.holdout:]
    report = calibrate(holdout, art.stats, art.model,
                       pipeline.make_llm_factory(args.llm), STRATEGIES[args.strategy], n=args.n)
    synthetic = any(l.synthetic for l in holdout)
    print(f"Calibration on {report.n} held-out sold listings "
          f"({'SYNTHETIC' if synthetic else 'REAL'} settle prices)\n")
    print(f"  simulator MAE      {_fmt(report.mae)}")
    print(f"  simulator MAPE     {report.mape:.1%}")
    print(f"  coverage@80        {report.coverage_80:.0%}   (target 80%)")
    print(f"  CRPS               {_fmt(report.crps)}")
    print(f"  baseline MAE       {_fmt(report.baseline_mae)}   (predict the SKU sold median)")
    print(f"\n  simulator {'BEATS' if report.beats_baseline else 'LOSES TO'} the baseline "
          f"on point accuracy.")


def cmd_eval(args) -> None:
    art = pipeline.load()
    listings = art.actives()[: args.listings]
    judge_llm = None
    if args.judge != "none":
        judge_llm = pipeline.make_judge(args.judge)
    report = ab_test(listings, art.stats, art.model, pipeline.make_llm_factory(args.llm),
                     [STRATEGIES[s] for s in STRATEGIES], n=args.n, seed=args.seed,
                     judge_llm=judge_llm, judge_sample=args.judge_sample)
    head = f"{'strategy':<18}{'n':>6}{'deal':>8}{'excess $':>11}{'discount':>10}{'turns':>7}"
    if judge_llm is not None:
        head += f"{'judged':>8}{'violations':>12}{'fabricated':>12}"
    print(head)
    for a in report.arms:
        row = (f"{a.strategy:<18}{a.n:>6}{a.deal_rate:>8.0%}"
               f"{a.mean_excess_savings:>11,.0f}{a.mean_discount_pct:>10.1%}{a.mean_turns:>7.1f}")
        if judge_llm is not None:
            row += (f"{a.judged:>8}"
                    f"{(a.violation_rate if a.violation_rate is not None else 0):>11.0%} "
                    f"{(a.fabrication_rate if a.fabrication_rate is not None else 0):>11.0%}")
        print(row)
    if report.deltas:
        print(f"\nPaired bootstrap vs {report.baseline} (excess savings, 95% CI):")
        for name, (d, lo, hi) in report.deltas.items():
            sig = "" if lo <= 0 <= hi else "  *"
            print(f"  {name:<40} {d:>+8.1f}  [{lo:>+7.1f}, {hi:>+7.1f}]{sig}")
        print("\n  * = CI excludes zero")


def cmd_drift(args) -> None:
    a = json.loads(open(args.from_snapshot).read())
    b = json.loads(open(args.to_snapshot).read())
    print(f"{'sku':<12}{'median A':>11}{'median B':>11}{'shift':>10}")
    for sku in sorted(set(a) | set(b)):
        ma, mb = a.get(sku, {}).get("median"), b.get(sku, {}).get("median")
        if ma is None or mb is None:
            print(f"{sku:<12}{_fmt(ma):>11}{_fmt(mb):>11}{'new/gone':>10}")
            continue
        print(f"{sku:<12}{ma:>11,.0f}{mb:>11,.0f}{(mb - ma) / ma:>9.1%}")


def cmd_models(args) -> None:
    """List what a configured key can actually reach. Beats guessing at model ids."""
    import os

    if args.provider == "openai":
        import openai

        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set")
        client = openai.OpenAI(base_url=os.environ.get("OPENAI_BASE_URL") or None)
        for m in sorted(client.models.list(), key=lambda m: m.id):
            print(m.id)
    else:
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set")
        for m in anthropic.Anthropic().models.list():
            print(m.id)


def cmd_serve(args) -> None:
    import uvicorn

    from understudy.api import DIST

    if not (DIST / "index.html").exists():
        print("note: dashboard not built yet — run `cd frontend && npm install && npm run build`")
    print(f"dashboard: http://{args.host}:{args.port}")
    uvicorn.run("understudy.api:app", host=args.host, port=args.port, reload=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="understudy",
                                description="Negotiation agent validated against fitted seller personas")
    sub = p.add_subparsers(dest="command", required=True)

    i = sub.add_parser("ingest", help="scrape eBay search results into the store")
    i.add_argument("--queries", default="rtx 3080,rtx 3070,rtx 3060,rtx 4070,rx 6800 xt")
    i.add_argument("--pages", type=int, default=2)
    i.add_argument("--sold", action="store_true", help="fetch sold/completed (usually CAPTCHA-walled)")
    i.add_argument("--infer-sales", action="store_true",
                   help="mark listings that vanished since the last snapshot as sold")
    i.set_defaults(func=cmd_ingest)

    b = sub.add_parser("build", help="resolve SKUs, assemble truth, fit the model")
    b.add_argument("--no-synthetic", action="store_true",
                   help="fail rather than fall back to synthetic sold data")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("sim", help="simulate negotiations on active listings")
    s.add_argument("--n", type=int, default=5)
    s.add_argument("--listings", type=int, default=3)
    s.add_argument("--strategy", default="two_offer_close", choices=list(STRATEGIES))
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--llm", default="stub", choices=["stub", "claude", "openai"])
    s.add_argument("--transcript", action="store_true")
    s.set_defaults(func=cmd_sim)

    r = sub.add_parser("recommend", help="advice for one listing")
    r.add_argument("--listing-id", default="")
    r.add_argument("--n", type=int, default=20)
    r.add_argument("--strategy", default="two_offer_close", choices=list(STRATEGIES))
    r.add_argument("--llm", default="stub", choices=["stub", "claude", "openai"])
    r.set_defaults(func=cmd_recommend)

    c = sub.add_parser("calibrate", help="predicted vs actual settle price")
    c.add_argument("--n", type=int, default=20)
    c.add_argument("--holdout", type=int, default=60)
    c.add_argument("--strategy", default="two_offer_close", choices=list(STRATEGIES))
    c.add_argument("--llm", default="stub", choices=["stub", "claude", "openai"])
    c.set_defaults(func=cmd_calibrate)

    e = sub.add_parser("eval", help="A/B the buyer strategies")
    e.add_argument("--n", type=int, default=5)
    e.add_argument("--listings", type=int, default=25)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--llm", default="stub", choices=["stub", "claude", "openai"])
    e.add_argument("--judge", default="none", choices=["none", "openai", "claude"],
                   help="grade transcripts for guardrail violations (costs a call each)")
    e.add_argument("--judge-sample", type=int, default=20,
                   help="max transcripts graded per strategy")
    e.set_defaults(func=cmd_eval)

    d = sub.add_parser("drift", help="compare two stats snapshots")
    d.add_argument("--from-snapshot", required=True, dest="from_snapshot")
    d.add_argument("--to-snapshot", required=True, dest="to_snapshot")
    d.set_defaults(func=cmd_drift)

    m = sub.add_parser("models", help="list models the configured key can reach")
    m.add_argument("--provider", default="openai", choices=["openai", "claude"])
    m.set_defaults(func=cmd_models)

    v = sub.add_parser("serve", help="run the dashboard")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    v.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)

    from understudy.llm import USAGE
    if USAGE.calls:
        print(f"\napi usage: {USAGE}")
