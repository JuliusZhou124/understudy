"""Strategy A/B testing.

Arms are compared **paired**: the same listings and the same seeds produce the
same personas in every arm, so a difference between arms is a difference in the
strategy and not in who they happened to negotiate against. Every reported
delta carries a paired bootstrap 95% CI, because at these sample sizes a bare
mean difference is not evidence of anything.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class ArmResult(BaseModel):
    strategy: str
    n: int
    deal_rate: float
    mean_excess_savings: float
    mean_discount_pct: float
    mean_turns: float
    # Populated only when a judge is supplied; None means "not measured",
    # which is a different claim from 0.0 and must not be displayed as one.
    judged: int = 0
    violation_rate: float | None = None
    fabrication_rate: float | None = None


class EvalReport(BaseModel):
    arms: list[ArmResult]
    deltas: dict[str, list[float]] = Field(default_factory=dict)  # name -> [delta, lo, hi]
    baseline: str | None = None


def paired_bootstrap(a: list[float], b: list[float], iters: int = 5000,
                     seed: int = 0) -> tuple[float, float, float]:
    """Mean paired difference (a - b) with a 95% bootstrap CI."""
    if len(a) != len(b):
        raise ValueError(f"paired arms must be the same length, got {len(a)} and {len(b)}")
    if not a:
        raise ValueError("paired_bootstrap needs at least one pair")
    x = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(iters, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def ab_test(listings, stats_by_sku, model, llm_factory, strategies, n: int = 10,
            seed: int = 0, judge_llm=None, judge_sample: int = 20) -> EvalReport:
    """Compare strategies. Supply `judge_llm` to also grade behaviour.

    Grading costs one model call per transcript, so at most `judge_sample`
    transcripts per arm are graded — enough for a rate, cheap enough to run.
    """
    from understudy.decide import build_packet
    from understudy.judge import grade_transcript
    from understudy.sim import simulate_listing

    per_arm: dict[str, list[float]] = {}
    arms: list[ArmResult] = []

    for strat in strategies:
        excess, discounts, turns, deals = [], [], [], []
        grades = []
        for l in listings:
            if not l.sku_id or l.sku_id not in stats_by_sku:
                continue
            stats = stats_by_sku[l.sku_id]
            packet = build_packet(l, stats, model)
            for r in simulate_listing(l, stats, model, strat, llm_factory, seed=seed, n=n):
                deals.append(1.0 if r.outcome == "deal" else 0.0)
                turns.append(float(r.turns))
                # No deal counts as paying the ask — otherwise a strategy that
                # only ever closes its easiest listing looks like the winner.
                price = r.final_price if r.final_price is not None else l.ask_price
                excess.append(stats.median - price)
                discounts.append((l.ask_price - price) / l.ask_price)
                if judge_llm is not None and len(grades) < judge_sample:
                    grades.append(grade_transcript(r, packet.facts, judge_llm))
        per_arm[strat.name] = excess
        arms.append(ArmResult(
            strategy=strat.name, n=len(excess),
            deal_rate=float(np.mean(deals)) if deals else 0.0,
            mean_excess_savings=float(np.mean(excess)) if excess else 0.0,
            mean_discount_pct=float(np.mean(discounts)) if discounts else 0.0,
            mean_turns=float(np.mean(turns)) if turns else 0.0,
            judged=len(grades),
            violation_rate=(float(np.mean([g.violations > 0 for g in grades]))
                            if grades else None),
            fabrication_rate=(float(np.mean([g.fabricated_fact for g in grades]))
                              if grades else None),
        ))

    if not arms:
        return EvalReport(arms=[])

    baseline = arms[0].strategy
    deltas: dict[str, list[float]] = {}
    for arm in arms[1:]:
        if len(per_arm[arm.strategy]) == len(per_arm[baseline]) and per_arm[baseline]:
            d, lo, hi = paired_bootstrap(per_arm[arm.strategy], per_arm[baseline], seed=seed)
            deltas[f"{arm.strategy}_vs_{baseline}"] = [d, lo, hi]

    return EvalReport(arms=sorted(arms, key=lambda a: -a.mean_excess_savings),
                      deltas=deltas, baseline=baseline)
