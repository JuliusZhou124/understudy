"""Does the simulated world match the real one?

This is the measurement the whole project exists to make. For a listing that
*did* sell, hide the sale, build a persona from features alone, simulate the
negotiation many times, and compare the predicted settle distribution to what
the item actually sold for.

Reported against a deliberately hard baseline — "just predict the SKU's sold
median". If the baseline wins on point accuracy, that is the finding, and the
README says so.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel


class CalibrationReport(BaseModel):
    n: int
    mae: float
    mape: float
    coverage_80: float
    crps: float
    baseline_mae: float

    @property
    def beats_baseline(self) -> bool:
        return self.mae < self.baseline_mae


def crps_sample(samples: np.ndarray, actual: float) -> float:
    """Continuous ranked probability score from an empirical sample.

    Rewards forecasts that are both accurate and sharp; zero only for a point
    mass on the truth.
    """
    s = np.asarray(samples, dtype=float)
    accuracy = float(np.mean(np.abs(s - actual)))
    spread = float(np.mean(np.abs(s[:, None] - s[None, :]))) / 2.0
    return accuracy - spread


def coverage(pred_samples: list[np.ndarray], actuals: list[float],
             lo: float = 0.1, hi: float = 0.9) -> float:
    """Share of actuals falling inside the predicted interval. Want ~= hi - lo."""
    if not actuals:
        return 0.0
    inside = 0
    for s, a in zip(pred_samples, actuals):
        low, high = np.percentile(s, lo * 100), np.percentile(s, hi * 100)
        inside += int(low <= a <= high)
    return inside / len(actuals)


def calibrate(listings, stats_by_sku, model, llm_factory, strategy,
              n: int = 20, seed: int = 0) -> CalibrationReport:
    from understudy.models import Listing
    from understudy.sim import simulate_listing

    preds: list[np.ndarray] = []
    actuals: list[float] = []
    baselines: list[float] = []

    for l in listings:
        if l.sold_price is None or not l.sku_id or l.sku_id not in stats_by_sku:
            continue
        stats = stats_by_sku[l.sku_id]
        # Hide the outcome: the persona must be built from observables only.
        hidden = Listing.model_validate({**l.model_dump(), "sold_price": None, "sold_at": None})
        # Evaluate as of the moment of sale: the *price* is hidden, the elapsed
        # time is not. Passing today=None here instead re-introduces the
        # train/serve skew on days_listed.
        results = simulate_listing(hidden, stats, model, strategy, llm_factory,
                                   seed=seed, n=n, today=l.sold_at)
        settles = [r.final_price for r in results if r.final_price is not None]
        preds.append(np.asarray(settles or [l.ask_price], dtype=float))
        actuals.append(float(l.sold_price))
        baselines.append(stats.median)

    if not actuals:
        raise ValueError("no held-out sold listings to calibrate on")

    med = np.asarray([float(np.median(p)) for p in preds])
    act = np.asarray(actuals)
    base = np.asarray(baselines)
    return CalibrationReport(
        n=len(actuals),
        mae=float(np.mean(np.abs(med - act))),
        mape=float(np.mean(np.abs(med - act) / np.maximum(act, 1e-9))),
        coverage_80=coverage(preds, actuals),
        crps=float(np.mean([crps_sample(p, a) for p, a in zip(preds, actuals)])),
        baseline_mae=float(np.mean(np.abs(base - act))),
    )
