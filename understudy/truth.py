"""Ground truth from sold listings, and the model fitted to it.

For a sold listing we observe both the asking price and the actual settle, so
`settle / ask` is a directly supervised label — the seller's revealed
reservation ratio. `ReservationModel` fits that label from features that are
all observable on a *live* listing, which is what makes it usable to
parameterise a persona for an item that has not sold yet.

Quantile regression rather than a point estimate: a persona needs a
distribution to sample from, and the spread is the interesting part.
"""

from __future__ import annotations

import pickle
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from understudy.models import Listing, SkuStats
from understudy.resolve import condition_flags

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
MIN_FIT_ROWS = 20

FEATURE_ORDER = [
    "ask_ratio",
    "price_cuts",
    "total_drop_pct",
    "days_listed",
    "is_business",
    "seller_feedback_log",
    "accepts_offers",
    "condition_penalty",
    "n_sold_log",
]

_NEGATIVE_FLAGS = {"mining", "as_is", "untested", "no_box", "coil_whine", "no_cooler"}
_POSITIVE_FLAGS = {"boxed", "warranty"}


def sku_stats(sold: list[Listing]) -> SkuStats:
    prices = sorted(l.sold_price for l in sold if l.sold_price is not None)
    if not prices:
        raise ValueError("sku_stats needs at least one sold listing")
    arr = np.asarray(prices, dtype=float)
    return SkuStats(
        sku_id=sold[0].sku_id or "unknown",
        n_sold=len(prices),
        median=float(np.median(arr)),
        q25=float(np.percentile(arr, 25)),
        q75=float(np.percentile(arr, 75)),
        prices=list(prices),
    )


def features(listing: Listing, stats: SkuStats, today: date | None = None) -> dict[str, float]:
    """Every feature here must be observable on an unsold listing — no leakage."""
    text = f"{listing.title} {listing.description or ''}"
    flags = set(listing.condition_flags) | set(condition_flags(text))
    penalty = len(flags & _NEGATIVE_FLAGS) - len(flags & _POSITIVE_FLAGS)
    return {
        "ask_ratio": listing.ask_price / stats.median if stats.median else 1.0,
        "price_cuts": float(listing.price_cuts),
        "total_drop_pct": float(listing.total_drop_pct),
        "days_listed": float(listing.days_listed(today)),
        "is_business": 1.0 if listing.seller_type == "business" else 0.0,
        "seller_feedback_log": float(np.log1p(listing.seller_feedback or 0)),
        "accepts_offers": 1.0 if listing.accepts_offers else 0.0,
        "condition_penalty": float(penalty),
        "n_sold_log": float(np.log1p(stats.n_sold)),
    }


def _vec(feats: dict[str, float]) -> np.ndarray:
    return np.asarray([feats[k] for k in FEATURE_ORDER], dtype=float)


class ReservationModel:
    """Predicts the distribution of settle/ask ratio from observable features."""

    def __init__(self, quantiles: tuple[float, ...] = QUANTILES):
        self.quantiles = quantiles
        self.models: dict[float, GradientBoostingRegressor] = {}

    def fit(self, sold: list[Listing], stats_by_sku: dict[str, SkuStats]) -> "ReservationModel":
        X, y = [], []
        for l in sold:
            if l.sold_price is None or l.ask_price <= 0:
                continue
            if not l.sku_id or l.sku_id not in stats_by_sku:
                continue
            X.append(_vec(features(l, stats_by_sku[l.sku_id], today=l.sold_at)))
            y.append(min(1.0, l.sold_price / l.ask_price))
        if len(X) < MIN_FIT_ROWS:
            raise ValueError(f"need >={MIN_FIT_ROWS} sold listings to fit, got {len(X)}")
        Xa, ya = np.asarray(X), np.asarray(y)
        for q in self.quantiles:
            m = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=200,
                                          max_depth=3, random_state=0)
            m.fit(Xa, ya)
            self.models[q] = m
        return self

    def predict_quantiles(self, feats: dict[str, float]) -> dict[float, float]:
        x = _vec(feats).reshape(1, -1)
        raw = {q: float(np.clip(self.models[q].predict(x)[0], 0.05, 1.0)) for q in self.quantiles}
        # Independently-fitted quantiles can cross; enforce monotonicity.
        running = -np.inf
        for q in sorted(raw):
            running = max(running, raw[q])
            raw[q] = running
        return raw

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str | Path) -> "ReservationModel":
        return pickle.loads(Path(path).read_bytes())
