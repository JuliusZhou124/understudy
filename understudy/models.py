"""Every type that crosses a module boundary lives here.

Deliberately logic-free apart from validators and two derived properties, so
that any module can import it without dragging in numpy/sklearn.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class PricePoint(BaseModel):
    date: date
    price: float


class Listing(BaseModel):
    id: str
    source: Literal["ebay"] = "ebay"
    url: str
    title: str
    description: str | None = None
    ask_price: float
    sold_price: float | None = None
    sold_at: date | None = None
    condition: str
    seller_hash: str
    seller_type: Literal["private", "business", "unknown"] = "unknown"
    seller_feedback: int | None = None
    first_seen: date
    price_history: list[PricePoint] = Field(default_factory=list)
    accepts_offers: bool = False
    phone: str | None = None
    sku_id: str | None = None
    condition_flags: list[str] = Field(default_factory=list)
    # True for rows produced by understudy.synthetic. Structural, not a comment:
    # every report splits real from synthetic on this field.
    synthetic: bool = False

    @computed_field
    @property
    def price_cuts(self) -> int:
        h = self.price_history
        return sum(1 for i in range(1, len(h)) if h[i].price < h[i - 1].price)

    @computed_field
    @property
    def total_drop_pct(self) -> float:
        """Drop from the peak listed price, not the first — histories can rise."""
        if not self.price_history:
            return 0.0
        peak = max(p.price for p in self.price_history)
        return max(0.0, (peak - self.ask_price) / peak) if peak else 0.0

    def days_listed(self, today: date | None = None) -> int:
        return max(0, ((today or date.today()) - self.first_seen).days)


class Sku(BaseModel):
    id: str
    brand: str
    model: str
    aliases: list[str] = Field(default_factory=list)


class SkuStats(BaseModel):
    sku_id: str
    n_sold: int
    median: float
    q25: float
    q75: float
    prices: list[float] = Field(default_factory=list)


class PersonaParams(BaseModel):
    """The latent behavioural profile of a synthetic seller."""

    reservation_ratio: float = Field(gt=0.0, le=1.0)
    concession_rate: float = Field(ge=0.0, le=1.0)
    patience: int = Field(ge=1, le=20)
    firmness: float = Field(ge=0.0, le=1.0)
    hostility: float = Field(ge=0.0, le=1.0)
    urgency: float = Field(ge=0.0, le=1.0)


class TranscriptTurn(BaseModel):
    speaker: Literal["buyer", "seller"]
    text: str


class SimResult(BaseModel):
    listing_id: str
    strategy: str
    seed: int
    outcome: Literal["deal", "no_deal", "walk", "timeout"]
    final_price: float | None = None
    offers: list[float] = Field(default_factory=list)
    turns: int = 0
    transcript: list[TranscriptTurn] = Field(default_factory=list)
    persona: PersonaParams | None = None


class Packet(BaseModel):
    """The buyer agent's entire fact source. If it isn't here, it doesn't know it."""

    listing_id: str
    headline: str
    facts: list[str]
    ask_price: float
    target: float
    opening: float
    walk_away: float


class Recommendation(BaseModel):
    listing_id: str
    p_deal: float
    expected_settle: float
    settle_p10: float
    settle_p90: float
    opening_offer: float
    packet: Packet
