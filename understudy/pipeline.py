"""Wiring. The one place that knows where artefacts live and how they connect."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from understudy.llm import (
    DEFAULT_CLAUDE_MODEL,
    AnthropicLLM,
    OpenAILLM,
    StubLLM,
    default_openai_model,
)
from understudy.models import Listing, SkuStats
from understudy.resolve import HashingEmbedder, SkuResolver, condition_flags, load_skus
from understudy.store import Store
from understudy.synthetic import synthesize_sold_history
from understudy.truth import MIN_FIT_ROWS, ReservationModel, sku_stats

DATA = Path("data")
DB_PATH = DATA / "listings.sqlite"
SKUS_PATH = DATA / "skus.json"
MODEL_PATH = DATA / "model.pkl"
STATS_PATH = DATA / "stats.json"

MIN_SOLD_PER_SKU = 20


STUB_ENGINE = "rule-based stub"

# Adding a provider is a row here plus a class in llm.py. Nothing else changes:
# the simulator only ever sees the LLM protocol.
PROVIDERS: dict[str, dict] = {
    "stub": {"key": None, "label": lambda: STUB_ENGINE},
    "claude": {"key": "ANTHROPIC_API_KEY", "label": lambda: DEFAULT_CLAUDE_MODEL},
    "openai": {"key": "OPENAI_API_KEY", "label": default_openai_model},
}


def provider_available(kind: str) -> bool:
    spec = PROVIDERS.get(kind)
    if spec is None:
        return False
    return spec["key"] is None or bool(os.environ.get(spec["key"]))


def claude_available() -> bool:
    return provider_available("claude")


def engine_label(kind: str) -> str:
    """What actually generated a transcript, for display. Never guess in the UI."""
    spec = PROVIDERS.get(kind)
    return spec["label"]() if spec else kind


def make_judge(kind: str):
    """A judge reports through a tool and never speaks, so it skips the
    spoken-turn follow-up that a negotiator needs."""
    if kind == "openai":
        return OpenAILLM(needs_text=False)
    return AnthropicLLM(model=DEFAULT_CLAUDE_MODEL)


def make_llm_factory(kind: str = "stub"):
    """(role, ask_price, persona, target, seed, strategy) -> LLM.

    One signature everywhere. The Claude client is built once and shared: a
    simulation spawns two agents per run, and constructing a client per agent
    would open hundreds of connections for a single sweep.
    """
    shared: list = []

    def factory(role, ask_price, persona, target, seed, strategy=None):
        if kind == "stub":
            return StubLLM(role, ask_price=ask_price, params=persona, target=target,
                           seed=seed, strategy=strategy if role == "buyer" else None)
        if not shared:
            shared.append(OpenAILLM() if kind == "openai"
                          else AnthropicLLM(model=DEFAULT_CLAUDE_MODEL))
        return shared[0]
    return factory


@dataclass
class Artifacts:
    store: Store
    model: ReservationModel
    stats: dict[str, SkuStats]

    def actives(self) -> list[Listing]:
        return [l for l in self.store.all()
                if l.sold_price is None and l.sku_id in self.stats
                and "lot" not in l.condition_flags]

    def sold(self, synthetic: bool | None = None) -> list[Listing]:
        rows = [l for l in self.store.all()
                if l.sold_price is not None and l.sku_id in self.stats]
        if synthetic is None:
            return rows
        return [l for l in rows if l.synthetic is synthetic]


def resolve_all(store: Store, skus_path: Path = SKUS_PATH) -> tuple[int, int]:
    """Assign a SKU and condition flags to every listing. Returns (matched, total)."""
    resolver = SkuResolver(load_skus(skus_path), HashingEmbedder())
    rows, matched = [], 0
    for l in store.all():
        sku_id, _score = resolver.resolve(l.title)
        flags = condition_flags(f"{l.title} {l.description or ''}")
        matched += sku_id is not None
        rows.append(l.model_copy(update={"sku_id": sku_id, "condition_flags": flags}))
    store.upsert(rows)
    return matched, len(rows)


def build(db_path: Path = DB_PATH, allow_synthetic: bool = True, seed: int = 0) -> dict:
    """Resolve SKUs, assemble sold-price truth, fit the model, save artefacts."""
    store = Store(db_path)
    # A fresh clone has the shipped dataset but no database. Seed it rather
    # than making the first command someone runs fail on an empty store.
    # Not named `seed`: that shadows this function's `seed: int` parameter and
    # the RNG then receives a Path.
    seed_file = DATA / "listings.json"
    if not store.all() and seed_file.exists():
        store.import_json(seed_file)
    matched, total = resolve_all(store)

    real_sold = [l for l in store.all() if l.sold_price is not None and not l.synthetic]
    used_synthetic = False
    if len(real_sold) < MIN_FIT_ROWS and allow_synthetic:
        actives = [l for l in store.all()
                   if l.sold_price is None and l.sku_id and "lot" not in l.condition_flags]
        store.upsert(synthesize_sold_history(actives, seed=seed))
        used_synthetic = True

    # Lots are excluded from the price statistics entirely — a 4-card lot is a
    # different product from a card, and leaving them in drags every median up.
    sold = [l for l in store.all()
            if l.sold_price is not None and l.sku_id and "lot" not in l.condition_flags]
    by_sku: dict[str, list[Listing]] = {}
    for l in sold:
        by_sku.setdefault(l.sku_id, []).append(l)

    stats = {sku: sku_stats(rows) for sku, rows in by_sku.items()
             if len(rows) >= MIN_SOLD_PER_SKU}
    if not stats:
        raise SystemExit(f"no SKU has >= {MIN_SOLD_PER_SKU} sold listings; nothing to fit")

    model = ReservationModel().fit([l for l in sold if l.sku_id in stats], stats)
    model.save(MODEL_PATH)
    STATS_PATH.write_text(json.dumps(
        {k: json.loads(v.model_dump_json()) for k, v in stats.items()}, indent=2))
    store.export_json(DATA / "listings.json")

    return {
        "listings": total,
        "sku_matched": matched,
        "skus_modelled": sorted(stats),
        "sold_rows": len(sold),
        "real_sold_rows": len(real_sold),
        "used_synthetic_sold_data": used_synthetic,
    }


def load(db_path: Path = DB_PATH) -> Artifacts:
    if not MODEL_PATH.exists() or not STATS_PATH.exists():
        raise SystemExit("artefacts missing — run `python -m understudy build` first")
    stats = {k: SkuStats.model_validate(v)
             for k, v in json.loads(STATS_PATH.read_text()).items()}
    return Artifacts(store=Store(db_path), model=ReservationModel.load(MODEL_PATH), stats=stats)
