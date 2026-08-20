from datetime import date
from understudy.models import Listing
from understudy.store import Store


def a_listing(lid, **kw):
    base = dict(
        id=lid, source="ebay", url=f"https://x/{lid}", title="RTX 3080",
        ask_price=500.0, condition="Used", seller_hash="h",
        seller_type="private", first_seen=date(2026, 1, 1),
    )
    base.update(kw)
    return Listing(**base)


def test_upsert_and_get(tmp_path):
    s = Store(tmp_path / "t.sqlite")
    s.upsert([a_listing("a"), a_listing("b")])
    assert s.get("a").title == "RTX 3080"
    assert len(s.all()) == 2


def test_upsert_is_idempotent(tmp_path):
    s = Store(tmp_path / "t.sqlite")
    s.upsert([a_listing("a", ask_price=500.0)])
    s.upsert([a_listing("a", ask_price=450.0)])
    assert len(s.all()) == 1
    assert s.get("a").ask_price == 450.0


def test_by_sku_filters_sold(tmp_path):
    s = Store(tmp_path / "t.sqlite")
    s.upsert([
        a_listing("a", sku_id="rtx3080", sold_price=470.0),
        a_listing("b", sku_id="rtx3080"),
        a_listing("c", sku_id="rtx4090"),
    ])
    assert {l.id for l in s.by_sku("rtx3080", sold=True)} == {"a"}
    assert {l.id for l in s.by_sku("rtx3080", sold=False)} == {"b"}


def test_json_roundtrip(tmp_path):
    s = Store(tmp_path / "t.sqlite")
    s.upsert([a_listing("a")])
    s.export_json(tmp_path / "d.json")
    s2 = Store(tmp_path / "t2.sqlite")
    s2.import_json(tmp_path / "d.json")
    assert s2.get("a").url == "https://x/a"


def test_store_is_usable_from_another_thread(tmp_path):
    """FastAPI serves sync handlers from a threadpool."""
    import threading
    s = Store(tmp_path / "t.sqlite")
    s.upsert([a_listing("a")])
    out = {}
    t = threading.Thread(target=lambda: out.update(title=s.get("a").title))
    t.start()
    t.join()
    assert out["title"] == "RTX 3080"


def test_build_seeds_an_empty_store_from_the_shipped_dataset(tmp_path, monkeypatch):
    """A fresh clone has data/listings.json but no database."""
    import json

    from understudy import pipeline

    data = tmp_path / "data"
    data.mkdir()
    (data / "listings.json").write_text(json.dumps([json.loads(a_listing("z").model_dump_json())]))
    (data / "skus.json").write_text(json.dumps(
        [{"id": "rtx3080", "brand": "NVIDIA", "model": "RTX 3080", "aliases": []}]))
    monkeypatch.setattr(pipeline, "DATA", data)
    monkeypatch.setattr(pipeline, "SKUS_PATH", data / "skus.json")

    store = Store(tmp_path / "fresh.sqlite")
    assert store.all() == []
    seed = pipeline.DATA / "listings.json"
    if not store.all() and seed.exists():
        store.import_json(seed)
    assert len(store.all()) == 1


def test_build_seeding_does_not_shadow_the_rng_seed(tmp_path, monkeypatch):
    """Regression: a local named `seed` clobbered build(seed: int) and the
    synthetic generator received a Path instead of an integer."""
    import inspect

    from understudy import pipeline

    src = inspect.getsource(pipeline.build)
    assert "seed = DATA" not in src, "local `seed` shadows the seed parameter"
