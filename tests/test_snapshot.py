from datetime import date

from understudy.ingest.snapshot import infer_sales, record_snapshot
from understudy.models import Listing
from understudy.store import Store


def listing(lid, ask, **kw):
    base = dict(id=lid, url=f"u{lid}", title="RTX 3080", ask_price=ask, condition="Used",
                seller_hash="h", seller_type="private", first_seen=date(2026, 1, 1))
    base.update(kw)
    return Listing(**base)


def test_first_snapshot_seeds_price_history(tmp_path):
    s = Store(tmp_path / "t.sqlite")
    record_snapshot(s, [listing("a", 500.0)], today=date(2026, 1, 1))
    stored = s.get("a")
    assert len(stored.price_history) == 1
    assert stored.price_history[0].price == 500.0


def test_repeat_snapshot_at_same_price_adds_no_duplicate_point(tmp_path):
    s = Store(tmp_path / "t.sqlite")
    record_snapshot(s, [listing("a", 500.0)], today=date(2026, 1, 1))
    record_snapshot(s, [listing("a", 500.0)], today=date(2026, 1, 5))
    assert len(s.get("a").price_history) == 1


def test_price_change_is_recorded_as_a_cut(tmp_path):
    s = Store(tmp_path / "t.sqlite")
    record_snapshot(s, [listing("a", 500.0)], today=date(2026, 1, 1))
    record_snapshot(s, [listing("a", 450.0)], today=date(2026, 1, 8))
    stored = s.get("a")
    assert len(stored.price_history) == 2
    assert stored.price_cuts == 1
    assert stored.ask_price == 450.0


def test_first_seen_is_preserved_across_snapshots(tmp_path):
    s = Store(tmp_path / "t.sqlite")
    record_snapshot(s, [listing("a", 500.0)], today=date(2026, 1, 1))
    record_snapshot(s, [listing("a", 450.0)], today=date(2026, 2, 1))
    assert s.get("a").first_seen == date(2026, 1, 1)
    assert s.get("a").days_listed(date(2026, 2, 1)) == 31


def test_disappearing_listing_is_inferred_sold_at_its_last_price(tmp_path):
    s = Store(tmp_path / "t.sqlite")
    record_snapshot(s, [listing("a", 500.0), listing("b", 400.0)], today=date(2026, 1, 1))
    record_snapshot(s, [listing("a", 480.0)], today=date(2026, 1, 8))
    sold = infer_sales(s, seen_ids={"a"}, today=date(2026, 1, 8))
    assert [l.id for l in sold] == ["b"]
    assert s.get("b").sold_price == 400.0
    assert s.get("b").sold_at == date(2026, 1, 8)


def test_already_sold_listings_are_not_reinferred(tmp_path):
    s = Store(tmp_path / "t.sqlite")
    record_snapshot(s, [listing("a", 500.0)], today=date(2026, 1, 1))
    infer_sales(s, seen_ids=set(), today=date(2026, 1, 8))
    assert infer_sales(s, seen_ids=set(), today=date(2026, 1, 15)) == []


def test_inferred_sales_are_not_marked_synthetic(tmp_path):
    s = Store(tmp_path / "t.sqlite")
    record_snapshot(s, [listing("a", 500.0)], today=date(2026, 1, 1))
    sold = infer_sales(s, seen_ids=set(), today=date(2026, 1, 8))
    assert sold[0].synthetic is False
