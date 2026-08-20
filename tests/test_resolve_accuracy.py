"""Accuracy of SKU resolution on hand-reviewed labels.

The fixture is 60 real listing titles, stratified toward the hard cases (Ti,
Super, XT, workstation and mobile parts), each read and confirmed by hand.
Correct includes *declining to match*: a Radeon VII or a mobile RTX 3060M must
resolve to None rather than to the nearest catalogue entry.

This test exists because the number originally reported for this component —
"97.6%" — was coverage, not accuracy: the share of titles assigned any SKU,
which says nothing about whether the SKU was right.
"""

import json
from pathlib import Path

import pytest

from understudy.resolve import HashingEmbedder, SkuResolver, load_skus

LABELS = Path(__file__).parent / "fixtures" / "sku_labels.json"
SKUS = Path(__file__).parent.parent / "data" / "skus.json"


@pytest.fixture(scope="module")
def labelled():
    if not LABELS.exists():
        pytest.skip("labels not generated")
    return json.loads(LABELS.read_text())


@pytest.fixture(scope="module")
def resolver():
    return SkuResolver(load_skus(SKUS), HashingEmbedder())


def test_resolution_accuracy_is_at_least_95_percent(labelled, resolver):
    wrong = [(r["title"], r["sku_id"], resolver.resolve(r["title"])[0])
             for r in labelled if resolver.resolve(r["title"])[0] != r["sku_id"]]
    accuracy = 1 - len(wrong) / len(labelled)
    print(f"\nSKU resolution accuracy: {accuracy:.1%} on n={len(labelled)}")
    for title, want, got in wrong[:5]:
        print(f"  {title[:56]!r}: want {want}, got {got}")
    assert accuracy >= 0.95


def test_variant_suffixes_are_never_collapsed(resolver):
    """The failure that made this test necessary: 48 RTX 3060 Ti listings
    resolving to the RTX 3060 and dragging its median with them."""
    pairs = [
        ("GIGABYTE GeForce RTX 3060 Ti VISION OC 8GB", "rtx3060ti"),
        ("NVIDIA GeForce RTX 3060 12GB", "rtx3060"),
        ("ASUS ROG Strix GeForce RTX 3080 Ti OC 12GB", "rtx3080ti"),
        ("GIGABYTE GeForce RTX 3080 VISION OC 10GB", "rtx3080"),
        ("ASUS TUF Gaming GeForce RTX 4070 Ti Super 16GB", "rtx4070tisuper"),
        ("MSI GeForce RTX 4070 Ti SUPRIM X 12GB", "rtx4070ti"),
        ("MSI Radeon RX 6800 GAMING X Trio 16GB", "rx6800"),
        ("AMD Radeon RX 6800 XT 16GB GDDR6", "rx6800xt"),
    ]
    for title, expected in pairs:
        assert resolver.resolve(title)[0] == expected, title


def test_unknown_cards_are_declined_not_guessed(resolver):
    for title in [
        "Mint AMD Radeon VII Reference GPU 16GB HBM2",
        "Dell OptiPlex 7040 desktop i5 8GB RAM",
        "Corsair RM850x power supply 850W",
    ]:
        assert resolver.resolve(title)[0] is None, title
