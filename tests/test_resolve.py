import numpy as np
import pytest

from understudy.models import Sku
from understudy.resolve import HashingEmbedder, SkuResolver, condition_flags

SKUS = [
    Sku(id="rtx3080", brand="NVIDIA", model="RTX 3080", aliases=["geforce rtx 3080", "3080 10gb"]),
    Sku(id="rtx3070", brand="NVIDIA", model="RTX 3070", aliases=["geforce rtx 3070", "3070 8gb"]),
    Sku(id="rx6800xt", brand="AMD", model="RX 6800 XT", aliases=["radeon rx 6800 xt", "6800xt"]),
]


@pytest.fixture
def resolver():
    return SkuResolver(SKUS, HashingEmbedder())


def test_embeddings_are_normalised():
    e = HashingEmbedder()
    e.fit(["rtx 3080", "rx 6800 xt", "rtx 3070"])
    v = e.embed(["rtx 3080", "rx 6800 xt"])
    assert v.shape[0] == 2
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-6)


def test_embedder_is_deterministic():
    def go():
        e = HashingEmbedder()
        e.fit(["rtx 3080 founders edition", "rx 6800 xt"])
        return e.embed(["rtx 3080 founders edition"])
    assert np.allclose(go(), go())


@pytest.mark.parametrize("title,expected", [
    ("NVIDIA GeForce RTX 3080 FE 10GB barely used w/ og box", "rtx3080"),
    ("EVGA RTX 3070 XC3 Ultra Gaming 8GB", "rtx3070"),
    ("AMD Radeon RX 6800 XT 16GB Reference", "rx6800xt"),
    ("ASUS RTX 3080 10GB GDDR6X RTX3080-10G-ICE-V2 NO HEATSINK", "rtx3080"),
])
def test_resolves_messy_titles(resolver, title, expected):
    sku_id, score = resolver.resolve(title)
    assert sku_id == expected
    assert score > 0.0


def test_unrelated_title_falls_below_threshold(resolver):
    assert resolver.resolve("Dell OptiPlex 7040 desktop i5 8GB RAM")[0] is None


def test_condition_flags_detected():
    flags = condition_flags("RTX 3080, used for mining, no box, sold as-is, slight coil whine")
    assert set(flags) >= {"mining", "no_box", "as_is", "coil_whine"}
    assert "boxed" not in flags


def test_condition_flags_positive_signals():
    assert set(condition_flags("RTX 3080 with original box, 2 year warranty")) >= {"boxed", "warranty"}


def test_multi_unit_lots_are_flagged():
    assert "lot" in condition_flags("LOT OF 4 NVIDIA RTX 3080 10GB GPUS TESTED WORKING")
    assert "lot" in condition_flags("4 X EVGA RTX 3060 Ti XC GAMING 8GB")
    assert "lot" in condition_flags("RTX 3070 bundle with power supply")


def test_single_cards_are_not_flagged_as_lots():
    assert "lot" not in condition_flags("EVGA GeForce RTX 3080 XC3 ULTRA GAMING 10GB")
    assert "lot" not in condition_flags("ASUS ROG STRIX GeForce RTX 4070 12GB")
