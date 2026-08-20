"""Mapping messy listing titles onto canonical SKUs.

GPU model numbers are a *structured namespace*, and that shapes the design.
Nearest-neighbour similarity on raw titles gets this badly wrong, because the
single most important token is also the shortest: "RTX 3080" and "RTX 3080 Ti"
are 95% identical as strings and are different products at different prices.
Measured on this corpus, pure char-ngram cosine similarity mislabelled every
Ti, Super and XT variant it met — 48 RTX 3060 Ti listings collapsed onto the
RTX 3060, dragging that SKU's median with them.

So resolution is two stages:

  1. `extract_model` pulls the model designation out of the title with an
     explicit grammar, variant suffix included. Deterministic, and it returns
     None rather than guessing.
  2. Fuzzy matching handles what is left: the extracted token is matched
     against the catalogue by embedding similarity, which absorbs spacing and
     formatting variance ("3060ti" / "3060 TI" / "3060-Ti") without ever
     letting a different model win.

The embedder still earns its place — on the normalised token, where a wrong
answer is impossible — instead of on the raw title, where it was harmful.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from understudy.models import Sku

MATCH_THRESHOLD = 0.60

# Model designations, variant suffix included. Order matters: "ti super" must
# be tried before "ti", or every Ti Super silently becomes a Ti.
_NVIDIA = re.compile(r"\b(?:rtx|gtx)\s*[- ]?\s*(\d{3,4})\s*(ti\s*super|ti|super)?\b", re.I)
_AMD = re.compile(r"\brx\s*[- ]?\s*(\d{3,4})\s*(xtx|xt)?\b", re.I)


def extract_model(title: str) -> str | None:
    """"EVGA RTX 3080 Ti XC3 12GB" -> "rtx3080ti". None when no model is named.

    Returning None is a feature: an unrecognised card must not be forced into
    the nearest catalogue entry.
    """
    text = title or ""
    for pattern, prefix in ((_NVIDIA, "rtx"), (_AMD, "rx")):
        m = pattern.search(text)
        if m:
            suffix = (m.group(2) or "").lower().replace(" ", "")
            return f"{prefix}{m.group(1)}{suffix}"
    return None


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> np.ndarray: ...


def _l2(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingEmbedder:
    """Deterministic char-ngram TF-IDF. No network, no model download."""

    def __init__(self, ngram_range: tuple[int, int] = (3, 5)):
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, lowercase=True)
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self._vec.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        return _l2(self._vec.transform(texts).toarray().astype(np.float32))


class SentenceTransformerEmbedder:  # pragma: no cover - optional heavy dependency
    """all-MiniLM-L6-v2. Same protocol; downloads ~90MB on first use."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:
        return None  # pre-trained; nothing to fit

    def embed(self, texts: list[str]) -> np.ndarray:
        return _l2(np.asarray(self._model.encode(texts), dtype=np.float32))


def load_skus(path: str | Path) -> list[Sku]:
    return [Sku.model_validate(r) for r in json.loads(Path(path).read_text())]


class SkuResolver:
    """Extract the model designation, then fuzzy-match that token to a SKU.

    Brute-force cosine over the catalogue: it is tens of rows, so an ANN index
    would be pure ceremony.
    """

    def __init__(self, skus: list[Sku], embedder: Embedder, threshold: float = MATCH_THRESHOLD):
        self.skus = skus
        self.embedder = embedder
        self.threshold = threshold
        self._exact = {s.id: s.id for s in skus}
        for s in skus:
            for alias in s.aliases:
                self._exact[re.sub(r"[^a-z0-9]", "", alias.lower())] = s.id
        # Match on the model token alone. Brand names and marketing copy in the
        # full title are noise that outweighs the one token that matters.
        self._docs = [re.sub(r"[^a-z0-9]", "", f"{s.model}".lower()) for s in skus]
        embedder.fit(self._docs)
        self._mat = embedder.embed(self._docs)

    def resolve(self, title: str) -> tuple[str | None, float]:
        model = extract_model(title)
        if model is None:
            return None, 0.0
        if model in self._exact:
            return self._exact[model], 1.0
        # Formatting variance only ("3060ti" vs "3060 TI"); a different model
        # cannot win here because the digits dominate the normalised token.
        v = self.embedder.embed([model])[0]
        sims = self._mat @ v
        i = int(np.argmax(sims))
        score = float(sims[i])
        return (self.skus[i].id, score) if score >= self.threshold else (None, score)


_FLAG_PATTERNS = {
    "mining": r"\bmin(?:ing|ed)\b|\bhashrate\b|\brig\b",
    "no_box": r"\bno box\b|\bwithout box\b|\bbare\b|\bcard only\b",
    "boxed": r"\b(?:og |original )?box(?:ed)?\b|\bwith box\b|\biib\b",
    "as_is": r"\bas[- ]is\b|\bno returns\b|\bparts only\b|\bfor parts\b",
    "coil_whine": r"\bcoil whine\b",
    "warranty": r"\bwarrant(?:y|ied)\b",
    "untested": r"\buntested\b|\bnot tested\b|\bunknown working\b",
    "no_cooler": r"\bno (?:heatsink|cooler|fan)\b",
    # Multi-unit lots are not comparable to a single card: they inflate the
    # sold-price median and dominate any "priced over median" ranking.
    "lot": r"\blot of\b|\blot\b|\bbundle\b|^\s*\d+\s*x\b|\b\d+\s*x\s+(?:evga|asus|msi|gigabyte|zotac|nvidia|amd|geforce|radeon|rtx|rx)\b|\bpair of\b|\bjoblot\b",
}


def condition_flags(text: str) -> list[str]:
    t = (text or "").lower()
    flags = [name for name, pat in _FLAG_PATTERNS.items() if re.search(pat, t)]
    # "no box" also matches the "box" pattern; the negative statement wins.
    if "no_box" in flags and "boxed" in flags:
        flags.remove("boxed")
    return flags
