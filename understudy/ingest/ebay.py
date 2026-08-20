"""eBay search-results parsing.

Split deliberately into a **pure** parser (`parse_search_html`) and a **thin**
live fetcher (`fetch_search`). The parser is the part with logic and the part
that breaks when eBay reskins their markup, so it is driven entirely by saved
HTML fixtures and fully testable offline.

Markup note (captured 2026-08-20): results are `<li class="s-card"
data-listingid=...>` containing `a.s-card__link` and a series of
`span.su-styled-text` cells. An older `li.s-item` layout is still served to
some sessions, so both are handled.
"""

from __future__ import annotations

import hashlib
import re
import urllib.parse
from datetime import date

from understudy.ingest.redact import hash_seller, redact_listing
from understudy.models import Listing

BASE = "https://www.ebay.com/sch/i.html"

# eBay 403s a fresh context's first navigation and CAPTCHA-walls some views;
# a page that never hydrated has no item links at all.
_MIN_ITEM_LINKS = 20


def search_url(query: str, *, sold: bool = False, page: int = 1) -> str:
    params: dict[str, object] = {"_nkw": query, "_ipg": 60}
    if page > 1:
        params["_pgn"] = page
    if sold:
        params |= {"LH_Sold": 1, "LH_Complete": 1}
    return f"{BASE}?{urllib.parse.urlencode(params, quote_via=urllib.parse.quote_plus)}"


def is_real_results_page(html: str) -> bool:
    """False for CAPTCHA interstitials and un-hydrated shells."""
    return "icaptcha" not in html and html.count("ebay.com/itm/") >= _MIN_ITEM_LINKS


_CARD_SPLIT = re.compile(r'(?=<li class="s-(?:card|item)\b)')
_LISTING_ID = re.compile(r'data-listingid="(\d+)"')
_ITM_HREF = re.compile(r'href="(https://[^"]*ebay\.com/itm/[^"]*)"')
_SPAN = re.compile(r'<span class="su-styled-text[^"]*"[^>]*>(.*?)</span>', re.S)
_TITLE_BLOCK = re.compile(r'class="s-(?:card__title|item__title)"[^>]*>(.*?)</div>', re.S)
_TAG = re.compile(r"<[^>]+>")
_PRICE = re.compile(r"\$\s?([\d,]+(?:\.\d{2})?)")
_SELLER = re.compile(r"^(\S+)\s+([\d.]+)%\s+positive\s+\(([\d.]+)\s*([KkMm]?)\)")
# Screen-reader text eBay appends inside the title anchor.
_TITLE_NOISE = re.compile(r"\s*Opens in a new window or tab\.?", re.IGNORECASE)
_CONDITION_WORDS = ("Pre-Owned", "Brand New", "New (Other)", "New", "Open box",
                    "Refurbished", "Parts Only", "For parts")


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", _TAG.sub(" ", fragment)).strip()


def _feedback_count(number: str, suffix: str) -> int:
    scale = {"k": 1_000, "m": 1_000_000}.get(suffix.lower(), 1)
    return int(float(number) * scale)


def _seller_type(seller: str, title: str, cells: list[str]) -> str:
    """eBay does not label this. Business-ish signals only; default to private.

    Getting this wrong in the private direction is safe (private sellers are
    never dialable); getting it wrong the other way is not, so the default
    is the restrictive one.
    """
    blob = f"{seller} {title} {' '.join(cells)}".lower()
    if re.search(r"\b(refurb|outlet|wholesale|liquidat|llc|inc\b|store|shop|tech|computers|electronics)\b", blob):
        return "business"
    return "private"


def _parse_card(chunk: str, *, sold: bool, today: date) -> Listing | None:
    href_match = _ITM_HREF.search(chunk)
    if not href_match:
        return None
    url = href_match.group(1).split("?")[0]

    cells = [_text(m.group(1)) for m in _SPAN.finditer(chunk)]
    cells = [c for c in cells if c]
    if not cells:
        return None

    title_match = _TITLE_BLOCK.search(chunk)
    title = _text(title_match.group(1)) if title_match else cells[0]
    title = _TITLE_NOISE.sub("", title.removeprefix("New Listing")).strip(" -·")
    if not title or title.lower().startswith("shop on ebay"):
        return None

    price = next(
        (float(m.group(1).replace(",", ""))
         for c in cells if (m := _PRICE.match(c))),
        None,
    )
    if price is None or price <= 0:
        return None

    condition = next(
        (c.rstrip(" ·") for c in cells
         if any(c.startswith(w) for w in _CONDITION_WORDS)),
        "Unspecified",
    )

    seller_raw, feedback = "", None
    for c in cells:
        if (m := _SELLER.match(c)):
            seller_raw = m.group(1)
            feedback = _feedback_count(m.group(3), m.group(4))
            break

    listing_id = (_LISTING_ID.search(chunk) or [None, None])[1] if _LISTING_ID.search(chunk) else None
    stable_id = hashlib.sha256((listing_id or url).encode()).hexdigest()[:16]

    # Subtitle cells (brand / chipset / memory / shipping) are the only free
    # text we keep, and they go through redaction like everything else.
    detail_cells = [c for c in cells[1:] if not _PRICE.match(c) and not _SELLER.match(c)]

    listing = Listing(
        id=stable_id,
        url=url,
        title=title,
        description=" · ".join(detail_cells[:6]) or None,
        ask_price=price,
        sold_price=price if sold else None,
        sold_at=today if sold else None,
        condition=condition,
        seller_hash=hash_seller(seller_raw or url),
        seller_type=_seller_type(seller_raw, title, detail_cells),
        seller_feedback=feedback,
        first_seen=today,
        accepts_offers=any("best offer" in c.lower() for c in cells),
    )
    return redact_listing(listing)


def parse_search_html(html: str, *, sold: bool, today: date | None = None) -> list[Listing]:
    today = today or date.today()
    out: list[Listing] = []
    seen: set[str] = set()
    for chunk in _CARD_SPLIT.split(html)[1:]:
        listing = _parse_card(chunk, sold=sold, today=today)
        if listing and listing.id not in seen:
            seen.add(listing.id)
            out.append(listing)
    return out


# ---------------------------------------------------------------------------
# Live fetching. Everything below touches the network and is excluded from the
# default test run; the parser above is the part under test.
# ---------------------------------------------------------------------------

_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def _page_html(page, url: str, tries: int = 4) -> str | None:  # pragma: no cover - live
    """Fetch one results page, retrying past CAPTCHA interstitials."""
    import time

    for attempt in range(tries):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            page.wait_for_timeout(2_500 + attempt * 2_000)
            html = page.content()
            if is_real_results_page(html):
                return html
        except Exception:
            pass
        time.sleep(3 + attempt * 3)
    return None


def fetch_search(query: str, *, sold: bool = False, pages: int = 1,
                 headless: bool = False) -> list[Listing]:  # pragma: no cover - live
    """Scrape live search results.

    eBay 403s the first navigation of a fresh browser context, so a throwaway
    warm-up request is issued before the real ones. Sold/completed search is
    additionally CAPTCHA-walled and frequently returns nothing — callers must
    handle an empty list.
    """
    from playwright.sync_api import sync_playwright

    listings: list[Listing] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=headless, args=["--disable-blink-features=AutomationControlled"]
        )
        ctx = browser.new_context(user_agent=_UA, locale="en-US",
                                  viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        try:  # warm-up: the first navigation is always rejected
            page.goto(f"{BASE}?_nkw=warmup", wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_timeout(1_500)
        except Exception:
            pass
        for n in range(1, pages + 1):
            html = _page_html(page, search_url(query, sold=sold, page=n))
            if html is None:
                break
            listings += parse_search_html(html, sold=sold)
        browser.close()
    return listings


def capture(query: str, *, sold: bool, out: str, headless: bool = False) -> bool:  # pragma: no cover - live
    """Save one results page to disk as a parser fixture. True if it was real."""
    from pathlib import Path

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=headless, args=["--disable-blink-features=AutomationControlled"]
        )
        ctx = browser.new_context(user_agent=_UA, locale="en-US",
                                  viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        try:
            page.goto(f"{BASE}?_nkw=warmup", wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_timeout(1_500)
        except Exception:
            pass
        html = _page_html(page, search_url(query, sold=sold))
        browser.close()
    if html is None:
        return False
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(html)
    return True
