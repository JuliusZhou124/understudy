"""PII stripping at the ingest boundary.

Nothing downstream of this module — not the store, not the packet, not the
LLM — is allowed to see a seller's handle, phone, email, or street address.
Redaction happens once, here, on the way in.
"""

from __future__ import annotations

import hashlib
import re

from understudy.models import Listing

# Ordered most-specific-first: emails contain digits that the phone pattern
# would otherwise chew into.
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_ADDRESS = re.compile(
    r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b",
    re.IGNORECASE,
)
_PHONE = re.compile(r"(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")


def hash_seller(handle: str) -> str:
    """Stable pseudonym for a seller. One-way; the handle is never stored."""
    return hashlib.sha256(handle.encode()).hexdigest()[:16]


def strip_pii(text: str | None) -> str | None:
    if text is None:
        return None
    text = _EMAIL.sub("[email]", text)
    text = _ADDRESS.sub("[address]", text)
    return _PHONE.sub("[phone]", text)


def redact_listing(listing: Listing) -> Listing:
    data = listing.model_dump()
    data["title"] = strip_pii(listing.title) or ""
    data["description"] = strip_pii(listing.description)
    # A private individual's phone number has no legitimate use in this system:
    # they are never dialable, so we do not keep it at all.
    if listing.seller_type == "private":
        data["phone"] = None
    return Listing.model_validate(data)
