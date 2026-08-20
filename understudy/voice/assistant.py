"""The voice agent's prompt.

One rule dominates: **never hand a TTS engine a numeral.** "$12,500" is read
aloud as "one dollar and two five zero zero". Every figure the agent may speak
is pre-rendered into English words here, and the prompt is asserted to contain
no dollar figures at all.

`spoken_usd` is ported from lowball's `assistant.ts`, which learned this the
hard way on a live demo.
"""

from __future__ import annotations

import re

from understudy.models import Packet
from understudy.strategies import Strategy, buyer_system_prompt

_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

_MONEY = re.compile(r"\$\s?([\d,]+(?:\.\d{1,2})?)")


def _under_thousand(x: int) -> str:
    parts: list[str] = []
    if x >= 100:
        parts.append(f"{_ONES[x // 100]} hundred")
        x %= 100
    if x >= 20:
        parts.append(f"{_TENS[x // 10]} {_ONES[x % 10]}".strip())
    elif x > 0:
        parts.append(_ONES[x])
    return " ".join(parts)


def spoken_usd(n: float) -> str:
    """Render a dollar amount as words. TTS mangles numerals."""
    n = int(round(n))
    if n == 0:
        return "zero dollars"
    millions, rest = divmod(n, 1_000_000)
    thousands, hundreds = divmod(rest, 1_000)
    words = " ".join(filter(None, [
        f"{_under_thousand(millions)} million" if millions else "",
        f"{_under_thousand(thousands)} thousand" if thousands else "",
        _under_thousand(hundreds) if hundreds else "",
    ]))
    return f"{words} {'dollar' if n == 1 else 'dollars'}"


def spell_prices(text: str) -> str:
    """Replace every $-figure in a block of text with its spoken form."""
    return _MONEY.sub(lambda m: spoken_usd(float(m.group(1).replace(",", ""))), text)


VOICE_RULES = """
SPEAKING NUMBERS: every number you say — prices, days, quantities — must be spoken as
English words. RIGHT: "four hundred sixty dollars", or casually "four sixty". WRONG:
bare numerals, a currency symbol followed by digits, or reading a price out digit by
digit. When you repeat a number the seller said, convert it to words too. No digits and
no dollar symbol in anything you say, ever.

PHONE MANNER: this is a live call. One or two short sentences per turn, then stop and
let them speak. Never monologue. Never read a list aloud.
"""


def voice_system_prompt(packet: Packet, strategy: Strategy) -> str:
    return spell_prices(buyer_system_prompt(packet, strategy)) + VOICE_RULES
