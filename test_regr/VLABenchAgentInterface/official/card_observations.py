"""Playing cards: one source of readable properties for ``selection_observations.py``.

The selection formulas know nothing about cards. They ask whether the chosen tag has the requested
property, and how many group members precede it. This module holds the card vocabulary that turns an
instruction into such a request ("poker jack of spades", "the 3rd largest poker"). Another readable
property, such as a printed label, would be one more source of the same requests.

A reading is a tuple whose first element orders the objects: a card reads ``(rank, suit)``.
"""
from __future__ import annotations

import re

CARD_NOUNS = ("poker", "card")
CARD_RANKS = {**{str(n): n for n in range(2, 11)}, "jack": 11, "queen": 12, "king": 13, "ace": 14}
_NOUN = "|".join(CARD_NOUNS)
_FACE = re.compile(rf"\b(?:{_NOUN})\s+(?P<rank>\d+|jack|queen|king|ace)\s+of\s+(?P<suit>spades|clubs|diamonds|hearts)\b")


def card_request(text, ordinal_pattern, ordinal):
    """The request a card instruction names, or None. ``text`` is normalized lower-case."""
    if match := _FACE.search(text):
        rank = CARD_RANKS.get(match["rank"])
        return None if rank is None else {"kind": "property", "noun": CARD_NOUNS[0], "value": (rank, match["suit"])}
    by_value = re.compile(rf"\bthe\s+(?:(?P<ordinal>{ordinal_pattern})\s+)?(?P<which>largest|biggest|smallest)\s+(?P<noun>{_NOUN})s?\b")
    if match := by_value.search(text):
        return {"kind": "value", "noun": match["noun"], "rank": ordinal(match["ordinal"]) if match["ordinal"] else 1,
                "start": "smallest" if match["which"] == "smallest" else "largest"}
    return None
