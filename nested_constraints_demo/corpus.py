"""Labelled test corpus for the nested-constraints demo DFA.

Each bucket maps a human-readable explanation to a tuple of symbol sequences
(strings).  The accompanying :func:`verify_acceptance` helper compiles each
sequence to compact labels via the bundle and reports whether the DFA's
verdict matches the bucket's expected status.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Bucket:
    name: str
    expected_accepted: bool
    rule_hint: str
    sequences: tuple[tuple[str, ...], ...]


VALID = Bucket(
    name="VALID",
    expected_accepted=True,
    rule_hint="rules all satisfied: contains A xor C, at most one B, ended with END",
    sequences=(
        ("A", "END"),
        ("C", "END"),
        ("A", "B", "END"),
        ("A", "B", "A", "END"),
        ("C", "C", "END"),
        ("A", "A", "B", "A", "END"),
    ),
)

INVALID_A_AND_C = Bucket(
    name="INVALID_A_AND_C",
    expected_accepted=False,
    rule_hint="violates not(existsA and existsC) -- sequence contains both A and C",
    sequences=(
        ("A", "C", "END"),
        ("A", "B", "C", "END"),
        ("C", "A", "END"),
    ),
)

INVALID_TWO_OR_MORE_B = Bucket(
    name="INVALID_TWO_OR_MORE_B",
    expected_accepted=False,
    rule_hint="violates atMostAL(B, 1) -- too many B's",
    sequences=(
        ("B", "B", "END"),
        ("A", "B", "B", "END"),
        ("B", "A", "B", "END"),
    ),
)

INVALID_AFTER_END = Bucket(
    name="INVALID_AFTER_END",
    expected_accepted=False,
    rule_hint="violates EOS-closure -- token appears after END",
    sequences=(
        ("A", "END", "A"),
        ("END", "A"),
        ("A", "END", "B"),
    ),
)

INVALID_MISSING_A_AND_C = Bucket(
    name="INVALID_MISSING_A_AND_C",
    expected_accepted=False,
    rule_hint="violates orL(existsA, existsC) -- neither A nor C appears",
    sequences=(
        ("END",),
        ("B", "END"),
    ),
)

INVALID_FORBIDDEN_D = Bucket(
    name="INVALID_FORBIDDEN_D",
    expected_accepted=False,
    rule_hint="violates atMostAL(D, 0) -- D is forbidden",
    sequences=(
        ("D", "END"),
        ("A", "D", "END"),
        ("C", "D", "END"),
    ),
)


BUCKETS: tuple[Bucket, ...] = (
    VALID,
    INVALID_A_AND_C,
    INVALID_TWO_OR_MORE_B,
    INVALID_AFTER_END,
    INVALID_MISSING_A_AND_C,
    INVALID_FORBIDDEN_D,
)


@dataclass(frozen=True)
class AcceptanceRecord:
    bucket: str
    symbols: tuple[str, ...]
    accepted: bool
    expected_accepted: bool
    rule_hint: str

    @property
    def passes(self) -> bool:
        return self.accepted == self.expected_accepted


def verify_acceptance(dfa, bundle, buckets: Sequence[Bucket] = BUCKETS) -> tuple[AcceptanceRecord, ...]:
    """Run *dfa* over every sequence in *buckets* and return the verdicts."""
    records: list[AcceptanceRecord] = []
    for bucket in buckets:
        for symbols in bucket.sequences:
            labels = [bundle.vocabulary.label_for_token(symbol) for symbol in symbols]
            records.append(
                AcceptanceRecord(
                    bucket=bucket.name,
                    symbols=tuple(symbols),
                    accepted=bool(dfa.accepts(labels)),
                    expected_accepted=bucket.expected_accepted,
                    rule_hint=bucket.rule_hint,
                )
            )
    return tuple(records)
