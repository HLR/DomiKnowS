"""Vocabulary abstraction for constrained token generation.

A :class:`TokenVocabulary` defines the *compact label space* used throughout
the generation pipeline.  It partitions all possible tokenizer output ids into:

- **Known tokens** — a fixed, ordered set of surface-form strings (e.g.
  ``["yes", "no", "<eos>"]``).  Each maps to a unique integer label
  ``0 … len(tokens)-1``.
- **Other** — every tokenizer id that falls outside the known set.  This
  class of ids maps to the single ``other_label`` (= ``len(tokens)``).

The full label space is therefore ``tokens + [other_token]`` with size
``len(tokens) + 1``.

If a *tokenizer* is provided, the class also supports bidirectional mapping
between surface-form token strings and raw tokenizer ids.  Without a
tokenizer only the label ↔ string direction is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TokenVocabulary:
    """Maps human token strings to compact integer generation labels.

    The compact label space is ``vocab + [other_token]``:
    - Labels ``0 … len(tokens)-1`` correspond to the known token strings.
    - Label ``len(tokens)`` (= :attr:`other_label`) represents every
      tokenizer id that is not in the known vocabulary.

    The class is a frozen dataclass, so instances are immutable and hashable.
    Manual ``__init__`` is used instead of the generated one to enforce
    invariants before the fields are frozen via ``object.__setattr__``.

    Attributes:
        tokens: Ordered tuple of known token surface strings.
        eos_token: The end-of-sequence token string; must be in *tokens*.
        other_token: Reserved string representing the out-of-vocabulary class.
            Defaults to ``"_other"``.
        tokenizer: Optional HuggingFace-compatible tokenizer.  Required for
            any operation that maps between strings and raw token ids.

    Example::

        vocab = TokenVocabulary(["yes", "no", "<eos>"], eos_token="<eos>")
        assert vocab.label_for_token("yes") == 0
        assert vocab.eos_label == 2
        assert vocab.other_label == 3   # index after the three known tokens
    """

    tokens: tuple[str, ...]
    eos_token: str
    other_token: str = "_other"
    tokenizer: object | None = None

    def __init__(
        self,
        tokens: Sequence[str],
        eos_token: str,
        other_token: str = "_other",
        tokenizer: object | None = None,
    ):
        """Construct a :class:`TokenVocabulary` with validation.

        Args:
            tokens: Ordered sequence of known token surface strings.
                The order determines label assignment (index = label id).
            eos_token: End-of-sequence token; must already appear in *tokens*.
            other_token: Reserved string for the out-of-vocabulary catch-all
                class.  Must not be in *tokens*.
            tokenizer: Optional tokenizer used for label ↔ token-id mapping.

        Raises:
            ValueError: If *eos_token* is not in *tokens*, *other_token* is in
                *tokens*, or *tokens* contains duplicates.
        """
        if eos_token not in tokens:
            raise ValueError(f"eos_token {eos_token!r} must be present in tokens")
        if other_token in tokens:
            raise ValueError(f"other_token {other_token!r} is reserved and must not be in tokens")
        if len(set(tokens)) != len(tokens):
            raise ValueError("tokens must be unique")

        # Use object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "tokens", tuple(tokens))
        object.__setattr__(self, "eos_token", eos_token)
        object.__setattr__(self, "other_token", other_token)
        object.__setattr__(self, "tokenizer", tokenizer)

    @property
    def labels(self) -> tuple[str, ...]:
        """All label strings in order: known tokens followed by *other_token*."""
        return self.tokens + (self.other_token,)

    @property
    def label_count(self) -> int:
        """Total number of labels in the compact label space (= ``len(tokens) + 1``)."""
        return len(self.labels)

    @property
    def eos_label(self) -> int:
        """Integer label id of the end-of-sequence token."""
        return self.label_for_token(self.eos_token)

    @property
    def other_label(self) -> int:
        """Integer label id of the out-of-vocabulary catch-all class.

        Always equal to ``len(tokens)`` — the last index in the label space.
        """
        return len(self.tokens)

    @property
    def alphabet(self) -> set[int]:
        """Full set of valid label ids ``{0, 1, …, label_count - 1}``."""
        return set(range(self.label_count))

    def label_for_token(self, token: str) -> int:
        """Return the integer label id for a token surface string.

        The special *other_token* string maps to :attr:`other_label` without
        needing to be in :attr:`tokens`.

        Args:
            token: A known token string or :attr:`other_token`.

        Returns:
            Integer label id in ``[0, label_count)``.

        Raises:
            KeyError: If *token* is not in the vocabulary and is not
                :attr:`other_token`.
        """
        if token == self.other_token:
            return self.other_label
        try:
            return self.tokens.index(token)
        except ValueError as exc:
            raise KeyError(f"token {token!r} is not in the generation vocabulary") from exc

    def token_for_label(self, label: int) -> str:
        """Return the token surface string for an integer label id.

        Args:
            label: Integer label id in ``[0, label_count)``.

        Returns:
            The corresponding token string, or :attr:`other_token` for
            :attr:`other_label`.

        Raises:
            IndexError: If *label* is out of range.
        """
        return self.labels[int(label)]

    def token_id_for_token(self, token: str) -> int:
        """Return the single tokenizer id for a known token surface string.

        Encodes *token* with the attached tokenizer and verifies that it
        produces exactly one id (i.e. the token is not split by the tokenizer).

        Args:
            token: A known token surface string.

        Returns:
            The raw tokenizer token id as an integer.

        Raises:
            ValueError: If no tokenizer is attached, or if *token* encodes to
                anything other than exactly one token id.
        """
        if self.tokenizer is None:
            raise ValueError("tokenizer is required to map token strings to token ids")
        encoded = self.tokenizer.encode(token)
        if len(encoded) != 1:
            raise ValueError(f"token {token!r} does not encode to exactly one token id: {encoded}")
        return int(encoded[0])

    @property
    def known_token_ids(self) -> tuple[int, ...]:
        """Raw tokenizer ids for every known token, in label order.

        Raises:
            ValueError: If no tokenizer is attached.
        """
        return tuple(self.token_id_for_token(token) for token in self.tokens)

    def token_id_for_label(self, label: int) -> int:
        """Return the raw tokenizer id for a known label.

        Args:
            label: Integer label id for a *known* token (not :attr:`other_label`).

        Returns:
            The raw tokenizer token id.

        Raises:
            ValueError: If *label* is :attr:`other_label` (no single id exists)
                or if no tokenizer is attached.
        """
        label = int(label)
        if label == self.other_label:
            raise ValueError("_other does not map to a single token id")
        return self.token_id_for_token(self.tokens[label])

    def label_for_token_id(self, token_id: int) -> int:
        """Map a raw tokenizer id to its compact label.

        Scans the known token ids linearly.  If *token_id* matches one of
        them, the corresponding label is returned; otherwise
        :attr:`other_label` is returned.

        Args:
            token_id: A raw tokenizer output id.

        Returns:
            Integer label id in ``[0, label_count)``.

        Raises:
            ValueError: If no tokenizer is attached.
        """
        if self.tokenizer is None:
            raise ValueError("tokenizer is required to map token ids to labels")
        token_id = int(token_id)
        # Linear scan over known ids; vocabulary is typically small.
        for label, known_id in enumerate(self.known_token_ids):
            if token_id == known_id:
                return label
        # No match — classify as out-of-vocabulary.
        return self.other_label

    def labels_for_token_ids(self, token_ids: Iterable[int]) -> list[int]:
        """Map an iterable of raw tokenizer ids to compact label ids.

        Convenience wrapper that calls :meth:`label_for_token_id` for each id.

        Args:
            token_ids: Iterable of raw tokenizer output ids.

        Returns:
            List of integer label ids in the same order as *token_ids*.

        Raises:
            ValueError: If no tokenizer is attached.
        """
        return [self.label_for_token_id(token_id) for token_id in token_ids]
