"""Bounded concatenations of declared patterns without enumerating sentences."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PatternProgress:
    count: int
    last: str | None
    used: frozenset
    remaining: frozenset
    complete: bool = False

    def __len__(self):
        return self.count

    def __iter__(self):
        return iter(self.used)

    def __getitem__(self, index):
        if index != -1 or not self.count:
            raise IndexError(index)
        return self.last


@dataclass(frozen=True)
class BoundedPatternLanguage:
    patterns: tuple
    maximum: int
    roots: tuple | None = None
    allow_empty: bool = False
    repeat: bool = True

    @property
    def start(self):
        return PatternProgress(0, None, frozenset(), frozenset(self.patterns if self.roots is None else self.roots), self.allow_empty)

    def advance(self, state, symbol):
        if state.count >= self.maximum:
            return None
        tails = {p[1:] for p in state.remaining if p and p[0] == symbol}
        if not tails:
            return None
        complete = () in tails
        tails.discard(())
        if complete and self.repeat:
            tails.update(self.patterns)
        return PatternProgress(state.count + 1, symbol, state.used | {symbol}, frozenset(tails), complete)

    def __contains__(self, sequence):
        state = self.start
        for symbol in sequence:
            state = self.advance(state, symbol)
            if state is None:
                return False
        return state.complete
