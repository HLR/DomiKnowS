"""Domain-independent retention checks for measured validation scores."""
from math import isfinite


def retains_validation_scores(candidate, baseline, *, maximum_regression=0.0):
    """Require every known baseline score to remain available and nonregressing.

    Scores use a higher-is-better convention. Unknown baseline values are not
    evidence; an entirely unknown baseline cannot establish retention.
    """
    if not isfinite(maximum_regression) or maximum_regression < 0:
        raise ValueError("maximum_regression must be finite and nonnegative")
    checked = False
    for name, previous in baseline.items():
        if previous is None:
            continue
        current = candidate.get(name)
        if isinstance(previous, bool) or not isfinite(previous):
            raise ValueError(f"invalid baseline validation score {name!r}")
        checked = True
        if current is None or isinstance(current, bool) or not isfinite(current):
            return False
        if current + maximum_regression + 1e-9 < previous:
            return False
    return checked
