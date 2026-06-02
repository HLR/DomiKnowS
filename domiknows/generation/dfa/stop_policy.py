"""StopPolicy: stopping criteria for DFA-guided decoders.

Replaces the historic ``max_new_tokens`` hard cap with a small dataclass that
collects every stopping criterion the decoders care about.  A policy is
validated up-front via :func:`validate_stop_policy` (must declare at least one
safety signal) and then evaluated each step via :func:`should_stop_decoding`.

The decoder entry points in :mod:`domiknows.generation.dfa.decoder` (and the
wrappers in :mod:`domiknows.generation.applications.{inference,adapters,hybrid}`)
all accept a ``stop_policy`` keyword.  When callers still pass the legacy
``max_new_tokens=N`` argument, :func:`stop_policy_from_legacy` maps it to a
``StopPolicy(max_steps=N)`` so old code keeps working without changes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class DecodeProgress:
    """Runtime snapshot passed to :attr:`StopPolicy.external_stop_fn` and used by :func:`should_stop_decoding`."""

    step_index: int
    elapsed_seconds: float
    dfa_state: Any
    prompt_token_count: int
    generated_token_ids: Sequence[int] = field(default_factory=tuple)
    generated_labels: Sequence[int] = field(default_factory=tuple)
    accepted: bool = False
    eos_emitted: bool = False
    # ``last_dfa_state_change_step`` is updated each time the DFA state moves to
    # a fresh state.  ``StopPolicy.max_stagnant_steps`` uses it to abort when
    # the decoder gets stuck in a non-progressing cycle.
    last_dfa_state_change_step: int = 0


@dataclass(frozen=True)
class StopPolicy:
    """Termination and safety policy for DFA-guided decoding.

    Exactly one of the following must hold (validated by
    :func:`validate_stop_policy`):

    * ``max_steps is not None``
    * ``timeout_seconds is not None``
    * ``external_stop_fn is not None``
    * ``stop_on_eos`` is ``True``
    * ``stop_on_eos_if_accepting`` is ``True``
    * ``stop_on_accepting_state`` is ``True``

    Any of these alone is a sufficient safety signal — the validator only
    rejects the degenerate policy that has no stop condition at all.
    """

    # Bounded-decoding cap (None ⇒ unbounded).
    max_steps: int | None = None
    # Wall-clock cap measured against ``DecodeProgress.elapsed_seconds``.
    timeout_seconds: float | None = None

    # EOS / accepting semantics.  The default mirrors the legacy behaviour:
    # stop only when EOS is emitted *and* the DFA is accepting.
    stop_on_eos_if_accepting: bool = True
    stop_on_eos: bool = False
    stop_on_accepting_state: bool = False

    # User callback; return True to stop.
    external_stop_fn: Callable[[DecodeProgress], bool] | None = None

    # Safety net for DFAs that may cycle without progress.
    max_stagnant_steps: int | None = None


def validate_stop_policy(policy: StopPolicy) -> None:
    """Raise ``ValueError`` unless *policy* declares at least one stop condition."""
    if not isinstance(policy, StopPolicy):
        raise TypeError(f"expected StopPolicy, got {type(policy).__name__}")
    if policy.max_steps is not None and policy.max_steps < 0:
        raise ValueError("StopPolicy.max_steps must be non-negative when set")
    if policy.timeout_seconds is not None and policy.timeout_seconds < 0:
        raise ValueError("StopPolicy.timeout_seconds must be non-negative when set")
    if policy.max_stagnant_steps is not None and policy.max_stagnant_steps < 0:
        raise ValueError("StopPolicy.max_stagnant_steps must be non-negative when set")
    signals = (
        policy.max_steps is not None,
        policy.timeout_seconds is not None,
        policy.external_stop_fn is not None,
        bool(policy.stop_on_eos),
        bool(policy.stop_on_eos_if_accepting),
        bool(policy.stop_on_accepting_state),
    )
    if not any(signals):
        raise ValueError(
            "StopPolicy declares no stopping criterion; set one of "
            "max_steps / timeout_seconds / external_stop_fn / stop_on_eos / "
            "stop_on_eos_if_accepting / stop_on_accepting_state."
        )


def should_stop_decoding(policy: StopPolicy, progress: DecodeProgress) -> bool:
    """Return True when *policy* says decoding should halt before the next step.

    Evaluated *before* the next token is sampled — i.e. ``progress.step_index``
    is the number of tokens already generated.  EOS-based stops are *not*
    re-checked here; the decoders themselves emit the EOS short-circuit after
    they've actually consumed the EOS token.  This function only handles
    safety / budget / external-callback stops.
    """
    if policy.max_steps is not None and progress.step_index >= policy.max_steps:
        return True
    if policy.timeout_seconds is not None and progress.elapsed_seconds >= policy.timeout_seconds:
        return True
    if policy.max_stagnant_steps is not None:
        stagnant = progress.step_index - progress.last_dfa_state_change_step
        if stagnant >= policy.max_stagnant_steps:
            return True
    if policy.external_stop_fn is not None and policy.external_stop_fn(progress):
        return True
    return False


def should_stop_on_token(policy: StopPolicy, *, eos_emitted: bool, accepted: bool) -> bool:
    """Return True when *policy* says decoding should halt *because* of the
    last emitted token (an EOS / accepting-state signal)."""
    if policy.stop_on_eos and eos_emitted:
        return True
    if policy.stop_on_eos_if_accepting and eos_emitted and accepted:
        return True
    if policy.stop_on_accepting_state and accepted:
        return True
    return False


def stop_policy_from_legacy(
    *,
    max_new_tokens: int | None,
    stop_policy: StopPolicy | None,
) -> StopPolicy:
    """Resolve the (max_new_tokens, stop_policy) pair into a single StopPolicy.

    Helper used by every decoder entry point that supports the deprecated
    ``max_new_tokens=`` kwarg:

    * if ``stop_policy`` is supplied, use it as-is (after validation).
    * else, treat ``max_new_tokens`` as the legacy hard cap and wrap it in a
      ``StopPolicy(max_steps=max_new_tokens, stop_on_eos_if_accepting=True)``
      — the historic semantics.
    * if both are ``None``, raise ``ValueError`` (the validator would also
      reject a `StopPolicy()` with no signals; we surface the call-site issue
      with a clearer message).
    """
    if stop_policy is not None and max_new_tokens is not None:
        raise ValueError(
            "Pass either stop_policy=... or max_new_tokens=... (legacy), not both."
        )
    if stop_policy is None:
        if max_new_tokens is None:
            raise ValueError(
                "Decoder requires a stop_policy=... (preferred) or max_new_tokens=... (legacy)."
            )
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        stop_policy = StopPolicy(
            max_steps=int(max_new_tokens),
            stop_on_eos_if_accepting=True,
        )
    validate_stop_policy(stop_policy)
    return stop_policy


def remaining_steps_for(policy: StopPolicy, step_index: int) -> int | None:
    """Return the ``remaining_steps`` budget passed to :meth:`DFA.allowed_tokens`.

    When the policy is unbounded (``max_steps is None``), returns ``None`` so
    :meth:`DFA.allowed_tokens` skips the reachability call entirely.  When
    bounded, returns ``max_steps - step_index`` clamped at 0.
    """
    if policy.max_steps is None:
        return None
    return max(0, int(policy.max_steps) - int(step_index))


def make_progress_tracker():
    """Return a small closure that timestamps decode progress.

    Use as::

        update_progress = make_progress_tracker()
        ...
        progress = update_progress(step_index=..., dfa_state=..., ...)

    The closure records ``start_time`` once and produces fresh
    :class:`DecodeProgress` snapshots on each call.
    """
    start_time = time.perf_counter()
    last_state = [None]
    last_change_step = [0]

    def _update(
        *,
        step_index: int,
        dfa_state: Any,
        prompt_token_count: int,
        generated_token_ids: Sequence[int] = (),
        generated_labels: Sequence[int] = (),
        accepted: bool = False,
        eos_emitted: bool = False,
    ) -> DecodeProgress:
        if last_state[0] is None or dfa_state != last_state[0]:
            last_change_step[0] = step_index
            last_state[0] = dfa_state
        return DecodeProgress(
            step_index=step_index,
            elapsed_seconds=time.perf_counter() - start_time,
            dfa_state=dfa_state,
            prompt_token_count=prompt_token_count,
            generated_token_ids=tuple(generated_token_ids),
            generated_labels=tuple(generated_labels),
            accepted=accepted,
            eos_emitted=eos_emitted,
            last_dfa_state_change_step=last_change_step[0],
        )

    return _update


__all__ = [
    "DecodeProgress",
    "StopPolicy",
    "make_progress_tracker",
    "remaining_steps_for",
    "should_stop_decoding",
    "should_stop_on_token",
    "stop_policy_from_legacy",
    "validate_stop_policy",
]
