# Plan: Remove `max_new_tokens` From DFA Decoding

## Goal
Replace fixed token-count stopping (`max_new_tokens`) with policy-driven stopping so decoding can run without a mandatory length cap, while preserving safety and DFA correctness.

## Why This Is Feasible
The DFA core already supports unbounded reachability checks when no step budget is supplied (`remaining_steps=None`).

- `domiknows/generation/dfa/core.py:82` (`can_reach_accepting`)
- `domiknows/generation/dfa/core.py:122` (`allowed_tokens`)

Most work is therefore in decoder control flow and API shape.

## Required Changes

### 1. Introduce a stop policy object
Replace direct `max_new_tokens` dependence with a stop-policy config.

Suggested fields:
- `max_steps: int | None` (optional hard cap)
- `stop_on_eos_if_accepting: bool`
- `stop_on_accepting_state: bool`
- `timeout_seconds: float | None`
- `external_stop_fn: Callable | None`

Candidate decoder entry points to update:
- `domiknows/generation/dfa/decoder.py:463`
- `domiknows/generation/dfa/decoder.py:568`
- `domiknows/generation/dfa/decoder.py:758`
- `domiknows/generation/dfa/decoder.py:847`
- `domiknows/generation/dfa/decoder.py:940`

### 2. Remove hard dependency on `remaining_steps` from decode loop budget
Currently, decode loops compute:
- `remaining_steps = max_new_tokens - step_idx`
and pass this to `dfa.allowed_tokens(...)`.

Refactor to:
- unbounded mode: `dfa.allowed_tokens(state, remaining_steps=None)`
- bounded mode: pass remaining budget only if `max_steps` is set in stop policy.

Locations:
- `domiknows/generation/dfa/decoder.py:516`
- `domiknows/generation/dfa/decoder.py:658`
- `domiknows/generation/dfa/decoder.py:818`
- `domiknows/generation/dfa/decoder.py:911`
- `domiknows/generation/dfa/decoder.py:1007`

### 3. Replace fixed-range loops with policy-driven loops
Current pattern:
- `for step_idx in range(max_new_tokens):`

New pattern:
- `while not stop_policy.should_stop(...):`

Keep an internal `step_idx` counter for metrics and optional safeguards.

Loop sites:
- `domiknows/generation/dfa/decoder.py:509`
- `domiknows/generation/dfa/decoder.py:646`
- `domiknows/generation/dfa/decoder.py:816`
- `domiknows/generation/dfa/decoder.py:907`

### 4. Keep mandatory safety guardrails
Even in "unbounded" mode, do not allow unconstrained infinite runtime.

Require at least one safeguard:
- `max_steps`
- `timeout_seconds`
- `external_stop_fn`

This prevents hung decode jobs when EOS never appears.

### 5. Make EOS stop behavior configurable
Current behavior stops early only on:
- `EOS` and `dfa.is_accepting(state)`

Retain as default, but support alternatives:
- EOS only
- accepting-state only
- EOS and accepting (current default)

EOS checks currently in:
- `domiknows/generation/dfa/decoder.py:533`
- `domiknows/generation/dfa/decoder.py:696`
- `domiknows/generation/dfa/decoder.py:838`
- `domiknows/generation/dfa/decoder.py:930`

### 6. Replace shared validator
Current validator:
- `_validate_common(max_new_tokens)`

Replace with stop-policy validation:
- verifies at least one safety stop mechanism
- validates timeout/limits

Location:
- `domiknows/generation/dfa/decoder.py:451`

### 7. Beam-specific completion rules
Without `max_new_tokens`, beam search needs explicit stop criteria:
- all beams finished
- no expandable beams remain
- stop policy triggered
- optional minimum finished beams threshold

### 8. Backward compatibility path
Keep API compatibility for one transition period:
- accept `max_new_tokens` as deprecated alias to `stop_policy.max_steps`
- emit deprecation warning
- remove in next major release

### 9. Documentation and examples
Update decoder docstrings and DFA README to explain:
- policy-driven stopping
- safety guards in unbounded mode
- behavior when EOS is very far away or absent

## Suggested Rollout Order
1. Add `StopPolicy` type and validation.
2. Convert greedy decode.
3. Convert sampling decode.
4. Convert beam decode.
5. Convert compact-label decode variants.
6. Add deprecation shim for `max_new_tokens`.
7. Update docs and examples.

## Acceptance Criteria
- Decoders can run with no explicit token-count cap.
- Runtime still guaranteed to terminate via policy safeguards.
- Existing callers using `max_new_tokens` continue working (with warning).
- DFA masking remains correct in bounded and unbounded policy modes.

## Draft API: `StopPolicy` and Function Signatures

### Proposed dataclass

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Any


@dataclass(frozen=True)
class StopPolicy:
	"""Termination and safety policy for DFA-guided decoding.

	At least one safety guard must be enabled:
	- max_steps is not None, or
	- timeout_seconds is not None, or
	- external_stop_fn is not None.
	"""

	# Optional hard cap on generated tokens.
	max_steps: int | None = None
	# Optional wall-clock bound for the decode loop.
	timeout_seconds: float | None = None

	# Completion semantics.
	stop_on_eos_if_accepting: bool = True
	stop_on_eos: bool = False
	stop_on_accepting_state: bool = False

	# User/system callback. Return True to stop decoding.
	# Signature receives current decode state.
	external_stop_fn: Callable[["DecodeProgress"], bool] | None = None

	# Safety fallback: if DFA cycles without progress for too long, stop.
	# (Optional but recommended in effectively unbounded mode.)
	max_stagnant_steps: int | None = None


@dataclass(frozen=True)
class DecodeProgress:
	"""Runtime progress snapshot passed to external_stop_fn."""

	step_index: int
	elapsed_seconds: float
	dfa_state: Any
	prompt_token_count: int
	generated_token_ids: Sequence[int]
	generated_labels: Sequence[int]
	accepted: bool
	eos_emitted: bool
```

### Proposed validator signatures

```python
def validate_stop_policy(policy: StopPolicy) -> None: ...


def should_stop_decoding(
	policy: StopPolicy,
	progress: DecodeProgress,
) -> bool: ...
```

### Decoder function signatures (new)

```python
def constrained_greedy_decode(
	model,
	input_ids,
	vocabulary,
	dfa,
	*,
	stop_policy: StopPolicy,
	eos_token_id: int | None = None,
	use_cache: bool = True,
):
	...


def constrained_beam_search_decode(
	model,
	input_ids,
	vocabulary,
	dfa,
	*,
	stop_policy: StopPolicy,
	eos_token_id: int | None = None,
	beam_size: int = 4,
	length_penalty: float = 1.0,
	early_stopping: bool = True,
	num_return_sequences: int = 1,
	use_cache: bool = True,
):
	...


def constrained_sample_decode(
	model,
	input_ids,
	vocabulary,
	dfa,
	*,
	stop_policy: StopPolicy,
	eos_token_id: int | None = None,
	temperature: float = 1.0,
	top_k: int | None = None,
	top_p: float | None = None,
	generator=None,
	use_cache: bool = True,
):
	...


def constrained_label_greedy_decode(
	model,
	input_ids,
	vocabulary,
	dfa,
	*,
	stop_policy: StopPolicy,
	eos_label: int | None = None,
	model_kwargs: dict | None = None,
	next_label_kwargs: dict | None = None,
):
	...


def constrained_label_beam_search_decode(
	model,
	input_ids,
	vocabulary,
	dfa,
	*,
	stop_policy: StopPolicy,
	eos_label: int | None = None,
	beam_size: int = 4,
	length_penalty: float = 1.0,
	early_stopping: bool = True,
	num_return_sequences: int = 1,
	model_kwargs: dict | None = None,
	next_label_kwargs: dict | None = None,
):
	...
```

### Backward-compatible transitional signatures (recommended)

```python
def constrained_greedy_decode(
	model,
	input_ids,
	vocabulary,
	dfa,
	max_new_tokens: int | None = None,
	*,
	stop_policy: StopPolicy | None = None,
	eos_token_id: int | None = None,
	use_cache: bool = True,
):
	"""If max_new_tokens is provided, map to StopPolicy(max_steps=max_new_tokens) and warn."""
	...
```

Apply the same transition pattern to all five decode entry points.

### Internal loop shape (target)

```python
step_idx = 0
while True:
	progress = DecodeProgress(...)
	if should_stop_decoding(policy, progress):
		break

	# 1) get logits
	# 2) compute allowed labels with remaining_steps=None unless policy.max_steps is set
	# 3) mask logits
	# 4) pick/sample next token
	# 5) advance DFA

	step_idx += 1
```
