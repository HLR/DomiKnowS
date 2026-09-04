# Plan: Remove `max_new_tokens` From DFA Decoding

**Status:** verified against the current `decoder.py` / `core.py` / `applications/` layout (last verified 2026-06-02).  Line numbers have been replaced with symbolic anchors so the plan does not drift as the surrounding code changes.

## Goal
Replace fixed token-count stopping (`max_new_tokens`) with policy-driven stopping so decoding can run without a mandatory length cap, while preserving safety and DFA correctness.

## Why This Is Feasible
The DFA core already supports unbounded reachability and allowed-token enumeration when the budget is omitted.  In `domiknows/generation/dfa/core.py`:

- `DFA.can_reach_accepting(state, max_steps=None)` — the parameter is named **`max_steps`**; `None` means "no depth limit" on the BFS over reachable, non-dead states.
- `DFA.allowed_tokens(state, remaining_steps=None)` — the parameter is named **`remaining_steps`**.  When it is `None`, the function *skips the reachability call entirely* and returns every non-dead successor of *state*, so no reachability budget needs to be plumbed in unbounded mode.

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

Candidate decoder entry points to update (all in `domiknows/generation/dfa/decoder.py`):
- `constrained_greedy_decode`
- `constrained_beam_search_decode`
- `constrained_sample_decode`
- `constrained_label_greedy_decode`
- `constrained_label_beam_search_decode`
- `constrained_label_sample_decode`

### 1b. Application-layer wrappers also passing `max_new_tokens`

The decoder entry points above are wrapped at the application layer.  These callers must be migrated to `StopPolicy` (with the same `max_new_tokens=` deprecation shim — see §8) so the public API stays consistent.

- `domiknows/generation/applications/inference.py` — the three top-level `greedy_label_inference` / `beam_label_inference` / `sample_label_inference` helpers each take and forward `max_new_tokens`, and the file **duplicates** the `_validate_common(max_new_tokens)` helper.  The duplicate must be removed when the decoder's validator is replaced (or both must point at the new `validate_stop_policy`).
- `domiknows/generation/applications/adapters.py` — `HuggingFaceGenerationAdapter.greedy / beam / sample` methods each accept `max_new_tokens` and pass it to the corresponding decoder.
- `domiknows/generation/applications/hybrid.py` — `HybridController.greedy_decode`, the candidate-rendering path, and the rerank-with-decoder path all carry a `max_new_tokens` argument.
- Compact-label generation heads expose `head.greedy_label_inference(..., max_new_tokens=...)` via a `**kwargs` passthrough on `CompactLabelGenerationHead.greedy_label_inference` (in `domiknows/generation/learners/common/base.py`).  Every `Tasks/` demo reaches the decoder through this surface.
- Doc references in `domiknows/generation/learners/README.md` (the `remaining_steps`-budget paragraph) and `domiknows/generation/learners/README_learning.md` (`max_new_tokens=8` examples) describe the old semantics and must be updated.

### 2. Remove hard dependency on `remaining_steps` from decode loop budget
Currently, decode loops compute:
- `remaining_steps = max_new_tokens - step_idx`
and pass this to `dfa.allowed_tokens(...)`.

Refactor to:
- unbounded mode: `dfa.allowed_tokens(state, remaining_steps=None)`
- bounded mode: pass remaining budget only if `max_steps` is set in stop policy.

Locations (grep `'remaining_steps = max_new_tokens - step_idx'` in `domiknows/generation/dfa/decoder.py` — present in the body of every entry point):
- `constrained_greedy_decode`
- `constrained_beam_search_decode`
- `constrained_sample_decode`
- `constrained_label_greedy_decode`
- `constrained_label_beam_search_decode`
- `constrained_label_sample_decode`

### 3. Replace fixed-range loops with policy-driven loops
Current pattern:
- `for step_idx in range(max_new_tokens):`

New pattern:
- `while not stop_policy.should_stop(...):`

Keep an internal `step_idx` counter for metrics and optional safeguards.

Loop sites (grep `'for step_idx in range(max_new_tokens):'` in `domiknows/generation/dfa/decoder.py` — one per entry point):
- `constrained_greedy_decode`
- `constrained_beam_search_decode`
- `constrained_sample_decode`
- `constrained_label_greedy_decode`
- `constrained_label_beam_search_decode`
- `constrained_label_sample_decode`

### 4. Keep mandatory safety guardrails
Even in "unbounded" mode, do not allow unconstrained infinite runtime.

Validation requires *at least one* of these conditions to hold:

- `max_steps is not None`, or
- `timeout_seconds is not None`, or
- `external_stop_fn is not None`, or
- `stop_on_eos` / `stop_on_eos_if_accepting` is `True` **and** the DFA's
  alphabet includes the EOS label.

The last condition lets pure EOS-driven stopping be a single, sufficient
safety signal — matching today's behaviour where a positive `max_new_tokens`
together with EOS termination is the only safety net.  Callers who want
to opt out (e.g. unit tests that exhaustively explore the DFA) should pass
an `external_stop_fn` that asserts deliberately.

Migration impact:

- The transitional shim (§8) maps `max_new_tokens=N` to `StopPolicy(max_steps=N)`, so existing call sites that pass a positive integer still satisfy the guard automatically.
- A caller who previously passed `max_new_tokens=None` (illegal today — `_validate_common` requires `>= 0`) would fail the new validator unless one of the other safety signals is set.  This stays consistent with the current behaviour.

### 5. Make EOS stop behavior configurable
Current behavior stops early only on:
- `EOS` and `dfa.is_accepting(state)`

Retain as default, but support alternatives:
- EOS only
- accepting-state only
- EOS and accepting (current default)

EOS checks currently in (grep `'next_label == eos_label and dfa.is_accepting(state)'` and `'next_token_id == eos_token_id'` in `domiknows/generation/dfa/decoder.py`):
- `constrained_greedy_decode`
- `constrained_beam_search_decode`
- `constrained_sample_decode`
- `constrained_label_greedy_decode`
- `constrained_label_beam_search_decode`
- `constrained_label_sample_decode`

### 6. Replace shared validator
Current validator:
- `_validate_common(max_new_tokens)`

Replace with stop-policy validation:
- verifies at least one safety stop mechanism (per §4)
- validates timeout/limits

Locations:
- `domiknows/generation/dfa/decoder.py` — top-level `_validate_common`.
- `domiknows/generation/applications/inference.py` — **duplicate** `_validate_common` defined locally.  Drop the duplicate or rewire it to call the new `validate_stop_policy`; otherwise the two validators will drift.

### 7. Beam-specific completion rules
Without `max_new_tokens`, beam search needs explicit stop criteria:
- all beams finished
- no expandable beams remain
- stop policy triggered
- optional minimum finished beams threshold

Preserve the existing per-beam plumbing.  Both `constrained_beam_search_decode` and `constrained_label_beam_search_decode` already track `BeamCandidate.finished` (set when `next_label == eos_label and dfa.is_accepting(next_state)`) and use an "all-finished short-circuit" inside the outer loop to break early when every candidate is finished.  The StopPolicy migration must keep this short-circuit — without it, beam decoding will pay the full `step_idx` budget even when every beam terminated several steps earlier.

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

Apply the same transition pattern to all six decode entry points.

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

## Verification

After the refactor lands, the following must all be green in *both* legacy `max_new_tokens=N` mode (through the deprecation shim) and the new `stop_policy=StopPolicy(...)` mode:

1. **Decoder-level tests** —
   ```
   pytest test_regr/generation/test_decoder.py -q
   ```
   Every `constrained_*_decode` test must pass, including any newly added test that exercises the `stop_policy` kwarg directly.
2. **Head-level passthrough tests** —
   ```
   pytest test_regr/generation/test_compact_heads.py \
          test_regr/generation/test_automata_heads.py \
          test_regr/generation/test_prompt_conditioned_automata_heads.py -q
   ```
   These exercise `head.greedy_label_inference(..., max_new_tokens=N)` (the `**kwargs` passthrough on `CompactLabelGenerationHead`).  Both the legacy call-shape and the new `stop_policy=` call-shape must work.
3. **End-to-end demos** —
   ```
   uv run --project Tasks/real_hmm_pmd_learning python Tasks/real_hmm_pmd_learning/run_demo.py
   uv run --project Tasks/real_hmm_pmd_learning python -m Tasks.nested_constraints_demo.run_demo
   ```
   Both demos must produce byte-identical greedy and DFA-constrained-greedy outputs in transitional `max_new_tokens=` mode (i.e. nothing changed from the user's perspective after the deprecation shim is in place).
4. **New unbounded-EOS regression test** — add `test_stop_policy_unbounded_eos_only` that constructs a tiny DFA accepting `EOS`, calls `constrained_greedy_decode` with `stop_policy=StopPolicy(stop_on_eos_if_accepting=True)` and **no** `max_steps`, and verifies the loop terminates on the first EOS without a length cap.
5. **Application-layer round-trip** — every wrapper listed in §1b (`applications/inference.py`, `applications/adapters.py`, `applications/hybrid.py`) must accept both `max_new_tokens=` and `stop_policy=`, and at least one test must exercise each side.
