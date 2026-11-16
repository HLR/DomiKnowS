from types import SimpleNamespace
import pytest
from domiknows.generation.dfa.productive import SemanticSearchLimitExceeded
from .epoch_recovery import retry_eai_search, canonical_eai_record


class Policy:
    def __init__(self, needed):
        self.needed = needed
        self._reachability = SimpleNamespace(max_state_expansions=10000)

    def allowed_tokens(self, state, remaining_steps=None):
        if self._reachability.max_state_expansions < self.needed:
            raise SemanticSearchLimitExceeded('unknown; never substitute an approximate set')
        return frozenset({1, 3})


def test_exact_budget_retry_preserves_allowed_set_and_restores_runtime():
    policy = Policy(20000); calls = []
    runtime = SimpleNamespace(dfa_for=lambda domain, context: policy); original = runtime.dfa_for
    with retry_eai_search(runtime, calls):
        dfa = runtime.dfa_for('eai', {})
        assert dfa.allowed_tokens('state', remaining_steps=5) == {1, 3}
        assert dfa.allowed_tokens('state', remaining_steps=5) == {1, 3}
        assert len(calls) == 1 and calls[0]['hard_constraints_relaxed'] is False
    assert runtime.dfa_for is original and policy._reachability.max_state_expansions == 10000


def test_exhaustion_remains_unknown_and_never_relaxes_constraints():
    policy = Policy(2000000); runtime = SimpleNamespace(dfa_for=lambda domain, context: policy)
    with pytest.raises(SemanticSearchLimitExceeded):
        with retry_eai_search(runtime, [], maximum=100000):
            runtime.dfa_for('eai', {}).allowed_tokens('state', remaining_steps=4)
    assert policy._reachability.max_state_expansions == 10000


def test_other_domains_keep_original_cap_and_failure():
    policy = Policy(20000); runtime = SimpleNamespace(dfa_for=lambda domain, context: policy); records = []
    with retry_eai_search(runtime, records):
        with pytest.raises(SemanticSearchLimitExceeded): runtime.dfa_for('vlabench', {}).allowed_tokens('state')
    assert records == []


def test_eai_prefix_verification_ignores_only_set_repr_order():
    first = {'instruction': 'read', 'labels': [1, 4],
             'goal': {'is_success': 1., 'predicted_state': "{('read', 'book'), ('holds', 'book')}",
                      'initial_state': 'set()', 'gold_state': "{('read', 'book')}"}}
    second = {**first, 'goal': {**first['goal'], 'predicted_state': "{('holds', 'book'), ('read', 'book')}"}}
    assert canonical_eai_record(first) == canonical_eai_record(second)
    wrong_object = {**second, 'goal': {**second['goal'], 'predicted_state': "{('holds', 'cup'), ('read', 'book')}"}}
    assert canonical_eai_record(first) != canonical_eai_record(wrong_object)
    assert first['goal']['predicted_state'].startswith("{('read'")


def test_native_reachability_retry_completes_exact_search_and_rejects_dead_branch():
    from domiknows.generation.dfa.productive import BoundedReachability
    final = 11001
    def step(state, label):
        if label == 1: return -1
        return state+1 if 0 <= state < final else None
    search = BoundedReachability(step, lambda state: state == final,
        lambda state: (0, 1) if state == 0 else ((0,) if 0 < state < final else ()))
    policy = SimpleNamespace(_reachability=search, allowed_tokens=search.allowed_tokens)
    runtime = SimpleNamespace(dfa_for=lambda domain, context: policy); events = []
    with pytest.raises(SemanticSearchLimitExceeded): policy.allowed_tokens(0, remaining_steps=11000)
    with retry_eai_search(runtime, events):
        assert runtime.dfa_for('eai', {}).allowed_tokens(0, remaining_steps=11000) == {0}
    assert events[0]['new_cap'] == 100000 and search.max_state_expansions == 10000
