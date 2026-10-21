"""Free decoding allows every label at every step and never rejects a sequence."""
from ..models import FreeDecoding


def test_free_decoding_allows_everything_and_always_accepts():
    dfa = FreeDecoding(label_count=10, eos_label=3)
    state = dfa.start_state
    blocked = FreeDecoding(label_count=10, eos_label=3, min_steps=1)
    assert 3 not in blocked.allowed_tokens(blocked.start_state) and 3 in blocked.allowed_tokens(1)
    for label in (9, 0, 3, 7):
        assert dfa.allowed_tokens(state, remaining_steps=5) == frozenset(range(10))
        state = dfa.step(state, label)
        assert state is not None and dfa.is_accepting(state)
