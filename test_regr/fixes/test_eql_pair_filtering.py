"""
Regression tests for issue #372: constraining decisions in pairs via ``eqL``.

Before this fix, ``eqL``:

1. Did **not** wrap a scalar required-value in a set, so downstream code
   in ``domiknows/graph/candidates.py`` performed ``attributeValue in
   requiredValue`` on a bare string — silently turning the check into a
   substring match (``'l' in 'l2'`` → ``True``).
2. Declared ``active``, ``sampleEntries`` and ``name`` as kwargs but
   never forwarded them to ``LogicalConstrain.__init__`` — so
   ``eqL(..., active=False)`` *looked* like it worked but didn't.
3. Hardcoded ``p=100`` with no way to override.
"""
from contextlib import contextmanager

from domiknows.graph import Concept, Graph
from domiknows.graph.logicalConstrain import eqL


@contextmanager
def _make(name):
    """Build an eqL-capable Concept inside an active Graph context."""
    with Graph(name):
        yield Concept(name=f'c_{name}')


# ---------------------------------------------------------------------------
# Scalar auto-wrap (main bug fix from #372)
# ---------------------------------------------------------------------------

def test_scalar_string_is_wrapped_in_set():
    with _make('t1') as c:
        lc = eqL(c, 'text', 'l2')
        assert lc.e[2] == {'l2'}
        assert isinstance(lc.e[2], set)


def test_scalar_bool_is_wrapped():
    with _make('t2') as c:
        lc = eqL(c, 'flag', True)
        assert lc.e[2] == {True}


def test_scalar_int_is_wrapped():
    with _make('t3') as c:
        lc = eqL(c, 'idx', 5)
        assert lc.e[2] == {5}


# ---------------------------------------------------------------------------
# Container coercion — list/tuple/frozenset also accepted
# ---------------------------------------------------------------------------

def test_set_is_kept_as_set():
    with _make('t4') as c:
        lc = eqL(c, 'text', {'l1', 'l2'})
        assert lc.e[2] == {'l1', 'l2'}
        assert isinstance(lc.e[2], set)


def test_frozenset_is_coerced_to_set():
    with _make('t5') as c:
        lc = eqL(c, 'text', frozenset({'l1', 'l2'}))
        assert isinstance(lc.e[2], set)
        assert lc.e[2] == {'l1', 'l2'}


def test_tuple_is_coerced_to_set():
    with _make('t6') as c:
        lc = eqL(c, 'text', ('l1', 'l2'))
        assert isinstance(lc.e[2], set)
        assert lc.e[2] == {'l1', 'l2'}


def test_list_is_coerced_to_set():
    with _make('t7') as c:
        lc = eqL(c, 'text', ['l1', 'l2'])
        assert isinstance(lc.e[2], set)
        assert lc.e[2] == {'l1', 'l2'}


# ---------------------------------------------------------------------------
# 2-argument form: eqL(concept, value) → filter on instanceID
# ---------------------------------------------------------------------------

def test_two_arg_form_uses_instanceid():
    with _make('t8') as c:
        lc = eqL(c, 'l2')
        assert lc.e[1] == 'instanceID'
        # Scalar is auto-wrapped to a set for exact-equality.
        assert lc.e[2] == {'l2'}


# ---------------------------------------------------------------------------
# Kwargs forwarding (active / sampleEntries / name / p)
# ---------------------------------------------------------------------------

def test_active_false_is_honoured():
    with _make('t9') as c:
        lc = eqL(c, 'text', 'l2', active=False)
        assert lc.active is False


def test_active_default_is_true():
    with _make('t10') as c:
        lc = eqL(c, 'text', 'l2')
        assert lc.active is True


def test_sample_entries_forwarded():
    with _make('t11') as c:
        lc = eqL(c, 'text', 'l2', sampleEntries=True)
        assert lc.sampleEntries is True


def test_priority_overridable():
    with _make('t12') as c:
        lc = eqL(c, 'text', 'l2', p=50)
        assert lc.p == 50


def test_priority_default_is_100():
    with _make('t13') as c:
        lc = eqL(c, 'text', 'l2')
        assert lc.p == 100


# ---------------------------------------------------------------------------
# headLC stays False (semantics preserved)
# ---------------------------------------------------------------------------

def test_head_lc_is_false():
    with _make('t14') as c:
        lc = eqL(c, 'text', 'l2')
        assert lc.headLC is False


# ---------------------------------------------------------------------------
# Footgun closed: substring match no longer possible
# ---------------------------------------------------------------------------

def test_single_char_does_not_substring_match():
    """Before: ``'l' in 'l2'`` returned True — silent substring bug.
    After: value is wrapped in a set, so membership is exact-equality."""
    with _make('t15') as c:
        lc = eqL(c, 'text', 'l')
        assert lc.e[2] == {'l'}
        assert 'l2' not in lc.e[2]


def test_empty_string_does_not_match_non_empty():
    with _make('t16') as c:
        lc = eqL(c, 'text', '')
        assert lc.e[2] == {''}
        assert 'anything' not in lc.e[2]


def test_docstring_references_372():
    assert eqL.__doc__ is not None
    assert '372' in eqL.__doc__
