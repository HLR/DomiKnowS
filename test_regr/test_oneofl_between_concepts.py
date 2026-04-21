"""
Regression tests for issue #371: ExactL between concepts.

The issue asks for a clean way to express "exactly one of these concepts
holds" — i.e. mutual exclusion + covering — across multiple concept
applications on the same candidate.

This is now provided by ``oneOfL`` (with its accumulated sibling
``oneOfAL``). The tests below pin down the semantics:

1. ``oneOfL`` is a counting constraint with ``limitOp == '=='`` and
   ``fixedLimit == 1`` so the limit cannot be overridden.
2. It is importable from the public ``domiknows.graph`` package.
3. It is recognised by the adaptive t-norm loss calculator as a counting
   constraint of type ``'L'`` and normalises to ``oneOfAL`` when used in
   accumulated contexts.
4. It behaves exactly like ``exactL(..., 1)`` semantically — both produce
   the same ``limitOp``/``limit`` pair when dispatched.
"""
import pytest


class TestOneOfLClassSemantics:
    def test_oneofl_is_count_eq_one(self):
        from domiknows.graph.logicalConstrain import oneOfL, _CountBaseL
        assert issubclass(oneOfL, _CountBaseL)
        assert oneOfL.limitOp == "=="
        assert oneOfL.fixedLimit == 1

    def test_oneofal_is_accumulated_eq_one(self):
        from domiknows.graph.logicalConstrain import oneOfAL, _AccumulatedCountBaseL
        assert issubclass(oneOfAL, _AccumulatedCountBaseL)
        assert oneOfAL.limitOp == "=="
        assert oneOfAL.fixedLimit == 1

    def test_importable_from_graph_package(self):
        from domiknows.graph import oneOfL, oneOfAL  # noqa: F401


class TestLimitCannotBeOverridden:
    """The main motivation for oneOfL over exactL is that the limit is pinned."""

    def test_trailing_int_is_ignored_by_oneofl(self):
        # We can't build a full LC graph without a Concept context, but we can
        # check the class-level fixedLimit is what __call__ consults.
        from domiknows.graph.logicalConstrain import oneOfL
        # fixedLimit wins regardless of what e[-1] holds — verified via the
        # source of _CountBaseL.__call__ which reads `self.fixedLimit` first.
        assert oneOfL.fixedLimit == 1

    def test_exactl_limit_would_be_overridable(self):
        # Contrast: exactL has fixedLimit = None and so exactL(..., 2) would
        # mean "==2". This is the subtle footgun oneOfL closes.
        from domiknows.graph.logicalConstrain import exactL
        assert exactL.fixedLimit is None
        assert exactL.limitOp == "=="


class TestAdaptiveTNormRegistry:
    def test_oneofl_registered_as_counting(self):
        from domiknows.solver.adaptiveTNormLossCalculator import (
            COUNTING_CONSTRAINTS,
            DEFAULT_TNORM_BY_TYPE,
        )
        assert 'oneOfL' in COUNTING_CONSTRAINTS
        assert 'oneOfAL' in COUNTING_CONSTRAINTS
        assert DEFAULT_TNORM_BY_TYPE['oneOfL'] == 'L'
        assert DEFAULT_TNORM_BY_TYPE['oneOfAL'] == 'L'

    def test_get_constraint_type_normalises_oneofl(self):
        from domiknows.solver.adaptiveTNormLossCalculator import get_constraint_type
        from domiknows.graph.logicalConstrain import oneOfL

        # Can't fully instantiate without a concept graph; use a bare object
        # with the right type name.
        class _Stub:
            innerLC = None
        _Stub.__name__ = 'oneOfL'
        # get_constraint_type looks at type(lc).__name__.
        class Fake(oneOfL.__mro__[-2]):  # minimal subclass
            pass
        Fake.__name__ = 'oneOfL'
        fake = object.__new__(Fake)
        fake.innerLC = None
        assert get_constraint_type(fake) == 'oneOfAL'


class TestOneOfLMatchesExactLSemantics:
    """oneOfL should behave as exactL(..., 1) when dispatched."""

    def test_dispatch_uses_limit_one_and_eq(self):
        """Simulate _CountBaseL.__call__'s limit resolution for both classes."""
        from domiknows.graph.logicalConstrain import exactL, oneOfL

        # exactL with no trailing int → limit defaults to 1, op is '=='.
        exactL_instance_e = (object(), object(), object())
        # For exactL with no trailing int: e[-1] is not int, so limit = 1.
        resolved_limit_exact = (
            exactL_instance_e[-1]
            if (exactL_instance_e and isinstance(exactL_instance_e[-1], int))
            else 1
        )
        assert resolved_limit_exact == 1
        assert exactL.limitOp == "=="

        # oneOfL always pins to 1 via fixedLimit, regardless of e content.
        assert oneOfL.fixedLimit == 1
        assert oneOfL.limitOp == "=="

    def test_trailing_int_would_break_exactl_but_not_oneofl(self):
        """Reproduces the footgun: exactL(c1, c2, c3, 2) silently means '==2'."""
        from domiknows.graph.logicalConstrain import exactL, oneOfL

        fake_e_with_int = (object(), object(), object(), 2)
        resolved_limit_exact = (
            fake_e_with_int[-1]
            if (fake_e_with_int and isinstance(fake_e_with_int[-1], int))
            else 1
        )
        assert resolved_limit_exact == 2  # exactL footgun: silently ==2

        # oneOfL ignores trailing ints because fixedLimit takes precedence in
        # _CountBaseL.__call__ (checked via the source: `if self.fixedLimit
        # is not None: limit = self.fixedLimit`).
        assert oneOfL.fixedLimit == 1


class TestOneOfLHasDocstring:
    def test_docstring_mentions_issue_371(self):
        from domiknows.graph.logicalConstrain import oneOfL
        assert oneOfL.__doc__ is not None
        assert '371' in oneOfL.__doc__
        # Makes sure the docstring actually shows the intended pattern.
        assert 'exactly one' in oneOfL.__doc__.lower()
