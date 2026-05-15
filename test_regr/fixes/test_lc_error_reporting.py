"""
Regression tests for issue #376 — better error reporting in logical constraints.

Verifies that:
1. Undefined variables in LC paths raise ValueError (not generic Exception)
2. Error messages include 'did you mean' suggestions for typos
3. Error messages include the offending path
4. Valid constraints still pass without error
"""
import pytest
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domiknows.graph import Graph, Concept, Relation, ifL, notL, andL, nandL


# ---------- helpers ----------

def _build_simple_graph():
    """sentence -> word -> tag, with contains relations."""
    with Graph('test_lc_err') as g:
        sentence = Concept('sentence')
        word = Concept('word')
        tag = Concept('tag')
        (s_w,) = sentence.contains(word)
        (w_t,) = word.contains(tag)
    return g, sentence, word, tag, s_w, w_t


# ---------- undefined variable detection ----------

class TestUndefinedVariableError:
    def test_typo_raises_valueerror(self):
        """Using 'e1' instead of 'el1' must raise ValueError."""
        with pytest.raises(ValueError, match="not defined"):
            _build_simple_graph()  # clear state first
            with Graph('g') as g:
                sentence = Concept('sentence')
                word = Concept('word')
                tag = Concept('tag')
                (s_w,) = sentence.contains(word)
                (w_t,) = word.contains(tag)
                ifL(
                    word('el1'),
                    notL(tag('t1', path=(('e1', s_w, w_t),)))
                )

    def test_error_suggests_correct_variable(self):
        """The error message should suggest 'el1' for mistyped 'e1'."""
        with pytest.raises(ValueError, match=r"Did you mean.*'el1'"):
            with Graph('g2') as g:
                sentence = Concept('sentence')
                word = Concept('word')
                tag = Concept('tag')
                (s_w,) = sentence.contains(word)
                (w_t,) = word.contains(tag)
                ifL(
                    word('el1'),
                    notL(tag('t1', path=(('e1', s_w, w_t),)))
                )

    def test_error_includes_path(self):
        """The error message should mention the offending path."""
        with pytest.raises(ValueError, match="Used in path"):
            with Graph('g3') as g:
                sentence = Concept('sentence')
                word = Concept('word')
                tag = Concept('tag')
                (s_w,) = sentence.contains(word)
                (w_t,) = word.contains(tag)
                ifL(
                    word('el1'),
                    notL(tag('t1', path=(('e1', s_w, w_t),)))
                )

    def test_completely_unknown_variable(self):
        """A variable name with no close match still raises ValueError."""
        with pytest.raises(ValueError, match="not defined"):
            with Graph('g4') as g:
                sentence = Concept('sentence')
                word = Concept('word')
                tag = Concept('tag')
                (s_w,) = sentence.contains(word)
                (w_t,) = word.contains(tag)
                ifL(
                    word('w1'),
                    notL(tag('t1', path=(('zzz_nonexistent', s_w, w_t),)))
                )

    def test_unknown_variable_lists_defined_vars(self):
        """When no close match, the error should list defined variables."""
        with pytest.raises(ValueError, match=r"(Did you mean|Variables defined)"):
            with Graph('g5') as g:
                sentence = Concept('sentence')
                word = Concept('word')
                tag = Concept('tag')
                (s_w,) = sentence.contains(word)
                (w_t,) = word.contains(tag)
                ifL(
                    word('w1'),
                    notL(tag('t1', path=(('xyz', s_w, w_t),)))
                )


# ---------- valid constraints still pass ----------

class TestValidConstraintsStillWork:
    def test_correct_variable_no_error(self):
        """A properly defined variable should not raise."""
        with Graph('g_ok') as g:
            sentence = Concept('sentence')
            word = Concept('word')
            tag = Concept('tag')
            (s_w,) = sentence.contains(word)
            (w_t,) = word.contains(tag)
            ifL(
                word('w1'),
                notL(tag('t1', path=('w1', s_w, w_t)))
            )
        # reaching here = no error, test passes

    def test_nandL_valid(self):
        """nandL with properly defined concepts must not raise."""
        with Graph('g_nand') as g:
            w = Concept('word')
            people = w(name='people')
            org = w(name='organization')
            nandL(people, org, active=True)
        # reaching here = no error


# ---------- error type consistency ----------

class TestErrorTypeConsistency:
    def test_all_lc_errors_are_valueerror(self):
        """Verify that validation errors from the LC checker are ValueError, not bare Exception."""
        # Trigger a path validation error — 'nosuchvar' is not defined anywhere
        with pytest.raises(ValueError):
            with Graph('g_card') as g:
                w = Concept('word')
                tag = Concept('tag')
                (w_t,) = w.contains(tag)
                ifL(
                    w('x'),
                    notL(tag('t', path=('nosuchvar', w_t)))
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
