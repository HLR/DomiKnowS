from __future__ import annotations

import importlib

import pytest

from domiknows.generation import (
    HMMGenerationHead,
    constraints_to_dfa_from_graph,
)
from domiknows.generation.dfa._lc_normalize import _ForbiddenLeaf, _NotNode, normalize_lc


def import_task_module(name: str):
    return importlib.import_module(f"Tasks.nested_constraints_demo.{name}")


def _labels(bundle, symbols):
    return [bundle.vocabulary.label_for_token(symbol) for symbol in symbols]


def test_demo_builds_self_contained_bundle_and_dfa():
    graph_module = import_task_module("graph")
    graph, bundle = graph_module.build_bundle()

    # Three head LCs registered.
    head_lcs = [name for name, lc in graph.logicalConstrains.items() if getattr(lc, "headLC", True)]
    assert len(head_lcs) == 3

    # Compile the DFA with the normalizer + minimize wired up; one warning is
    # expected from LC #2's heterogeneous andL salvage.
    with pytest.warns(RuntimeWarning, match="not supported by generation DFA discovery"):
        dfa = constraints_to_dfa_from_graph(graph, bundle, on_unsupported="warn")

    # Valid sequences accepted.
    assert dfa.accepts(_labels(bundle, ["A", "B", "END"]))
    assert dfa.accepts(_labels(bundle, ["C", "END"]))
    # ¬(∃A ∧ ∃C) violation.
    assert not dfa.accepts(_labels(bundle, ["A", "C", "END"]))
    # atMostAL(B, 1) violation.
    assert not dfa.accepts(_labels(bundle, ["B", "B", "END"]))
    # forbidden-D violation (regular leaf inside LC #2).
    assert not dfa.accepts(_labels(bundle, ["D", "END"]))
    # EOS-closure violation.
    assert not dfa.accepts(_labels(bundle, ["A", "END", "A"]))


def test_demo_normalize_collapses_notL_existsAL():
    graph_module = import_task_module("graph")
    graph, bundle = graph_module.build_bundle()
    head_lcs = [lc for _name, lc in graph.logicalConstrains.items() if getattr(lc, "headLC", True)]
    lc1 = head_lcs[0]

    normal = normalize_lc(lc1, bundle=bundle)

    # Collect every leaf in the normalized mirror tree.
    leaves = []

    def collect(node):
        children = getattr(node, "e", ())
        if not children:
            leaves.append(node)
            return
        if getattr(node, "_kind", None) == "_forbidden_token":
            leaves.append(node)
            return
        for child in children:
            collect(child)

    collect(normal.tree)

    forbidden_tokens = {leaf.token for leaf in leaves if isinstance(leaf, _ForbiddenLeaf)}
    # De Morgan + notL(existsAL(t)) collapse should produce forbidden leaves for A and C.
    assert "A" in forbidden_tokens
    assert "C" in forbidden_tokens

    # No surviving `_NotNode((existsAL(...),))` — the normalizer should have
    # collapsed every such pair into a `_ForbiddenLeaf`.

    def walk(node):
        yield node
        for child in getattr(node, "e", ()):
            yield from walk(child)

    for node in walk(normal.tree):
        if isinstance(node, _NotNode):
            inner = node.e[0] if node.e else None
            inner_kind = getattr(inner, "_kind", None) or (type(inner).__name__ if inner is not None else None)
            assert inner_kind != "existsAL", "notL(existsAL) should have been rewritten"


def test_demo_heterogeneous_andL_salvage():
    graph_module = import_task_module("graph")
    graph, bundle = graph_module.build_bundle()
    head_lcs = [lc for _name, lc in graph.logicalConstrains.items() if getattr(lc, "headLC", True)]
    lc2 = head_lcs[1]

    normal = normalize_lc(lc2, bundle=bundle)

    # The inner andL of raw concept tuples should be surfaced as irregular.
    assert len(normal.irregular_children) >= 1


def test_demo_program_builds_with_default_discrete_hmm_learner():
    learning_program = import_task_module("learning_program")

    # ``stream_count=4`` and ``pad_size=12`` matches the parity test in
    # ``test_real_hmm_pmd_learning_task.py`` so the surface is consistent.
    artifacts = learning_program.build_learning_program(stream_count=4, pad_size=12, random_seed=0)

    assert artifacts.program is not None
    assert artifacts.learner_name == "discrete-hmm"
    assert isinstance(artifacts.model, HMMGenerationHead)
    assert artifacts.model.prompt_conditioning == "initial"


def test_demo_stream_generator_uses_valid_and_invalid_outputs():
    learning_program = import_task_module("learning_program")
    artifacts = learning_program.build_learning_program(stream_count=4, stream_seed=0, pad_size=6, random_seed=0)

    names = [example.name for example in artifacts.stream_examples]
    assert "valid" in names
    assert "invalid" in names
    # Sanity: every example carries a known prompt name.
    prompt_names = {example.prompt_name for example in artifacts.stream_examples}
    assert prompt_names.issubset({"with_A", "with_C"})


def test_dfa_performance_constraint_catalog_compiles_and_classifies():
    """Smoke-test the benchmark's constraint catalog.

    Builds a level-5 graph through the benchmark's ``apply_scaled_constraints``,
    confirms the matcher produces at least one supported analysis, and that
    the resulting DFA correctly accepts a hand-picked valid sequence and
    rejects an invalid one.  Guards the catalog from silent regressions
    without invoking the timing-heavy benchmark itself.
    """
    perf_module = import_task_module("run_dfa_performance")
    from domiknows.generation.dfa.graph_discovery import (
        analyze_generation_constraints,
        constraints_to_dfa_from_graph,
    )

    graph, bundle, num_lcs = perf_module._build_bench_graph(5)
    assert num_lcs == 5

    analyses = analyze_generation_constraints(graph, bundle, on_unsupported="warn")
    assert any(analysis.supported for analysis in analyses), "no supported analyses"

    dfa = constraints_to_dfa_from_graph(graph, bundle, on_unsupported="warn")

    def labels(symbols):
        return [bundle.vocabulary.label_for_token(symbol) for symbol in symbols]

    # ``A END`` satisfies every level-1..5 rule: EOS-closure (END is last),
    # at-most-one-B (zero Bs), forbidden-D (no D), multi-token implication
    # (after-trigger allows END), and "at least one of A or C".
    assert dfa.accepts(labels(["A", "END"]))
    # ``B B END`` violates the at-most-one-B rule.
    assert not dfa.accepts(labels(["B", "B", "END"]))
