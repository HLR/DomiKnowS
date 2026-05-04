import importlib.util
import sys
from pathlib import Path

import pytest

from domiknows.generation import (
    constraints_to_dfa_from_graph,
    discover_generation_constraints,
    discover_generation_enforcement,
)


def load_collie_module(filename, module_name):
    collie_dir = Path(__file__).resolve().parents[2] / "Tasks" / "collie"
    module_path = collie_dir / filename
    sys.path.insert(0, str(collie_dir))
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(collie_dir))


def load_collie_graph():
    return load_collie_module("graph.py", "collie_graph_for_generation_test")


def test_collie_build_graph_uses_generation_encoder():
    collie_graph = load_collie_graph()

    class FakeTokenizer:
        def encode(self, token):
            return {"<|endoftext|>": [0], " The": [1], " slide": [2]}[token]

    graph, bundle = collie_graph.build_graph(
        lm=None,
        tokenizer=FakeTokenizer(),
        vocab=["<|endoftext|>", " The", " slide"],
    )

    assert graph is not None
    assert bundle[3].name == "generated_token"
    assert len(graph.logicalConstrains) >= 5


def test_collie_generation_bundle_exposes_vocabulary_and_constraints():
    collie_graph = load_collie_graph()

    class FakeTokenizer:
        def encode(self, token):
            return {"<|endoftext|>": [0], " The": [1], " slide": [2]}[token]

    graph, bundle = collie_graph.build_generation_bundle(
        tokenizer=FakeTokenizer(),
        vocab=["<|endoftext|>", " The", " slide"],
    )

    discovered_constraints = discover_generation_constraints(graph, bundle)
    enforcement = discover_generation_enforcement(graph, bundle)
    dfa = constraints_to_dfa_from_graph(graph, bundle)
    eos = bundle.vocabulary.label_for_token("<|endoftext|>")
    the = bundle.vocabulary.label_for_token(" The")
    slide = bundle.vocabulary.label_for_token(" slide")

    assert graph is not None
    assert bundle.context.vocabulary is bundle.vocabulary
    assert [constraint.name for constraint in discovered_constraints] == [
        "no non-EOS tokens can follow an EOS token",
        "at most 4 non-EOS tokens are generated",
        "at least 1 ' The' token(s) are generated",
        "at least 1 ' slide' token(s) are generated",
        "if ' The' appears then at most 16 non-EOS tokens are generated",
    ]
    assert enforcement.dfa_constraints == discovered_constraints
    assert len(enforcement.latent_specs) == 1
    assert enforcement.latent_specs[0].if_label == the
    assert enforcement.latent_specs[0].formula == slide
    assert dfa.accepts([the, slide, eos, eos])
    assert not dfa.accepts([the, eos, slide])
    assert not dfa.accepts([the, eos, eos])
    assert not dfa.accepts([slide, eos, eos])


def test_collie_latent_example_computes_graph_marked_loss():
    latent_example = load_collie_module("latent_example.py", "collie_latent_example_for_generation_test")

    _, bundle, enforcement, probs, loss = latent_example.build_latent_example()
    the = bundle.vocabulary.label_for_token(" The")
    slide = bundle.vocabulary.label_for_token(" slide")

    assert probs.shape == (4, 4)
    assert len(enforcement.latent_specs) == 1
    assert enforcement.latent_specs[0].if_label == the
    assert enforcement.latent_specs[0].formula == slide
    assert loss.item() == pytest.approx(0.04)
