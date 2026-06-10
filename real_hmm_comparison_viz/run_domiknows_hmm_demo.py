"""Minimal DomiKnowS-aware DFA + HMM configuration demo.

This script intentionally does *not* build the JSON/HTML visualization flow and
does *not* include the plain baseline DiscreteHMM.  It is meant to be read as a
small end-user recipe:

1. Build a declarative DomiKnowS graph.
2. Wrap that graph as a generation bundle.
3. Compile the graph constraints into a hard DFA.
4. Compile the same DFA support into a DomiKnowS-aware HMM state space.
5. Score and decode one candidate string.
"""
from __future__ import annotations

import argparse

from domiknows.generation import (
    analyze_generation_constraints,
    constraints_to_dfa_from_graph,
    domiknows_hmm_from_generation_constraints,
    explain_dfa_rejection,
    generation_bundle_from_graph,
)

try:
    from .flow import _comparison_state_name_fn
    from .graph import EOS_TOKEN, VOCAB, build_graph, build_two_constraint_graph
except ImportError:  # pragma: no cover - direct script execution
    from flow import _comparison_state_name_fn
    from graph import EOS_TOKEN, VOCAB, build_graph, build_two_constraint_graph


ONE_CONSTRAINT_CANDIDATES = {
    "valid": ("A", "B", "C", "A"),
    "invalid": ("A", "B", "C", "B"),
}

TWO_CONSTRAINT_CANDIDATES = {
    "valid": ("A", "B", "C", "END"),
    "two_b": ("A", "B", "C", "B", "END"),
    "missing_c": ("A", "B", "END"),
}


def build_bundle(demo: str):
    """Build the human-authored graph and adapt it for generation tooling."""

    graph, _parts = build_graph() if demo == "one" else build_two_constraint_graph()

    # The traditional graph contains domain concepts named string/position and
    # an EnumConcept named generated_symbol.  The bundle tells generation tools
    # which graph objects play those roles; it does not create new domain logic.
    bundle = generation_bundle_from_graph(
        graph,
        vocab=VOCAB,
        eos_token=EOS_TOKEN,
        text_name="string",
        token_name="position",
        generated_token_name="generated_symbol",
        before_relation_name="precedes",
        first_role_name="earlier",
        second_role_name="later",
    )
    return graph, bundle


def label_sequence(bundle, tokens: tuple[str, ...]) -> list[int]:
    """Map readable token strings into the compact DFA label ids."""

    return [bundle.vocabulary.label_for_token(token) for token in tokens]


def run_demo(*, demo: str, candidate_name: str) -> dict:
    """Run the minimal DomiKnowS-aware configuration path."""

    candidates = ONE_CONSTRAINT_CANDIDATES if demo == "one" else TWO_CONSTRAINT_CANDIDATES
    if candidate_name not in candidates:
        raise ValueError(f"unknown candidate {candidate_name!r}; expected one of {tuple(candidates)}")
    candidate = candidates[candidate_name]

    graph, bundle = build_bundle(demo)

    # 1. Discover the graph constraints that are regular enough to enforce
    # during generation.  This is the symbolic/logic side.
    analyses = analyze_generation_constraints(graph, bundle, on_unsupported="error")

    # 2. Compile the discovered constraints into a hard DFA.  The DFA is the
    # exact yes/no verifier: it accepts or rejects a compact-label sequence.
    dfa = constraints_to_dfa_from_graph(graph, bundle, on_unsupported="error")
    labels = label_sequence(bundle, candidate)
    accepted = dfa.accepts(labels)
    rejection = None if accepted else explain_dfa_rejection(dfa, labels)

    # 3. Compile the same DFA support into a DomiKnowS-aware HMM.  This creates
    # hidden states automatically from productive DFA edges:
    #
    #     (dfa_state_before, emitted_symbol, dfa_state_after)
    #
    # The HMM still has probabilities, but its support is constrained by the
    # graph-derived DFA language.
    hmm = domiknows_hmm_from_generation_constraints(
        graph,
        bundle,
        symbols=VOCAB,
        eos_token=EOS_TOKEN,
        state_name_fn=_comparison_state_name_fn(demo),
        on_unsupported="error",
    )

    score = hmm.score(candidate)
    viterbi = hmm.viterbi(candidate)
    compilation = hmm.constraint_hmm_compilation

    return {
        "graph": graph.name,
        "candidate": candidate,
        "constraints": [analysis.lc_name for analysis in analyses if analysis.supported],
        "dfa_accepted": accepted,
        "dfa_rejection": rejection,
        "dfa_state_count": len(dfa.states),
        "hmm_state_count": len(hmm.state_names),
        "hmm_score": score,
        "hmm_viterbi_states": viterbi.states,
        "generated_hmm_states": tuple(state.name for state in compilation.states),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", choices=("one", "two"), default="two")
    parser.add_argument("--candidate", default="two_b")
    parser.add_argument("--show-states", action="store_true")
    args = parser.parse_args(argv)

    result = run_demo(demo=args.demo, candidate_name=args.candidate)

    print("Minimal DomiKnowS-aware DFA + HMM demo")
    print("Graph:", result["graph"])
    print("Candidate:", " ".join(result["candidate"]))
    print("Discovered constraints:")
    for name in result["constraints"]:
        print(f" - {name}")
    print("DFA accepted:", result["dfa_accepted"])
    if result["dfa_rejection"]:
        print("DFA rejection:", result["dfa_rejection"])
    print("DFA states:", result["dfa_state_count"])
    print("Auto-generated HMM states:", result["hmm_state_count"])
    print("HMM log-likelihood:", result["hmm_score"] if result["hmm_score"] != float("-inf") else "-inf")
    print("HMM Viterbi path:", result["hmm_viterbi_states"] or "<no legal path>")

    if args.show_states:
        print("\nGenerated HMM states from DFA edges:")
        for state in result["generated_hmm_states"]:
            print(f" - {state}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
