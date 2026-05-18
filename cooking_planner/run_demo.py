"""Run the declarative cooking planner graph-HMM demo."""
from __future__ import annotations

import argparse

from domiknows.generation import explain_dfa_rejection
from domiknows.generation.planning import planning_bundle_from_graph, planning_dfa_from_graph

try:
    from .graph import build_graph
    from .learning_program import build_graph_hmm_head, fit_graph_hmm, run_one_head_step
    from .planner_agent import MockCookingPlannerAgent
except ImportError:
    from graph import build_graph
    from learning_program import build_graph_hmm_head, fit_graph_hmm, run_one_head_step
    from planner_agent import MockCookingPlannerAgent


def _format_plan(plan) -> str:
    return " -> ".join(plan)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dish", choices=("cookie", "omelette", "salad"), default="cookie")
    parser.add_argument("--candidates", type=int, default=6)
    parser.add_argument("--hmm-iterations", type=int, default=20)
    parser.add_argument("--head-steps", type=int, default=1)
    parser.add_argument("--show-invalid", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    graph, _parts = build_graph(args.dish)
    bundle = planning_bundle_from_graph(graph, selected_task=args.dish)
    dfa = planning_dfa_from_graph(bundle)
    artifacts = fit_graph_hmm(bundle, max_iter=args.hmm_iterations, random_seed=args.seed)
    head = build_graph_hmm_head(bundle, random_seed=args.seed)

    print("Cooking planner path: declarative DomiKnowS graph -> planning DFA + graph-HMM")
    print(f"Dish: {bundle.selected_task}")
    print("Actions:", bundle.action_names)
    print("Phases:", bundle.phase_names)
    print("Required actions:", bundle.selected_required_actions)
    print("Reference plan:", _format_plan(bundle.selected_reference_plan))
    print("HMM log-likelihoods:", artifacts.hmm.fit_result_.log_likelihoods)
    if args.head_steps:
        for step in range(args.head_steps):
            print(f"GraphHMMGenerationHead step {step + 1}:", run_one_head_step(head, bundle))

    planner = MockCookingPlannerAgent(seed=args.seed)
    scored = []
    print("\nCandidate plans:")
    for candidate in planner.propose(bundle, count=args.candidates):
        accepted = dfa.accepts(candidate.actions)
        rejection = None if accepted else explain_dfa_rejection(dfa, candidate.actions)
        score = artifacts.hmm.score(candidate.actions)
        viterbi = None
        if score != float("-inf"):
            viterbi = artifacts.hmm.viterbi(candidate.actions).states
        scored.append((accepted, score, candidate))
        if accepted or args.show_invalid:
            print(f"- {candidate.source}")
            print(f"  plan: {_format_plan(candidate.actions)}")
            print(f"  accepted: {accepted}")
            if rejection:
                print(f"  rejection: {rejection}")
            print(f"  hmm_score: {score:.4f}" if score != float("-inf") else "  hmm_score: -inf")
            if viterbi is not None:
                print(f"  viterbi_phases: {viterbi}")

    valid = [item for item in scored if item[0]]
    if valid:
        best = max(valid, key=lambda item: item[1])
        print("\nBest accepted plan:")
        print(f"  source: {best[2].source}")
        print(f"  plan: {_format_plan(best[2].actions)}")
        print(f"  hmm_score: {best[1]:.4f}")
    else:
        print("\nNo accepted candidate plan was proposed.")

    print(f"\nDFA states: {len(dfa.states)}")
    print(f"Graph: {graph.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
