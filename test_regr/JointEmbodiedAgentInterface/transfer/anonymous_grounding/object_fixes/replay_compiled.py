"""Check compiled token masks against immutable, archived executor trajectories.

This is logit/mask replay, not fresh neural inference or physical execution.
No scoring-private goals or reference plans are opened.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from ...protocol import read, freeze, sha, episodes, public_input, context
from ....generalization.shared_constraints import bind_shared
from ..planner_binding.verification_protocol import exact_policy_search
from .contract import ExecutorContract, executor_policy_metadata
from .executor_graph import CAPABILITIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-operations', type=int, default=12)
    args = parser.parse_args()
    from test_regr.VLABenchAgentInterface.world_graph import build_vlabench_world_graph
    from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary
    vocabulary = PlanVocabulary.from_world(build_vlabench_world_graph('executor_lc_replay'))
    selected = []
    for cohort, snapshots in [('validation', ('epoch-1',)), ('confirmation', ('initialization', 'epoch-1'))]:
        config = args.archive/(cohort+'_config.json')
        manifest = Path(read(config)['external_manifest'])
        for snapshot in snapshots:
            decisions = args.archive/cohort/snapshot/'executor/libero/decisions.jsonl'
            selected.append((cohort, snapshot, config, manifest, decisions))
    settings = {'policy': executor_policy_metadata(), 'vocabulary_checksum': vocabulary.checksum,
        'max_operations': args.max_operations, 'kind': 'archived-logit and compiled-mask replay only',
        'inputs': {str(p):sha(p) for row in selected for p in row[2:]}}
    freeze(args.output/'settings.json', settings)
    if (args.output/'records.jsonl').exists():
        raise ValueError('preserve prior replay; select a fresh output directory')
    counts = dict(episodes=0, decisions=0, exact_masks=0, mask_differences=0,
                  greedy_divergences=0, invalid_prefixes=0, accepted_plans=0, unknown=0)
    with (args.output/'records.jsonl').open('x') as stream:
        for cohort, snapshot, config, manifest, decisions in selected:
            by_id = defaultdict(list)
            for line in decisions.read_text().splitlines():
                row = json.loads(line)
                if 'raw_argmax_allowed' in row:
                    by_id[row['episode_id']].append(row)
            examples = episodes(read(manifest))
            if len(examples) != 100 or set(by_id) != {e['episode_id'] for e in examples}:
                raise ValueError('archive episode coverage differs')
            for episode in examples:
                counts['episodes'] += 1
                public = public_input(episode)
                base = bind_shared(vocabulary, args.max_operations, context(public, 'libero'))
                dfa = ExecutorContract(base, vocabulary, public, CAPABILITIES, args.max_operations)
                state = dfa.start_state; prefix = []; stopped = False; retries = []
                from domiknows.generation.dfa.productive import SemanticSearchLimitExceeded
                try:
                    with exact_policy_search(dfa, retries):
                        for index, row in enumerate(by_id[episode['episode_id']]):
                            if row['prefix'] != prefix:
                                raise ValueError('archive trace prefix differs')
                            if row['remaining_steps'] != args.max_operations*5-index:
                                raise ValueError('archive decoding horizon differs')
                            allowed = dfa.allowed_tokens(state, remaining_steps=row['remaining_steps'])
                            tokens = {vocabulary.token_for_label(t) for t in allowed}
                            original = set(row['allowed'])
                            if set(row['unmasked_logits']) != set(vocabulary.tokens) | {'<other>'}:
                                raise ValueError('archive vocabulary/logit coverage differs')
                            chosen = min(allowed, key=lambda t: (-row['unmasked_logits'][vocabulary.token_for_label(t)],t)) if allowed else None
                            matches = chosen is not None and vocabulary.token_for_label(chosen) == row['chosen']
                            counts['decisions'] += 1
                            counts['exact_masks'] += tokens == original
                            counts['mask_differences'] += tokens != original
                            counts['greedy_divergences'] += not matches
                            stream.write(json.dumps({'cohort':cohort, 'snapshot':snapshot, 'episode_id':episode['episode_id'],
                                'step':index, 'exact_mask':tokens==original, 'removed':sorted(original-tokens),
                                'added':sorted(tokens-original), 'same_greedy_choice':matches})+'\n')
                            if not matches:
                                stopped = True; break  # never replay an invalid/different prefix
                            label = vocabulary.label_for_token(row['chosen'])
                            state = dfa.step(state, label)
                            if state is None:
                                counts['invalid_prefixes'] += 1; stopped = True; break
                            prefix.append(row['chosen'])
                    counts['accepted_plans'] += not stopped and dfa.is_accepting(state)
                except SemanticSearchLimitExceeded:
                    counts['unknown'] += 1  # never replace unknown feasibility with an approximate mask
                if retries:
                    stream.write(json.dumps({'cohort':cohort, 'snapshot':snapshot,
                        'episode_id':episode['episode_id'], 'exact_search_retries':retries})+'\n')
                stream.flush()
            print(f'[compiled-mask-replay] {cohort} {snapshot}: {counts}', flush=True)
    summary = {**counts, 'passed':counts['accepted_plans']==300 and counts['mask_differences']==0
        and counts['greedy_divergences']==0 and counts['unknown']==0,
        'settings_sha256':sha(args.output/'settings.json'), 'no_scoring_references_read':True,
        'no_model_or_simulator_execution':True}
    freeze(args.output/'summary.json', summary)
    if not summary['passed']:
        raise RuntimeError('strict compiled-mask equivalence failed; inspect complete replay records')


if __name__ == '__main__':main()
