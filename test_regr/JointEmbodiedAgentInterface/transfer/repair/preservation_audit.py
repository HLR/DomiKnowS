"""Post-generation audit of matched decoding; never changes generation or rules."""
import argparse
import json
from pathlib import Path

from ..protocol import read, freeze, sha, episodes, public_input
from .input_protocol import constraint_context, constraint_public


def first_difference(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                min(len(left), len(right)) if len(left) != len(right) else None)


def compare_episode(free, constrained, compiled_accepts, surface_accepts=None):
    """Legality is supplied only by the DomiKnowS-compiled policy callback."""
    if free['episode_id'] != constrained['episode_id']:
        raise ValueError('paired episode identity differs')
    if free['input_hash'] != constrained['input_hash']:
        raise ValueError('paired neural input differs')
    eligible = not free['generation_error'] and compiled_accepts(free['prediction'])
    if eligible:
        if surface_accepts is not None and not surface_accepts(free['labels']):
            raise ValueError('projection rejected a compiled-legal normalized greedy output')
        if constrained['generation_error'] or free['labels'] != constrained['labels'] or free['prediction'] != constrained['prediction']:
            raise ValueError('repaired DFA changed a compiled-legal greedy output')
    if not constrained['generation_error'] and not compiled_accepts(constrained['prediction']):
        raise ValueError('constrained output violates the compiled policy')
    return {'episode_id': free['episode_id'], 'preservation_required': bool(eligible),
            'free_generation_error': free['generation_error'],
            'constrained_generation_error': constrained['generation_error'],
            'first_token_divergence': first_difference(free['labels'], constrained['labels']),
            'canonical_plan_changed': free['prediction'] != constrained['prediction']}


def load_rows(path, identities):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines()]
    indexed = {row['episode_id']: row for row in rows}
    if len(rows) != len(identities) or set(indexed) != set(identities):
        raise ValueError('condition has incomplete or duplicate episode identities')
    return indexed


def collect(root):
    # Public observations, tokens, and the frozen declared rule compiler suffice.
    # No model weights, goal files, native success sensors, or reference plans.
    from test_regr.VLABenchAgentInterface.world_graph import build_vlabench_world_graph
    from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary, plan_to_tokens
    from test_regr.VLABenchAgentInterface.native_planning import byte_token_map
    from ...generalization.shared_constraints import bind_shared, SHARED_RULES
    from .surface import surface_decoder, SURFACE_VERSION
    from transformers import AutoTokenizer

    root = Path(root)
    config = read(root/'config.json')
    manifest = root/'confirmation/manifest.json'
    selected = episodes(read(manifest))
    identities = [e['episode_id'] for e in selected]
    world = build_vlabench_world_graph('repair_preservation_audit_world')
    initialization = read(Path(config.get('training_parent',root))/'native_base/initialization.json')
    saved_tokens = initialization['vocabularies']['vlabench']
    vocabulary = PlanVocabulary.from_world(world, max_entities=sum(t.startswith('obj:') for t in saved_tokens))
    if list(vocabulary.tokens) != saved_tokens:
        raise ValueError('audit vocabulary differs from frozen initialization')
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    token_bytes = byte_token_map(tokenizer)
    outputs = {}; fingerprints = {}; cache = {}
    for system in ('base', 'native-trained', 'joint-initialization', 'joint-trained'):
        folders = [root/'evaluation'/system/p for p in ('none', 'shared-verified')]
        settings = [read(p/'settings.json') for p in folders]
        if settings[0]['policy'] != 'none' or settings[1]['policy'] != 'shared-verified':
            raise ValueError('condition policy labels differ')
        for key in set(settings[0]) | set(settings[1]):
            if key != 'policy' and settings[0].get(key) != settings[1].get(key):
                raise ValueError('condition setting differs: '+key)
        if settings[0]['manifest_sha256'] != sha(manifest):
            raise ValueError('manifest differs from planning settings')
        rows = [load_rows(p/'predictions.jsonl', identities) for p in folders]
        results = []
        for episode in selected:
            public = public_input(episode)
            bound = constraint_context(public)
            compiled = bind_shared(vocabulary, 12, bound)

            def accepts(plan):
                try:
                    labels = [vocabulary.label_for_token(t) for t in plan_to_tokens(plan, public['entities'], world=world)]
                    state = compiled.start_state
                    for label in labels:
                        if state is None or label not in compiled.allowed_tokens(state):
                            return False
                        state = compiled.step(state, label)
                    return state is not None and compiled.is_accepting(state)
                except (ValueError, KeyError, TypeError):
                    return False

            surface = None
            if settings[0]['representation'] == 'native':
                observation = constraint_public(public)
                key = (tuple(observation['entities']), tuple(observation['movable']))
                if key not in cache:
                    cache[key] = surface_decoder(token_bytes, tokenizer.eos_token_id, observation, vocabulary, 12)
                surface = cache[key].accepts
            identity = episode['episode_id']
            results.append(compare_episode(rows[0][identity], rows[1][identity], accepts, surface))
        outputs[system] = {'episodes': results,
            'preserved_compiled_legal': sum(r['preservation_required'] for r in results),
            'canonical_changes': sum(r['canonical_plan_changed'] for r in results)}
        for folder in folders:
            for name in ('settings.json', 'predictions.jsonl'):
                fingerprints[str(folder/name)] = sha(folder/name)
    result = {'passed': True, 'systems': outputs, 'manifest_sha256': sha(manifest),
        'files': fingerprints, 'surface_version': SURFACE_VERSION,
        'compiled_rules': sorted(SHARED_RULES), 'native_success_or_goals_opened': False,
        'scope': 'Matched greedy preservation and actual constrained-plan legality only'}
    freeze(root/'confirmation/preservation_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    collect(parser.parse_args().root)
