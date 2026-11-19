"""Verify real HF processor inputs without loading or scoring any model."""
import argparse
from copy import deepcopy
from pathlib import Path

from ..protocol import read, freeze, sha, digest, episodes, public_input
from ..matched_baseline.protocol import common_prompt
from ..matched_baseline.evaluate import input_record
from .input_protocol import IMAGE_INSTRUCTION, generation_context
from .protocol import verify


def collect(root):
    from transformers import AutoProcessor
    from ...models import JointQwenVLPlanner
    from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary

    root = Path(root); config = read(root/'config.json'); verify(config)
    if config['evaluation_input_protocol'] != IMAGE_INSTRUCTION:
        raise ValueError('strict input contract required')
    vocabulary = PlanVocabulary.default(max_entities=config['max_entities'])
    initialization = read(Path(config['training_parent'])/'native_base/initialization.json')
    if list(vocabulary.tokens) != initialization['vocabularies']['vlabench']:
        raise ValueError('processor audit vocabulary differs')

    class Probe:
        processor = AutoProcessor.from_pretrained(config['model'], local_files_only=True)
        _prepare_inputs_and_boundaries = JointQwenVLPlanner._prepare_inputs_and_boundaries
        def _prompt_base(self, *args):
            raise AssertionError('fallback prompt forbidden')
        def _chat(self, *args):
            raise AssertionError('fallback chat forbidden')

    probe = Probe(); results = []
    with common_prompt(probe):
        for episode in episodes(read(root/'confirmation/manifest.json')):
            public = public_input(episode); altered = deepcopy(public)
            altered.update(goals=['__PRIVATE_AUDIT_CANARY__'], reference_plan=['__PRIVATE_AUDIT_CANARY__'],
                task_id='__PRIVATE_AUDIT_CANARY__', predicted_roles={'source': 999, 'destination': 999},
                movable=[], positions={'__PRIVATE_AUDIT_CANARY__':[0,0,0]}, initial_relations=[],
                entity_types={'__PRIVATE_AUDIT_CANARY__':'__PRIVATE_AUDIT_CANARY__'},
                button_hints='__PRIVATE_AUDIT_CANARY__')
            altered['entities'] = ['__PRIVATE_AUDIT_CANARY__'+str(i) for i in range(len(public['entities']))]
            records = []
            for observation in (public, altered):
                bound = generation_context(observation, vocabulary)
                inputs, boundaries = probe._prepare_inputs_and_boundaries('vlabench', bound, ())
                rendered = input_record(probe.processor, bound['_matched_core'], observation, None, inputs)
                if '__PRIVATE_AUDIT_CANARY__' in rendered['rendered_prompt']:
                    raise ValueError('private metadata reached rendered prompt')
                records.append({'rendered': rendered, 'boundaries': boundaries})
            if records[0] != records[1]:
                raise ValueError('metadata/reference perturbation changed processor inputs')
            results.append({'episode_id': episode['episode_id'], 'input_sha256': digest(records[0]),
                **records[0]})
    result = {'passed': True, 'examples': len(results), 'input_protocol': IMAGE_INSTRUCTION,
        'config_sha256': sha(root/'config.json'), 'model_weights_loaded': False,
        'model_generation_or_goal_scoring': False, 'metadata_and_reference_invariance': True,
        'episodes': results}
    freeze(root/'confirmation/model_input_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    collect(parser.parse_args().root)
