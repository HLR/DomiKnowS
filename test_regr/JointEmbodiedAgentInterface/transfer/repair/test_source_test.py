from types import SimpleNamespace
import pytest
from .source_test import observation_inputs, metric_columns, ordered_observations


def test_macro_uses_original_official_0_80_scale():
    scores = dict(zip(('M&T','Spatial','ComSense','Semantic','PhysLaw','Complex'), (80,70,60,50,40,30)))
    report = {'corrected': {'planning_metrics': {'Macro': 88.0,
        'official_mapping': {'legacy_total_score_by_dimension': scores}, 'overall': {'score': 90.0}}}}
    actual = metric_columns(report)
    assert actual['Macro'] == 55.0 and actual['progress_score'] == 90.0
    assert actual['coverage_Macro_0_100'] == 88.0


def test_test_split_requires_all_exact_identities():
    ids = [str(i) for i in range(450)]
    manifest = {'complete':True,'references_read':False,'test_training_used':False,
        'vlabench':[{'episode_id':i} for i in reversed(ids)]}
    assert [r['episode_id'] for r in ordered_observations(manifest, ids)] == ids
    manifest['vlabench'][-1] = manifest['vlabench'][0]
    with pytest.raises(ValueError, match='exact frozen'):
        ordered_observations(manifest, ids)


def test_test_reference_fields_and_names_cannot_enter_model_context():
    from test_regr.VLABenchAgentInterface.plan_codec import PlanVocabulary
    example = SimpleNamespace(entities=('PRIVATE_A','PRIVATE_B'), instruction='Move the object.',
        operation_sequence=['SECRET_ANSWER'], task='SECRET_TASK')
    observed = {'canonical_entities':['PRIVATE_A','PRIVATE_B'],'files':{},
        'raw_image':'raw.png','canonical_image':'tags.png','target':'SECRET_TARGET'}
    public, bound, constraint = observation_inputs(example, observed, PlanVocabulary.default(max_entities=4))
    assert public['entities'] == ['entity_000','entity_001']
    assert 'SECRET' not in str(bound) and 'PRIVATE' not in str(bound)
    assert set(constraint) == {'entity_table'}
