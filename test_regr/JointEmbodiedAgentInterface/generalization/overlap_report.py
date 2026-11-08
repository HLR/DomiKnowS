"""Semantic capability review plus measured rule opportunities and rejections."""
from .shared_constraints import SHARED_RULES, EXCLUDED_NONUNIVERSAL


def overlap_report(manifest, reports):
    inventories = {key: [{'name': r['name'], 'active': r['active'], 'formula': r['formula']}
                        for r in manifest['components'][key]['rules']]
                   for key in ('eai_world', 'eai_generation', 'vlabench_world', 'vlabench_generation')}
    families = [
        {'family':'sequence_frame_and_action_signature',
         'EAI':'native token frame and action arity; different action vocabulary',
         'VLABench':'native graph grammar plus operation signature LCs',
         'LIBERO':'adapter reuses VLABench grammar; native LIBERO supplies continuous actions, not graph plans',
         'RLBench':'continuous action-mode contract; no graph-token adapter in this campaign'},
        {'family':'entity_reference_integrity',
         'EAI':'generated_object_exists_in_task_world, grounded from task observations',
         'VLABench':'operation_pointer_is_valid, grounded from the entity table',
         'LIBERO':'same pointer LC with canonical all-entity table, independent of the goal',
         'RLBench':'scene object handles exist; mapping to planner tokens not implemented'},
        {'family':'anti_self_placement', 'EAI':'not asserted equivalent without action-role proof',
         'VLABench':'anti_self_placement', 'LIBERO':'same compiled LC and pick/place role meanings',
         'RLBench':'potentially groundable, not declared/enforced by this adapter'},
        {'family':'static_fixture_affordance', 'EAI':'task argument/precondition rules; no universal skill-name equivalence',
         'VLABench':'pick_excludes_static_fixtures', 'LIBERO':'same LC; fixture labels available in input ontology',
         'RLBench':'scene geometry available; adapter/proof absent'},
        {'family':'holding_and_closed_container_preconditions',
         'EAI':'native source-holding and destination-open-if-known LCs',
         'VLABench':'native task/pattern and observed-access constraints; not all are universal',
         'LIBERO':'holding checked by abstract scorer/executor; not presented as a reused graph LC',
         'RLBench':'native physics and task success conditions; no shared LC implementation'},
        {'family':'spatial_and_containment_goals',
         'EAI':'SimpleTL goal/state predicates', 'VLABench':'scene/Complex task facts',
         'LIBERO':'BDDL On/In goals held by scorer only; parent relations are public observations',
         'RLBench':'task success conditions; not equivalent to generation constraints'},
        {'family':'task_sequences_and_temporal_obligations',
         'EAI':'native temporal goals', 'VLABench':'native task-specific templates and obligations',
         'LIBERO':'no VLABench task template declared; native goal is not a generation input',
         'RLBench':'task-specific conditions and demonstrations; no transfer mapping'},
        {'family':'physical_property_QA', 'EAI':'not the VLABench material-order question protocol',
         'VLABench':'metadata-assisted density/friction/etc. rule families',
         'LIBERO':'not supplied by Spatial/Object suites', 'RLBench':'no matching six-dimension QA protocol'},
    ]
    measured = {}
    for key, report in reports.items():
        measured[key] = {'examples': report['examples'],
            'declared_profile': sorted(SHARED_RULES) if key.endswith('shared-verified') else key.split(':')[-1],
            'groundable': ('privileged native task identity and metadata facts' if key.endswith('full-metadata')
                           else 'no generation constraints' if key.endswith(':none')
                           else 'pointer/fixture/role facts from all-entity input; no held-out task targets'),
            'applicable_event_denominator': 'candidate-transition events matching each rule domain and phase',
            'applicable_events': report.get('applicable_events', {}),
            'actually_triggered_rejections': report.get('triggered_rejections', {})}
    active_vla = sum(r['active'] for r in inventories['vlabench_world'])
    return {'version': 1, 'inventories': inventories, 'families': families, 'measured': measured,
        'verified_transfer': {'vlabench_libero_world_rules': sorted(SHARED_RULES),
            'whitelisted_world_rule_count': len(SHARED_RULES), 'native_vlabench_active_world_head_rules': active_vla,
            'native_literal_EAI_VLABench_rule_equivalence_claimed': False,
            'common_external_executable_skills': ['pick', 'place'],
            'external_tasks':20, 'external_evaluation_episodes':100,
            'excluded_nonuniversal':EXCLUDED_NONUNIVERSAL},
        'limitations':['No cross-dataset literal-name Jaccard is used as a semantic equivalence claim.',
                      'Observed goals/physics do not imply a corresponding generation constraint is declared.',
                      'RLBench is reviewed for feasibility only; no execution results are reported.',
                      'External physical success uses a calibrated fixed executor, not the learned controller.']}
