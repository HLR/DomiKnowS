"""Synthetic review workflow: python -m test_regr.examples.constraint_induction_demo.

The explicit approval below simulates a human review of this toy domain. A real
application must obtain that decision from its reviewer before calling approve.
"""
import torch
from domiknows.graph import Graph, Concept, DataNode
from domiknows.solver.constraint_induction import ConstraintInducer


def main():
    with Graph('induction_demo') as graph:
        world, item = Concept('world'), Concept('item')
        world.contains(item)
        person, employee, organization = [item(name=name) for name in ('person','employee','organization')]
    examples = []
    for index in range(8):
        root = DataNode(instanceID=index, ontologyNode=world)
        node = DataNode(instanceID=0, ontologyNode=item)
        root.addChildDataNode(node)
        for concept, label, probability in ((person,1,.1),(employee,1,.9),(organization,0,.1)):
            values = torch.tensor([1-probability,probability])
            node.attributes[f'<{concept.name}>/label'] = torch.tensor(label)
            node.attributes[f'<{concept.name}>/local/softmax'] = values
            node.attributes[f'<{concept.name}>'] = values.log()
        examples.append(root)
    inducer = ConstraintInducer(person, employee, organization)
    proposals = inducer.propose(examples[:4], validation_examples=examples[4:],
        guidance=['Employee implies person.',
                  'A person cannot also be an organization.',
                  'At most 1 employee per example.'],
        complete_domains={i:['employee'] for i in range(8)})
    guided = [c for c in proposals.to_dict()['candidates'] if c['origins'][0]['method']=='guidance']
    print(proposals.render_text())
    print(proposals.preview([c['id'] for c in guided], examples[:1]).render_text())
    candidate = next(c for c in guided if c['specification']['template']=='implication')
    decision = proposals.approve(candidate['id'], reviewer='demo-reviewer',
        rationale='Reviewed the toy domain definition and held-out annotations.')
    activation = proposals.activate(decision, graph)
    print('Activated:', activation['rule_path'])
    review = graph.logicalConstrains[activation['rule_path'].rsplit('/',1)[-1]].induction_provenance['decision']
    print('Review provenance:', {key: review[key] for key in ('reviewer','rationale','validation_source')})
    proposals.deactivate(activation, reviewer='demo-reviewer', rationale='End the demonstration.')
    print('Audit events:', [e['action'] for e in proposals.to_dict()['events']])
    # Save proposals.to_json(indent=2) for a detached audit artifact. Importing
    # JSON does not reconstruct trusted approval objects or activate rules.


if __name__ == '__main__':
    main()
