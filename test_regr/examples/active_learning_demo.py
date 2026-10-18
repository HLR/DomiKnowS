"""Run: python -m test_regr.examples.active_learning_demo."""
import torch
from domiknows.graph import Graph, Concept, DataNode, ifL
from domiknows.solver.active_learning import ConstraintActiveLearner


def main():
    with Graph('annotation_demo'):
        world, item = Concept('world'), Concept('item')
        world.contains(item)
        selected, eligible = item(name='selected'), item(name='eligible')
        ifL(selected('x'), eligible('x'))
    examples = []
    for sample, probabilities in enumerate(((.9, .1), (.5, .8), (.1, .9))):
        root = DataNode(instanceID=sample, ontologyNode=world)
        child = DataNode(instanceID=0, ontologyNode=item)
        for concept, p in zip((selected, eligible), probabilities):
            values = torch.tensor([1-p, p])
            child.attributes[f'<{concept.name}>'] = values.log()
            child.attributes[f'<{concept.name}>/local/softmax'] = values
        root.addChildDataNode(child)
        examples.append(root)
    for mode in ('ilp', 'circuit', 'tnorm'):
        learner = ConstraintActiveLearner(selected, eligible, mode=mode)
        with learner.session() as session:
            first = session.select(examples, budget=1)
            print(first.render_text())
            chosen = first.to_dict()['selected'][0]
            # An annotation service supplies labels separately. This mask only
            # records which targets have been annotated and prevents reselection.
            mask = {chosen['sample_id']: [target['id'] for target in chosen['targets']]}
            second = session.select(examples, budget=1, annotations=mask)
            print('Next round:', second.to_dict()['selected'][0]['sample_id'])
            print('Reused explanations:', second.to_dict()['stats']['explanations_reused'])


if __name__ == '__main__':
    main()
