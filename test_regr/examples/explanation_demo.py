"""Run with python -m test_regr.examples.explanation_demo from the repository."""
import torch

from domiknows.graph import Graph, Concept, DataNode, execute, existsL, ifL


def main():
    with Graph('explanation_demo') as graph:
        scene = Concept('scene')
        item = Concept('item')
        scene.contains(item)
        selected = item(name='selected')
        eligible = item(name='eligible')
        ifL(selected('x'), eligible('x'))
        execute(existsL(selected('x')))

    root = DataNode(instanceID=0, ontologyNode=scene)
    root.current_device = 'cpu'
    child = DataNode(instanceID=0, ontologyNode=item)
    child.current_device = 'cpu'
    for concept, probability in ((selected, .8), (eligible, .3)):
        values = torch.tensor([1-probability, probability])
        child.attributes[f'<{concept.name}>'] = torch.log(values)
        child.attributes[f'<{concept.name}>/local/softmax'] = values
    root.addChildDataNode(child)
    constraint = DataNode(instanceID=0, ontologyNode=graph.get_constraint_concept())
    constraint.attributes['ELC0/label'] = torch.tensor(0)
    root.addChildDataNode(constraint)

    for mode in ('ilp', 'circuit', 'tnorm'):
        report = root.explainInference(selected, eligible, mode=mode, tnorm='P')
        print(report.render_text())
        identifier = next(p['id'] for p in report.to_dict()['predictions']
                          if p['path'].endswith('/selected'))
        alternative = report.counterfactual({identifier: True})
        print(alternative.render_text())


if __name__ == '__main__':
    main()
