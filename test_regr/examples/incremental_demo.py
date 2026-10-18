"""Run: python -m test_regr.examples.incremental_demo"""
import torch

from domiknows.graph import Graph, Concept, DataNode, ifL, existsL, execute


def main():
    with Graph('incremental_demo') as graph:
        world = Concept('world')
        item = Concept('item')
        world.contains(item)
        selected = item(name='selected')
        eligible = item(name='eligible')
        rule = ifL(selected('x'), eligible('x'))
        execute(existsL(selected('x')))
    root = DataNode(instanceID=0, ontologyNode=world)
    child = DataNode(instanceID=0, ontologyNode=item)
    for concept, p in ((selected, .8), (eligible, .7)):
        child.attributes[f'<{concept.name}>'] = torch.log(torch.tensor([1-p, p]))
        child.attributes[f'<{concept.name}>/local/softmax'] = torch.tensor([1-p, p])
    root.addChildDataNode(child)
    constraint = DataNode(instanceID=0, ontologyNode=graph.get_constraint_concept())
    constraint.attributes['ELC0/label'] = torch.tensor(0)
    root.addChildDataNode(constraint)
    for mode in ('ilp', 'circuit', 'tnorm'):
        child.attributes['<selected>/local/softmax'].copy_(torch.tensor([.2, .8]))
        rule.active = True
        with root.incrementalInference(selected, eligible, mode=mode) as session:
            first = session.infer(populate=False)
            child.attributes['<selected>/local/softmax'].copy_(torch.tensor([.9, .1]))
            changed = session.infer(populate=False)
            rule.active = False
            edited = session.infer(populate=False)
            print(mode, first.status, changed.answers, edited.answers)
            print('Update statistics:', changed.stats)


if __name__ == '__main__':
    main()
