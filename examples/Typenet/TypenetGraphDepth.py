import sys
sys.path.append('../../')

from regr.graph import Graph, Concept
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.graph.concept import EnumConcept
import json
import config
from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import nandL, ifL
from itertools import combinations
from owlready2 import onto_path, get_ontology

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('application') as app_graph:
    mention = Concept(name='mention')

    with open('depth_classes.json', 'r') as class_file:
        classes = json.load(class_file)

    num_classes = sum([len(l) for l in classes]) - len(classes)
    print('%d constraint labels loaded' % (num_classes))
    #assert num_classes == config.num_types
    #label = mention(name='tag', ConceptClass=EnumConcept, values=lbl_list)

    labels = []

    # add labels at each depth
    for i, depth_classes in enumerate(classes):
        #print(depth_classes)
        labels.append(mention(name='tag_%d' % i, ConceptClass=EnumConcept, values=depth_classes))

    print('max depth:', len(labels))

    # add non constraint labels
    # NO_TYPES handled by setting all tags to NONE
    label_other = mention(name='tag_other', ConceptClass=EnumConcept, values=config.missing_types)
    print('number of missing labels added:', len(config.missing_types))

    assert config.num_types == len(config.missing_types) + num_classes

    ## add constraints

    # disjoint at each depth constraint
    for lbl_depth in labels:
        for l1, l2 in combinations(lbl_depth.attributes, 2):
            nandL(l1, l2)

    for l1, l2 in combinations(label_other.attributes, 2):
        nandL(l1, l2)


    # if child then parent constraint
    onto_path.append("./")
    onto_path.append('ontology/ML')
    onto = get_ontology('typenet.owl').load()
    typenet_classes = list(onto.classes())

    def get_concept(name, depth):
        idx = depth - 4
        assert idx >= 0

        return (labels[idx], labels[idx].get_index(name))

    def dfs(parent, child, depth=1):
        if not (parent == None or child.name[:6] == 'Synset' or parent.name[:6] == 'Synset'):
            ifL(get_concept(parent.name, depth - 1), get_concept(child.name, depth))
            #print(child, parent)
            #print(get_concept(parent.name, depth - 1), get_concept(child.name, depth))

        for next_child in onto.get_children_of(child):
            dfs(child, next_child, depth=depth + 1)

    dfs(None, typenet_classes[0])



