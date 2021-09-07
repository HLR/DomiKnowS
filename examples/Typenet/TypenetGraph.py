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
    mention_group = Concept(name='mention_group')
    mention = Concept(name='mention')

    mention_group_contains, = mention_group.contains(mention)

    onto_path.append("./")
    onto = get_ontology('typenet.owl').load()
    typenet_classes = list(onto.classes())

    concepts = {}

    num_struct_types = 0

    # add types in structure
    def dfs(parent, child, depth=1):
        global num_struct_types
        concepts[child.name] = Concept(name=child.name)
        if not child.name[:6] == 'Synset':
            mention[concepts[child.name]] = ReaderSensor(keyword=child.name, label=True)
            num_struct_types += 1

        if not parent == None and config.use_constraints:
            concepts[child.name].is_a(concepts[parent.name])

        for next_child in onto.get_children_of(child):
            dfs(child, next_child, depth=depth + 1)

    dfs(None, typenet_classes[0])

    concepts['Synset__entity__n__01'].is_a(mention)

    # add types not included in structure
    for t in config.missing_types:
        concepts[t] = Concept(name=t)

        if config.use_constraints:
            concepts[t].is_a(mention)

        if not t[:6] == 'Synset':
            mention[concepts[t]] = ReaderSensor(keyword=t, label=True)

    total_types = num_struct_types + len(config.missing_types)

    print('%d types in structure + %d other types = %d total types' % (num_struct_types, len(config.missing_types), total_types))

    assert total_types == config.num_types

#app_graph.visualize("./image")