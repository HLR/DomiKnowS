from itertools import combinations
import json
import os,sys
currentdir = os.path.dirname(os.getcwd())
root = os.path.dirname(currentdir)
sys.path.append(root)

from regr.graph import Graph, Concept, Relation
from regr.graph.concept import EnumConcept
from regr.graph.logicalConstrain import nandL, exactL, ifL, orL


with open("concepts.json") as f:
    hierarchy = json.load(f)

def process_hierarchy(path):
    with open(path) as f:
        par_child = json.load(f)
    structure = dict()
    for vdummy in par_child['Thing']['children']:
        key = list(vdummy.keys())[0]
        val = vdummy[key]
        structure[key] = set()
        if 'children' in val:
            for vdummy1 in val['children']:
                key1 = list(vdummy1.keys())[0]
                val1 = vdummy1[key1]
                structure[key].add(key1)
                structure[key1] = set()
                if 'children' in val1:
                    for vdummy2 in val1['children']:
                        key2 = list(vdummy2.keys())[0]
                        val2 = vdummy2[key2]
                        structure[key1].add(key2)
                        structure[key2] = set()
                        if 'children' in val2:
                            for vdummy3 in val2['children']:
                                key3 = list(vdummy3.keys())[0]
                                val3 = vdummy3[key3]
                                structure[key2].add(key3)
    return structure

structure = process_hierarchy("hierarchy.json")



Graph.clear()
Concept.clear()
Relation.clear()


with Graph('VQA') as graph:
    image_group = Concept(name='image_group')
    image = Concept(name='image')
    image_group_contains, = image_group.contains(image)

    level1 = image(name="level1", ConceptClass=EnumConcept,
                     values=hierarchy['level1'])
    level2 = image(name="level2", ConceptClass=EnumConcept,
                  values=hierarchy['level2'] + ["None"])
    level3 = image(name="level3", ConceptClass=EnumConcept,
                  values=hierarchy['level3'] + ["None"])
    level4 = image(name="level4", ConceptClass=EnumConcept,
                  values=hierarchy['level4'] + ["None"])

    NEW_LC = True

    if NEW_LC:
        counter = 0
        for key in structure:
            if len(structure[key]):
                if key in hierarchy['level1']:
                    ifL(orL(*[level2.__getattr__(key1) for key1 in structure[key]]), level1.__getattr__(key))
                elif key in hierarchy['level2']:
                    ifL(orL(*[level3.__getattr__(key1) for key1 in structure[key]]), level2.__getattr__(key))
                elif key in hierarchy['level3']:
                    ifL(orL(*[level4.__getattr__(key1) for key1 in structure[key]]), level3.__getattr__(key))

        exactL(*[level1.__getattr__(key) for key in hierarchy['level1']])
        exactL(*[level2.__getattr__(key) for key in hierarchy['level2']])
        exactL(*[level3.__getattr__(key) for key in hierarchy['level3']])
        exactL(*[level4.__getattr__(key) for key in hierarchy['level4']])
        # print("number of relations: ", relations)