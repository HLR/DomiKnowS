from itertools import combinations
import json
import sys
sys.path.append(".")
sys.path.append("../..")

from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.concept import EnumConcept
from domiknows.graph.logicalConstrain import nandL, exactL, ifL, orL, andL, notL


with open("Tasks/VQA/concepts.json") as f:
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

structure = process_hierarchy("Tasks/VQA/hierarchy.json")



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
                    # ifL(orL(*[level2.__getattr__(key1) for key1 in structure[key]]), level1.__getattr__(key))
                    ifL(level1.__getattr__(key), exactL(*[level2.__getattr__(key1) for key1 in structure[key]]))
                elif key in hierarchy['level2']:
                    # ifL(orL(*[level3.__getattr__(key1) for key1 in structure[key]]), level2.__getattr__(key))
                    ifL(level2.__getattr__(key), exactL(*[level3.__getattr__(key1) for key1 in structure[key]]))
                elif key in hierarchy['level3']:
                    # ifL(orL(*[level4.__getattr__(key1) for key1 in structure[key]]), level3.__getattr__(key))
                    ifL(level3.__getattr__(key), exactL(*[level4.__getattr__(key1) for key1 in structure[key]]))
            else:
                if key in hierarchy['level1']:
                    ifL(level1.__getattr__(key), andL(level2.__getattr__('None'), level3.__getattr__('None'), level4.__getattr__('None')))
                elif key in hierarchy['level2']:
                    ifL(level2.__getattr__(key), andL(level3.__getattr__('None'), level4.__getattr__('None')))
                elif key in hierarchy['level3']:
                    ifL(level3.__getattr__(key), level4.__getattr__('None'))


                

        # ifL(level1.__getattr__('None'), andL(*[level2.__getattr__('None'), level3.__getattr__('None'), level4.__getattr__('None')]))
        ifL(
            level2.__getattr__('None'), 
            andL(*[level3.__getattr__('None'), level4.__getattr__('None')])
        )
        
        ifL(
            notL(level2.__getattr__('None')), 
            ifL(
                level3.__getattr__('None'), 
                level4.__getattr__('None')
                )
        )

        # ifL(
        #     level1.__getattr__('None'), 
        #     level2.__getattr__('None')
        # )
        # ifL(
        #     level2.__getattr__('None'), 
        #     level3.__getattr__('None')
        # )
        # ifL(
        #     level3.__getattr__('None'), 
        #     level4.__getattr__('None')
        # )

        exactL(*[level1.__getattr__(key) for key in hierarchy['level1']])
        exactL(*[level2.__getattr__(key) for key in hierarchy['level2'] + ["None"]])
        exactL(*[level3.__getattr__(key) for key in hierarchy['level3'] + ["None"]])
        exactL(*[level4.__getattr__(key) for key in hierarchy['level4'] + ["None"]])
        # print("number of relations: ", relations)