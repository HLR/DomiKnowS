import sys
sys.path.append("../..")

import torch

from domiknows.graph import Graph, Concept, Relation
from domiknows.graph import ifL, notL, andL, orL, exactL
from domiknows.graph.concept import EnumConcept


Graph.clear()
Concept.clear()
Relation.clear()

import logging
logging.basicConfig(level=logging.INFO)

with Graph('20news') as graph:

    news_group = Concept(name="news_group")
    
    news = Concept(name = 'news')
    news_group_contains, = news_group.contains(news)
    
    level1_list = [ 
    "comp", "rec", "sci", "misc", "talk", "alt", "soc", "None"
    ]
    level2_list = [
        "os", "sys", "windows", "graphics", "motorcycles", "sport", "autos", "religion",
        "electronics", "med", "space", "forsale", "politics", "religion", "crypt", "None"
    ]
    level3_list = [
        "misc", "guns", "ibm", "mac", "baseball", "hockey", "mideast", "None"
    ]
    
    level1 = news(name="level1", ConceptClass=EnumConcept, values=["comp", "rec", "sci", "misc", "talk", "alt", "soc", "None"])
    
    level2 = news(name="level2", ConceptClass=EnumConcept, values=["os", "sys", "windows", "graphics", "motorcycles", "sport", "autos", "religion", "electronics", "med", "space", "forsale", "politics", "crypt", "None"])
    
    level3 = news(name="level3", ConceptClass=EnumConcept, values=["misc", "guns", "ibm", "mac", "baseball", "hockey", "mideast", "None"])
    
    hierarchy_1 = {"comp": {"graphics", "os", "sys", "windows"}, "rec": {"autos", "motorcycles", "sport"},
            "sci": {"crypt", "electronics", "med", "space"}, "misc": {"forsale"}, 
             "talk": {"politics", "religion"}, "alt": {}, "soc": {}, "None": {},
            }
    hierarchy_2 = {
        "windows": {}, "os": {}, "religion": {}, "politics": {"misc", "guns", "mideast"},
        "sys": {"ibm", "mac"}, "sport": {"hockey", "baseball"}
    }
    
    for _parent in level1_list:
        if _parent in hierarchy_1.keys():
            if len(hierarchy_1[_parent]):
                ifL(level1.__getattr__(_parent), exactL(*[level2.__getattr__(key1) for key1 in hierarchy_1[_parent]]))
            else:
                ifL(level1.__getattr__(_parent), andL(level2.__getattr__("None"), level3.__getattr__("None")))
        else:
            ifL(level1.__getattr__(_parent), andL(level2.__getattr__("None"), level3.__getattr__("None")))
        
        
    for _parent in level2_list:
        if _parent in hierarchy_2.keys():
            if len(hierarchy_2[_parent]):
                ifL(level2.__getattr__(_parent), exactL(*[level3.__getattr__(key1) for key1 in hierarchy_2[_parent]]))
            else:
                ifL(level2.__getattr__(_parent), level3.__getattr__("None"))
        else:
            ifL(level2.__getattr__(_parent), level3.__getattr__("None"))
