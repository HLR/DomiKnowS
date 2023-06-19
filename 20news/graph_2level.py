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
        "comp.os", "comp.sys", "comp.windows", "comp.graphics", "rec.motorcycles", "rec.sport", "rec.autos", "talk.religion",
        "sci.electronics", "sci.med", "sci.space", "misc.forsale", "talk.politics", "sci.crypt", "alt.atheism", "soc.religion"
    ]
    level2_list = [
        "misc", "guns", "ibm", "mac", "baseball", "hockey", "mideast", "None"
    ]
    
    level1 = news(name="level1", ConceptClass=EnumConcept, values=level1_list)
        
    level2 = news(name="level2", ConceptClass=EnumConcept, values=["misc", "guns", "ibm", "mac", "baseball", "hockey", "mideast", "None"])
    
    hierarchy_1 = {
        "talk.politics": {"misc", "guns", "mideast"},
        "comp.sys": {"ibm", "mac"}, "rec.sport": {"hockey", "baseball"}
    }
    
    for _parent in level1_list:
        if _parent in hierarchy_1.keys():
            if len(hierarchy_1[_parent]):
                ifL(level1.__getattr__(_parent), exactL(*[level2.__getattr__(key1) for key1 in hierarchy_1[_parent]]))
            else:
                ifL(level1.__getattr__(_parent), level2.__getattr__("None"))
        else:
            ifL(level1.__getattr__(_parent), level2.__getattr__("None"))