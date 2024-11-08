from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, andL, nandL, atMostL, existsL, orL

def setup_graph(fix_constraint=False):
    Graph.clear()
    Concept.clear()
    Relation.clear()
    
    GRAPH_CONSTRAIN = True  ### This can be either True or False 
    with Graph('global') as graph:
        ### The definition of a linguistic graph is based on the base concepts
        with Graph('linguistic') as ling_graph:
            text_document = Concept(name='text_document')
            
        ### Defining the graph information of the application based on the task concepts
        with Graph('application', auto_constraint=False) as app_graph:
            named_entity = Concept(name='named_entity')
            person = named_entity(name='person', auto_constraint=False)
            organization = named_entity(name='organization', auto_constraint=False)
            location = named_entity(name='location', auto_constraint=None)
            date = named_entity(name='date')
            miscellaneous = named_entity(name='miscellaneous')
            
            ### The 'pair' relationship would connect two named entities to each other
            pair = Concept(name='pair')
            ### has_a relationship is used to indicate a many-to-many relationship between two named entities stored in variables of type pair
            (rel_pair_entity1, rel_pair_entity2) = pair.has_a(arg1=named_entity, arg2=named_entity)
            
            employment = pair(name='employment')
            located = pair(name='located')
            part_whole = pair(name='part-whole')
            personal_social = pair(name='personal-social')
            other_relation = pair(name='other')
            
            
            ### Defining the constraints
            if fix_constraint:
                ifL(
                    person('x'), 
                    orL(
                        employment(path=('x', rel_pair_entity1.reversed)),
                        personal_social(path=('x', rel_pair_entity1.reversed))
                    ),
                    name="LC_person_attendance"
                )
            else:
                ifL(
                    person('x'), 
                    orL(
                        employment(path=('x', rel_pair_entity1)),
                        personal_social(path=('x', rel_pair_entity1))
                    ),
                    name="LC_person_attendance"
                )
            
    return graph
