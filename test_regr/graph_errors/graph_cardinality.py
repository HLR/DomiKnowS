from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, andL, nandL, atMostL, existsL

def setup_graph(fix_constraint=False):
    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph('global') as graph:
        with Graph('linguistic') as ling_graph:
            text_document = Concept(name='text_document')
        
        with Graph('application', auto_constraint=False) as app_graph:
            entity = Concept(name='entity')
            person = entity(name='person', auto_constraint=False)
            organization = entity(name='organization', auto_constraint=False)
            location = entity(name='location', auto_constraint=False)
            date = entity(name='date')
            other_entity_types = entity(name='other_entity_types')
        
            relation = Concept(name='relation')
            employment = relation(name='employment')
            located = relation(name='located')
            part_whole = relation(name='part-whole')
            personal_social = relation(name='personal-social')
            other_relation = relation(name='other')
        
            pair = Concept(name='pair')
            (rel_pair_entity1, rel_pair_entity2) = pair.has_a(arg1=entity, arg2=entity)
        
            if fix_constraint:
                ifL(
                    entity('x'),
                    atMostL(
                        person(path=('x')),
                        organization(path=('x')),
                        location(path=('x')),
                        date(path=('x')),
                        other_entity_types(path=('x')),
                        1
                    ),
                    name="constraint_only_one_entity_fixed"
                )
            else:
                ifL(
                    entity('x'),
                    atMostL(
                        1,
                        person(path=('x')),
                        organization(path=('x')),
                        location(path=('x')),
                        date(path=('x')),
                        other_entity_types(path=('x'))
                    ),
                    name="constraint_only_one_entity"
                )

    return graph
