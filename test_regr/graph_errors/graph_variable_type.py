from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, andL, nandL, atMostL, existsL, notL
from domiknows.graph.concept import EnumConcept


def setup_graph(fix_constraint=False):
    Graph.clear()
    Concept.clear()
    Relation.clear()
    
    with Graph('Accident_and_Weather_Conditions') as graph:

        # Basic Concepts
        accident_details = Concept(name='accident_details')
        weather_details = Concept(name='weather_details')

        # Task Concepts
        accident_cause = accident_details(name='accident_cause', ConceptClass=EnumConcept, 
                                        values=['human error', 'mechanical failure', 'road conditions', 'weather conditions', 'other'])
        weather_condition = weather_details(name='weather_condition', ConceptClass=EnumConcept, 
                                        values=['clear', 'cloudy', 'rainy', 'snowy', 'foggy', 'stormy', 'other'])
        
        # Constraints
        # If the accident cause is weather condition, then the weather quality is not clear.
        
        if fix_constraint:
            ifL(
                accident_cause.__getattr__('weather conditions')('x'),
                notL(weather_condition.__getattr__('clear')('y'))
            )
        else:
            ifL(
                accident_cause.__getattr__('weather conditions')('x'),
                notL(weather_condition.__getattr__('clear')(path=('x'))),
                name = "testLC"
            )

    return graph