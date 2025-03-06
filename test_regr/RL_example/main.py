from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import orL, existsL, ifL, notL, andL, atMostL, atLeastL, atMostAL, atLeastAL
from domiknows.graph.concept import EnumConcept

Graph.clear()
Concept.clear()
Relation.clear()


def get_graph(grid_size=3,max_steps=12):

    with Graph('global') as graph:
        R_Location, F_Location, Location_status = [None]*max_steps, [None]*max_steps, [None]*max_steps
        for step in range(max_steps):
            R_Location[step] = Concept(name=f'RescueHelicpter_Location_{step}', ConceptClass=EnumConcept,values=[f"{str(i)}" for i in range(grid_size*grid_size)])
            F_Location[step] = Concept(name=f'FireHelicpter_Location_{step}', ConceptClass=EnumConcept,values=[f"{str(i)}" for i in range(grid_size*grid_size)])

        for i in range(grid_size*grid_size):
            Location_status[i] = Concept(name=f'Location_status_{step}', ConceptClass=EnumConcept,values=['Clear','Fire','Rescue','Fire&Rescue','NotSetYet'])

        # Rescue Helicopter cant goto fire location
        for step in range(max_steps):
            for i in range(grid_size*grid_size):
                notL(andL(getattr(R_Location[step], f"{str(i)}" ), getattr(Location_status[step], 'Fire')))
                notL(andL(getattr(R_Location[step], f"{str(i)}"), getattr(Location_status[step], 'Fire&Rescue')))

        # Fire Helicopter and Rescue Helicopter cant be too far (manhatan distance of 2)
        for step in range(max_steps):
            for i in range(grid_size*grid_size):
                for j in range(grid_size * grid_size):
                    if abs(i//grid_size - j//grid_size) + abs(i%grid_size - j%grid_size) > 2:
                        notL(andL(getattr(R_Location[step], f"{str(i)}"), getattr(F_Location[step], f"{str(j)}")))



    return graph

