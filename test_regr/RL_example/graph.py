from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import orL, existsL, ifL, notL, andL, atMostL, atLeastL, atMostAL, atLeastAL
from domiknows.graph.concept import EnumConcept

Graph.clear()
Concept.clear()
Relation.clear()


def get_graph(grid_size=3,max_steps=12):

    with Graph('global') as graph:
        R_Location, F_Location, Location_status = [None]*max_steps, [None]*max_steps, [None]*max_steps
        Location = Concept(name='Location')
        for step in range(max_steps):
            R_Location[step] = Location(name=f'RescueHelicpter_Location_{step}', ConceptClass=EnumConcept,values=[f"{str(i)}" for i in range(grid_size*grid_size)])
            F_Location[step] = Location(name=f'FireHelicpter_Location_{step}', ConceptClass=EnumConcept,values=[f"{str(i)}" for i in range(grid_size*grid_size)])

        Status = Concept(name='Statues')
        for i in range(grid_size*grid_size):
            Location_status[i] = Status(name=f'Location_status_{step}', ConceptClass=EnumConcept,values=['Clear','Fire','Rescue','FireAndRescue'])

        # Rescue Helicopter cant goto fire location
        for step in range(1,max_steps):
            for i in range(grid_size*grid_size):
                ifL(notL(existsL(*[getattr(F_Location[step_], f"{str(i)}") for step_ in range(step)])),
                    notL(andL(getattr(R_Location[step], f"{str(i)}" ), getattr(Location_status[i], 'Fire'))))

                ifL(notL(existsL(*[getattr(F_Location[step_], f"{str(i)}") for step_ in range(step)])),
                    notL(andL(getattr(R_Location[step], f"{str(i)}"), getattr(Location_status[i], 'FireAndRescue'))))

        # Fire Helicopter and Rescue Helicopter cant be too far (manhatan distance of 2)
        for step in range(max_steps):
            for i in range(grid_size*grid_size):
                for j in range(grid_size * grid_size):
                    if abs(i//grid_size - j//grid_size) + abs(i%grid_size - j%grid_size) > 2:
                        notL(andL(getattr(R_Location[step], f"{str(i)}"), getattr(F_Location[step], f"{str(j)}")))

        # All fires must be extinguished and all people must be rescued
        for i in range(grid_size * grid_size):
            ifL(getattr(Location_status[i], 'Fire'),
                existsL(*[getattr(F_Location[step], f"{str(i)}") for step in range(max_steps)]))
            ifL(getattr(Location_status[i], 'FireAndRescue'),
                existsL(*[getattr(F_Location[step], f"{str(i)}") for step in range(max_steps)]))
            ifL(getattr(Location_status[i], 'Rescue'),
                existsL(*[getattr(R_Location[step], f"{str(i)}") for step in range(max_steps)]))
            ifL(getattr(Location_status[i], 'FireAndRescue'),
                existsL(*[getattr(R_Location[step], f"{str(i)}") for step in range(max_steps)]))

    return graph, R_Location, F_Location, Location_status , Location, Status
