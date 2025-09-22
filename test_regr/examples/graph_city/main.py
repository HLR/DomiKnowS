import sys
import pytest

sys.path.append('.')
sys.path.append('../..')


def model_declaration():
    import torch

    from domiknows.sensor.pytorch.sensors import ReaderSensor
    from domiknows.sensor.pytorch.query_sensor import DataNodeReaderSensor
    from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateSensor
    from domiknows.program import LearningBasedProgram
    from domiknows.program.model.pytorch import PoiModel

    from graph import (
        graph, world, city, world_contains_city, cityLink, city1, city2, 
        firestationCity, mainFirestation, ancillaryFirestation, 
        emergencyService, groceryShop
    )
    from sensor import (
        DummyCityLearner, MainFirestationLearner, AncillaryFirestationLearner,
        EmergencyServiceLearner, GroceryShopLearner
    )

    graph.detach()

    lcConcepts = {}
    for _, lc in graph.logicalConstrains.items():
        if lc.headLC:  
            lcConcepts[lc.name] = lc.getLcConcepts()
            
    # --- City
    world['index'] = ReaderSensor(keyword='world')
    city['index'] = ReaderSensor(keyword='city')
    city[world_contains_city] = EdgeSensor(
        city['index'], world['index'], 
        relation=world_contains_city, 
        forward=lambda x, _: torch.ones_like(x).unsqueeze(-1)
    )

    # --- Neighbor
    cityLink[city1.reversed, city2.reversed] = CompositionCandidateSensor(
        city['index'], 
        relations=(city1.reversed, city2.reversed), 
        forward=lambda *_, **__: True
    )

    def readNeighbors(*_, data, datanode):
        city1_node = datanode.relationLinks[city1.name][0]
        city2_node = datanode.relationLinks[city2.name][0]
        if city1_node.getAttribute('index') in data[int(city2_node.getAttribute('index'))]:
            return True
        else:
            return False
        
    cityLink['neighbor'] = DataNodeReaderSensor(
        city1.reversed, city2.reversed, 
        keyword='links', 
        forward=readNeighbors
    )

    # --- Learners for all concepts
    city[firestationCity] = DummyCityLearner('index')
    city[mainFirestation] = MainFirestationLearner('index')
    city[ancillaryFirestation] = AncillaryFirestationLearner('index')
    city[emergencyService] = EmergencyServiceLearner('index')
    city[groceryShop] = GroceryShopLearner('index')
    
    program = LearningBasedProgram(graph, PoiModel, poi=[world, city, cityLink])
    return program


@pytest.mark.gurobi
def test_comparison_constraints():
    from reader import CityReader
    from graph import (
        city, firestationCity, mainFirestation, ancillaryFirestation,
        emergencyService, groceryShop, cityLink
    )

    lbp = model_declaration()
    dataset = CityReader().run()

    for datanode in lbp.populate(dataset=dataset):
        assert datanode != None
        assert len(datanode.getChildDataNodes()) == 9

        # Initialize attributes for all concepts
        for child_node in datanode.getChildDataNodes():
            assert child_node.ontologyNode == city
            
        # Call solver with all concepts
        conceptsRelations = (
            firestationCity, mainFirestation, ancillaryFirestation,
            emergencyService, groceryShop
        )
        datanode.inferILPResults(*conceptsRelations, fun=None, minimizeObjective=True)
        
        # Collect results
        firestation_count = 0
        main_firestation_count = 0
        ancillary_firestation_count = 0
        emergency_count = 0
        grocery_count = 0
        
        for child_node in datanode.getChildDataNodes():
            if child_node.getAttribute(firestationCity, 'ILP').item() > 0:
                firestation_count += 1
            if child_node.getAttribute(mainFirestation, 'ILP').item() > 0:
                main_firestation_count += 1
            if child_node.getAttribute(ancillaryFirestation, 'ILP').item() > 0:
                ancillary_firestation_count += 1
                firestation_count += 1
            if child_node.getAttribute(emergencyService, 'ILP').item() > 0:
                emergency_count += 1
            if child_node.getAttribute(groceryShop, 'ILP').item() > 0:
                grocery_count += 1
        
        # Verify comparison constraints
        print(f"Firestations: {firestation_count}")
        print(f"Main Firestations: {main_firestation_count}")
        print(f"Ancillary Firestations: {ancillary_firestation_count}")
        print(f"Emergency Services: {emergency_count}")
        print(f"Grocery Shops: {grocery_count}")
        
        # Test constraint validations
        assert main_firestation_count < ancillary_firestation_count  # lessL
        assert emergency_count >= firestation_count  # greaterEqL
        assert grocery_count > emergency_count  # greaterL
        assert firestation_count == main_firestation_count + ancillary_firestation_count  # equalCountsL
        assert emergency_count != grocery_count  # notEqualCountsL
        assert main_firestation_count == 1  # exactAL
        assert emergency_count >= 2  # atLeastAL
        assert grocery_count >= 3  # atLeastAL
        assert grocery_count <= 9  # lessEqL (9 cities total)

if __name__ == '__main__':
    pytest.main([__file__])