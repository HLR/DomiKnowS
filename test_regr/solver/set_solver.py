import pytest

#------------
#  This is not a working example YET.
#  This is just to start an inference problem that is independent from learning, this can be a good test.
#  we can test both logical operations and learners on this example. This is based on classical set-coloring problem.
#
#------------

@pytest.fixture()
def ontology_graph(request):
    from regr.graph import Graph, Concept
    Graph.clear()
    Concept.clear()

    #------------------
    # sample inference setup
    #------------------
    # just for reference

    '''
    conceptNamesList = ["STATION","NoStation"]
    '''

    #------------------
    # create graphs
    #------------------
    with Graph('map') as city_Graph:
        city_Graph.ontology = ('http://ontology.ihmc.us/ML/SPRL.owl', './')
        city = Concept(name = 'city')
        neighborhood = Concept(name = 'neighborhood')
        fireStation = Concept(name = 'fireStation')
        fireStation.is_a(city)
        city.contains(neighborhood)


    yield city_Graph

    #------------------
    # tear down
    #------------------
    Graph.clear()
    Concept.clear()

@pytest.fixture()
def map_input(ontology_graph):
    from regr.graph import DataNode

    test_ont_graph = ontology_graph
    neighborhood = test_ont_graph['neightborhood']
    city = test_ont_graph['city']

    #------------------
    # sample input
    #------------------

    neighborhoodGraph = {
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 1]
    }



    test_dataNode = DataNode(instanceID = 0, instanceValue = neighborhood0, ontologyNode = neighborhood, \
                             childInstanceNodes = [DataNode(instanceID = 0, instanceValue = neighborhood0[0][0], ontologyNode = city, childInstanceNodes = []),
                                                   DataNode(instanceID = 1, instanceValue = neighborhood0[0][1], ontologyNode = city, childInstanceNodes = []),
                                                   DataNode(instanceID = 2, instanceValue = neighborhood0[0][2], ontologyNode = city, childInstanceNodes = []),
                                                   DataNode(instanceID = 3, instanceValue = neighborhood0[1][0], ontologyNode = city, childInstanceNodes = []),
                                                   DataNode(instanceID = 4, instanceValue = neighborhood0[1][1], ontologyNode = city, childInstanceNodes = []),
                                                   DataNode(instanceID = 5, instanceValue = neighborhood0[1][2], ontologyNode = city, childInstanceNodes = [])])

    yield test_dataNode

@pytest.fixture()
def model_trial(ontology_graph):
    from regr.graph import Trial

    city_graph = ontology_graph['map']
    fireStation = city_graph['fireStation']
    city = city_graph['city']
    neighborhood = city_graph['neighborhood']

    #------------------
    # sample output from learners
    #------------------
    model_trial = Trial()  # model_trail should come from model run
    # fireStation
    # technically, we can change the 0, 1, 2, ... index with any other hashable object
    model_trial[fireStation, 0] = 0.37

    model_trial[fireStation, 1] = 0.72

    model_trial[fireStation, 2] = 0.78

    model_trial[fireStation, 3] = 0.01

    model_trial[fireStation, 4] = 0.42

    return model_trial

@pytest.mark.gurobi
def test_main_sprl(ontology_graph, sprl_input, model_trial):

    city_graph = ontology_graph['map']
    firestation = city_graph['fireStation']

    test_data_node = sprl_input

    #------------------
    # Call solver on data Node for one neighborhood
    #------------------
    inference_trial = test_data_node.inferILPConstrains(model_trial,firestation)

    #------------------
    # evaluate
    #------------------
    # -- firestation resutls
    assert inference_trial[firestation, 0] == 1
    # same thing as
    assert firestation.predict(0, trial=inference_trial) == 1
    assert inference_trial[firestation, 1] == 1
    assert inference_trial[firestation, 2] == 1
    assert inference_trial[firestation, 3] == 1