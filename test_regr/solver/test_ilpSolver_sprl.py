import pytest

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
    conceptNamesList = ["TRAJECTOR", "LANDMARK", "SPATIAL_INDICATOR", "NONE_ENTITY"]
    relationNamesList = ["triplet", "spatial_triplet", "region", "none_relation"]
    '''

    #------------------
    # create graphs
    #------------------
    with Graph('spLanguage') as splang_Graph:
        splang_Graph.ontology = ('http://ontology.ihmc.us/ML/SPRL.owl', './')

        with Graph('linguistic') as ling_graph:
            ling_graph.ontology = ('http://ontology.ihmc.us/ML/PhraseGraph.owl', './')
            phrase = Concept(name = 'phrase')
            sentence = Concept(name = 'sentence')

        with Graph('application') as app_graph:
            app_graph.ontology = ('http://ontology.ihmc.us/ML/SPRL.owl', './')

            trajector = Concept(name='TRAJECTOR')
            landmark = Concept(name='LANDMARK')
            none_entity = Concept(name='NONE_ENTITY')
            spatial_indicator = Concept(name='SPATIAL_INDICATOR')
            trajector.is_a(phrase)
            landmark.is_a(phrase)
            none_entity.is_a(phrase)
            spatial_indicator.is_a(phrase)

            triplet = Concept(name='triplet')
            triplet.has_a(first=phrase, second=phrase, third=phrase)

            spatial_triplet = Concept(name='spatial_triplet')
            spatial_triplet.is_a(triplet)
            spatial_triplet.has_a(first=landmark)
            spatial_triplet.has_a(second=trajector)
            spatial_triplet.has_a(third=spatial_indicator)

            region = Concept(name='region')
            region.is_a(spatial_triplet)
            distance = Concept(name='distance')
            distance.is_a(spatial_triplet)
            direction = Concept(name='direction')
            direction.is_a(spatial_triplet)

            none_relation= Concept(name='none_relation')
            none_relation.is_a(triplet)
            none_relation.has_a(first=landmark, second=trajector, third=spatial_indicator)

    yield splang_Graph

    #------------------
    # tear down
    #------------------
    Graph.clear()
    Concept.clear()

@pytest.fixture()
def sprl_input(ontology_graph):
    from regr.graph import DataNode

    test_ont_graph = ontology_graph
    sentence = test_ont_graph['linguistic/sentence']
    phrase = test_ont_graph['linguistic/phrase']

    #------------------
    # sample input
    #------------------
    sentence0 = "About 20 kids in traditional clothing and hats waiting on stairs ."

    # NB
    # Input to the inference in SPRL example is a bit different.
    # After processing the original sentence, only phrases (with feature extracted) remain.
    # They might not come in the original order.
    # There could be repeated token in different phrase.
    # However, those should not influence the design of inference interface.
    phrases = ["stairs",                # GT: LANDMARK
               "About 20 kids ",        # GT: TRAJECTOR
               "About 20 kids",         # GT: TRAJECTOR
               "on",                    # GT: SPATIALINDICATOR
               "hats",                  # GT: NONE
               "traditional clothing"]  # GT: NONE
    # SPRL relations are triplet with this combination
    # (landmark, trajector, spatialindicator)
    # Relation GT:
    # ("stairs", "About 20 kids ", "on") : triplet (and other? @Joslin please confirm if there is any other)
    # ("stairs", "About 20 kids", "on") : triplet

    test_phrase = [(phrase, 'NP') for phrase in phrases] # Not feasible to have POS-tag. Usually they are noun phrase

    test_dataNode = DataNode(instanceID = 0, instanceValue = sentence0, ontologyNode = sentence, \
                             childInstanceNodes = [DataNode(instanceID = 0, instanceValue = phrases[0], ontologyNode = phrase, childInstanceNodes = []),
                                                   DataNode(instanceID = 1, instanceValue = phrases[1], ontologyNode = phrase, childInstanceNodes = []),
                                                   DataNode(instanceID = 2, instanceValue = phrases[2], ontologyNode = phrase, childInstanceNodes = []),
                                                   DataNode(instanceID = 3, instanceValue = phrases[3], ontologyNode = phrase, childInstanceNodes = []),
                                                   DataNode(instanceID = 4, instanceValue = phrases[4], ontologyNode = phrase, childInstanceNodes = []),
                                                   DataNode(instanceID = 5, instanceValue = phrases[5], ontologyNode = phrase, childInstanceNodes = [])])

    yield test_dataNode

@pytest.fixture()
def model_trial(ontology_graph):
    from regr.graph import Trial

    application_graph = ontology_graph['application']
    trajector = application_graph['TRAJECTOR']
    landmark = application_graph['LANDMARK']
    spatial_indicator = application_graph['SPATIAL_INDICATOR']
    none_entity = application_graph['NONE_ENTITY']
    triplet = application_graph['triplet']
    spatial_triplet = application_graph['spatial_triplet']

    #------------------
    # sample output from learners
    #------------------
    model_trial = Trial()  # model_trail should come from model run
    # phrase
    # technically, we can change the 0, 1, 2, ... index with any other hashable object
    model_trial[trajector,         0] = 0.37
    model_trial[landmark,          0] = 0.68
    model_trial[spatial_indicator, 0] = 0.05
    model_trial[none_entity,       0] = 0.20

    model_trial[trajector,         1] = 0.72
    model_trial[landmark,          1] = 0.15
    model_trial[spatial_indicator, 1] = 0.03
    model_trial[none_entity,       1] = 0.61

    model_trial[trajector,         2] = 0.78
    model_trial[landmark,          2] = 0.33
    model_trial[spatial_indicator, 2] = 0.02
    model_trial[none_entity,       2] = 0.48

    model_trial[trajector,         3] = 0.01
    model_trial[landmark,          3] = 0.03
    model_trial[spatial_indicator, 3] = 0.93
    model_trial[none_entity,       3] = 0.03

    model_trial[trajector,         4] = 0.42
    model_trial[landmark,          4] = 0.43
    model_trial[spatial_indicator, 4] = 0.03
    model_trial[none_entity,       4] = 0.05

    model_trial[trajector,         5] = 0.22
    model_trial[landmark,          5] = 0.13
    model_trial[spatial_indicator, 5] = 0.01
    model_trial[none_entity,       5] = 0.52

    # triplet relation
    model_trial[triplet, (0, 1, 3)] = 0.85  # ("stairs", "About 20 kids ", "on") - GT
    model_trial[triplet, (0, 2, 3)] = 0.78  # ("stairs", "About 20 kids", "on") - GT
    model_trial[triplet, (0, 4, 3)] = 0.42  # ("stairs", "hat", "on")
    model_trial[triplet, (0, 5, 3)] = 0.85  # ("stairs", "traditional clothing", "on")
    model_trial[triplet, (1, 0, 3)] = 0.32  # ("About 20 kids ", "stairs", "on")
    model_trial[triplet, (1, 2, 3)] = 0.30  # ("About 20 kids ", "About 20 kids", "on")
    model_trial[triplet, (1, 4, 3)] = 0.32  # ("About 20 kids ", "hat", "on")
    model_trial[triplet, (1, 5, 3)] = 0.29  # ("About 20 kids ", "traditional clothing", "on")
    model_trial[triplet, (2, 0, 3)] = 0.31  # ("About 20 kids", "stairs", "on")
    model_trial[triplet, (2, 1, 3)] = 0.30  # ("About 20 kids", "About 20 kids ", "on")
    model_trial[triplet, (2, 4, 3)] = 0.25  # ("About 20 kids ", "hat", "on")
    model_trial[triplet, (2, 5, 3)] = 0.27  # ("About 20 kids ", "traditional clothing", "on")

    # spatial_triplet relation
    model_trial[spatial_triplet, (0, 1, 3)] = 0.25  # ("stairs", "About 20 kids ", "on") - GT
    model_trial[spatial_triplet, (0, 2, 3)] = 0.38  # ("stairs", "About 20 kids", "on") - GT
    model_trial[spatial_triplet, (0, 4, 3)] = 0.22  # ("stairs", "hat", "on")
    model_trial[spatial_triplet, (0, 5, 3)] = 0.39  # ("stairs", "traditional clothing", "on")
    model_trial[spatial_triplet, (1, 0, 3)] = 0.32  # ("About 20 kids ", "stairs", "on")
    model_trial[spatial_triplet, (1, 2, 3)] = 0.30  # ("About 20 kids ", "About 20 kids", "on")
    model_trial[spatial_triplet, (1, 4, 3)] = 0.22  # ("About 20 kids ", "hat", "on")
    model_trial[spatial_triplet, (1, 5, 3)] = 0.39  # ("About 20 kids ", "traditional clothing", "on")
    model_trial[spatial_triplet, (2, 0, 3)] = 0.31  # ("About 20 kids", "stairs", "on")
    model_trial[spatial_triplet, (2, 1, 3)] = 0.30  # ("About 20 kids", "About 20 kids ", "on")
    model_trial[spatial_triplet, (2, 4, 3)] = 0.15  # ("About 20 kids ", "hat", "on")
    model_trial[spatial_triplet, (2, 5, 3)] = 0.27  # ("About 20 kids ", "traditional clothing", "on")

    # region relation
    model_trial[spatial_triplet, (0, 1, 3)] = 0.25  # ("stairs", "About 20 kids ", "on") - GT
    model_trial[spatial_triplet, (0, 2, 3)] = 0.38  # ("stairs", "About 20 kids", "on") - GT
    model_trial[spatial_triplet, (0, 4, 3)] = 0.22  # ("stairs", "hat", "on")
    model_trial[spatial_triplet, (0, 5, 3)] = 0.12  # ("stairs", "traditional clothing", "on")
    model_trial[spatial_triplet, (1, 0, 3)] = 0.32  # ("About 20 kids ", "stairs", "on")
    model_trial[spatial_triplet, (1, 2, 3)] = 0.30  # ("About 20 kids ", "About 20 kids", "on")
    model_trial[spatial_triplet, (1, 4, 3)] = 0.22  # ("About 20 kids ", "hat", "on")
    model_trial[spatial_triplet, (1, 5, 3)] = 0.39  # ("About 20 kids ", "traditional clothing", "on")
    model_trial[spatial_triplet, (2, 0, 3)] = 0.31  # ("About 20 kids", "stairs", "on")
    model_trial[spatial_triplet, (2, 1, 3)] = 0.30  # ("About 20 kids", "About 20 kids ", "on")
    model_trial[spatial_triplet, (2, 4, 3)] = 0.15  # ("About 20 kids ", "hat", "on")
    model_trial[spatial_triplet, (2, 5, 3)] = 0.27  # ("About 20 kids ", "traditional clothing", "on")

    # none_relation
    model_trial[spatial_triplet, (0, 1, 3)] = 0.15  # ("stairs", "About 20 kids ", "on")
    model_trial[spatial_triplet, (0, 2, 3)] = 0.08  # ("stairs", "About 20 kids", "on")

    return model_trial

@pytest.mark.gurobi
def test_main_sprl(ontology_graph, sprl_input, model_trial):
    print("You are here!")
    application_graph = ontology_graph['application']
    trajector = application_graph['TRAJECTOR']
    landmark = application_graph['LANDMARK']
    spatial_indicator = application_graph['SPATIAL_INDICATOR']
    none_entity = application_graph['NONE_ENTITY']
    triplet = application_graph['triplet']
    spatial_triplet = application_graph['spatial_triplet']
    none_relation = application_graph['none_relation']

    test_data_node = sprl_input

    #------------------
    # Call solver on data Node for sentence 0
    #------------------
    inference_trial = test_data_node.inferILPConstrains(model_trial, trajector, landmark, spatial_indicator, none_entity, triplet, spatial_triplet, none_relation)

    #------------------
    # evaluate
    #------------------
    # -- phrase results:
    assert inference_trial[landmark, 0] == 1  # "stairs" - LANDMARK
    # same thing as
    assert landmark.predict(0, trial=inference_trial) == 1  # "stairs" - LANDMARK

    assert inference_trial[trajector, 1] == 1  # "About 20 kids " - TRAJECTOR
    assert inference_trial[trajector, 2] == 1  # "About 20 kids" - TRAJECTOR

    assert inference_trial[spatial_indicator, 3] == 1  # "on" - SPATIALINDICATOR

    assert inference_trial[none_entity, 4] == 1 # "hats" - NONE
    assert inference_trial[none_entity, 5] == 1 # "traditional clothing" - NONE

    # -- relation results:

    assert inference_trial[triplet, (0, 1, 3)] == 1 # ("stairs", "About 20 kids ", "on")
    assert inference_trial[triplet, (0, 2, 3)] == 1 # ("stairs", "About 20 kids", "on")

    assert inference_trial[none_relation, (0, 1, 3)] == 0 # ("stairs", "About 20 kids ", "on")
    assert inference_trial[none_relation, (0, 2, 3)] == 0 # ("stairs", "About 20 kids", "on")
