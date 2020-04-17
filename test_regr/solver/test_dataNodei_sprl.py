from itertools import combinations
import pytest
import numpy as np

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
        with Graph('linguistic') as ling_graph:
            ling_graph.ontology = ('http://ontology.ihmc.us/ML/PhraseGraph.owl', './')
            phrase = Concept(name = 'phrase')
            sentence = Concept(name = 'sentence')

        with Graph('application') as app_graph:
            app_graph.ontology = ('http://ontology.ihmc.us/ML/SPRL.owl', './examples/SpRL/')

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

    application_graph = ontology_graph['application']
    trajector = application_graph['TRAJECTOR']
    landmark = application_graph['LANDMARK']
    spatial_indicator = application_graph['SPATIAL_INDICATOR']
    none_entity = application_graph['NONE_ENTITY']
    triplet = application_graph['triplet']
    spatial_triplet = application_graph['spatial_triplet']
    region = application_graph['region']
    
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

    test_dataNode = DataNode(instanceID = 0, instanceValue = sentence0, ontologyNode = sentence)
    test_dataNode.relationLinks['contains'] = []
    for i, phrase_data in enumerate(phrases):
        phrase_node = DataNode(instanceID=i, instanceValue=phrase_data, ontologyNode=phrase)
        test_dataNode.relationLinks['contains'].append(phrase_node)
    
    stairs, about_20_kids_, about_20_kids, on, hats, traditional_clothing = test_dataNode.getChildDataNodes()

    stairs.attributes['<{}>'.format(trajector)] = [0.63, 0.37]
    stairs.attributes['<{}>'.format(landmark)] = [0.32, 0.68]
    stairs.attributes['<{}>'.format(spatial_indicator)] = [0.95, 0.05]
    stairs.attributes['<{}>'.format(none_entity)] = [0.80, 0.20]
    # stairs.predictions = {
    #     DataNode.PredictionType["Learned"] : {
    #         (triplet, (about_20_kids_, on)) : 0.85,
    #         (triplet, (about_20_kids, on)) : 0.78,
    #         (spatial_triplet, (about_20_kids_, on)) : 0.74,
    #         (spatial_triplet, (about_20_kids, on)) : 0.83,
    #         (region, (about_20_kids_, on)) : 0.65,
    #         (region, (about_20_kids, on)) : 0.88
    #         # Fill the rest of triple relation with 0.2 below
    #         }}

    about_20_kids_.attributes['<{}>'.format(trajector)] = [0.28, 0.72]
    about_20_kids_.attributes['<{}>'.format(landmark)] = [0.85, 0.15]
    about_20_kids_.attributes['<{}>'.format(spatial_indicator)] = [0.97, 0.03]
    about_20_kids_.attributes['<{}>'.format(none_entity)] = [0.39, 0.61]

    about_20_kids.attributes['<{}>'.format(trajector)] = [0.22, 0.78]
    about_20_kids.attributes['<{}>'.format(landmark)] = [0.67, 0.33]
    about_20_kids.attributes['<{}>'.format(spatial_indicator)] = [0.98, 0.02]
    about_20_kids.attributes['<{}>'.format(none_entity)] = [0.52, 0.48]

    on.attributes['<{}>'.format(trajector)] = [0.99, 0.01]
    on.attributes['<{}>'.format(landmark)] = [0.97, 0.03]
    on.attributes['<{}>'.format(spatial_indicator)] = [0.07, 0.93]
    on.attributes['<{}>'.format(none_entity)] = [0.97, 0.03]

    hats.attributes['<{}>'.format(trajector)] = [0.58, 0.42]
    hats.attributes['<{}>'.format(landmark)] = [0.57, 0.43]
    hats.attributes['<{}>'.format(spatial_indicator)] = [0.97, 0.03]
    hats.attributes['<{}>'.format(none_entity)] = [0.05, 0.95]

    traditional_clothing.attributes['<{}>'.format(trajector)] = [0.78, 0.22]
    traditional_clothing.attributes['<{}>'.format(landmark)] = [0.87, 0.13]
    traditional_clothing.attributes['<{}>'.format(spatial_indicator)] = [0.99, 0.01]
    traditional_clothing.attributes['<{}>'.format(none_entity)] = [0.48, 0.52]

    yield test_dataNode

@pytest.mark.gurobi
def test_main_sprl(ontology_graph, sprl_input):
    from regr.graph import DataNode

    application_graph = ontology_graph['application']
    trajector = application_graph['TRAJECTOR']
    landmark = application_graph['LANDMARK']
    spatial_indicator = application_graph['SPATIAL_INDICATOR']
    none_entity = application_graph['NONE_ENTITY']
    triplet = application_graph['triplet']
    spatial_triplet = application_graph['spatial_triplet']
    region = application_graph['region']

    sentence_node = sprl_input
    stairs, about_20_kids_, about_20_kids, on, hats, traditional_clothing =  sentence_node.getChildDataNodes()

    #------------------
    # Call solver on data Node for sentence 0
    #------------------
    tokenResult, pairResult, tripleResult = sentence_node.inferILPConstrains(trajector, landmark, spatial_indicator, none_entity, triplet, spatial_triplet, region)

    #------------------
    # evaluate
    #------------------
    # -- phrase results:
    # assert stairs.predictions[DataNode.PredictionType["ILP"]][landmark] == 1  # "stairs" - LANDMARK
    # assert about_20_kids_.predictions[DataNode.PredictionType["ILP"]][trajector] == 1  # "About 20 kids " - TRAJECTOR
    # assert about_20_kids.predictions[DataNode.PredictionType["ILP"]][trajector] == 1  # "About 20 kids" - TRAJECTOR

    # assert on.predictions[DataNode.PredictionType["ILP"]][spatial_indicator] == 1  # "on" - SPATIALINDICATOR
    assert sum(tokenResult[landmark.name]) == 1
    assert tokenResult[landmark.name][stairs.instanceID] == 1
    assert sum(tokenResult[trajector.name]) == 2
    assert tokenResult[trajector.name][about_20_kids_.instanceID] == 1
    assert tokenResult[trajector.name][about_20_kids.instanceID] == 1
    assert sum(tokenResult[spatial_indicator.name]) == 1
    assert tokenResult[spatial_indicator.name][on.instanceID] == 1
    assert sum(tokenResult[none_entity.name]) == 2
    assert tokenResult[none_entity.name][hats.instanceID] == 1
    assert tokenResult[none_entity.name][traditional_clothing.instanceID] == 1
    
    # -- relation results:
    # assert stairs.predictions[DataNode.PredictionType["ILP"]][triplet, (about_20_kids_, on)] == 1
    # assert stairs.predictions[DataNode.PredictionType["ILP"]][triplet, (about_20_kids, on)] == 1

    # assert stairs.predictions[DataNode.PredictionType["ILP"]][spatial_triplet, (about_20_kids_, on)] == 1
    # assert stairs.predictions[DataNode.PredictionType["ILP"]][spatial_triplet, (about_20_kids, on)] == 1

    
# Left for reference - not used anymore
def model_trial(ontology_graph, sprl_input):
    from regr.graph import Trial

    application_graph = ontology_graph['application']
    trajector = application_graph['TRAJECTOR']
    landmark = application_graph['LANDMARK']
    spatial_indicator = application_graph['SPATIAL_INDICATOR']
    none_entity = application_graph['NONE_ENTITY']
    triplet = application_graph['triplet']
    spatial_triplet = application_graph['spatial_triplet']
    region = application_graph['region']

    stairs, about_20_kids_, about_20_kids, on, hats, traditional_clothing =  sprl_input.childInstanceNodes
    #stairs, about_20_kids_, about_20_kids, on, hats, traditional_clothing = [0,1,2,3,4,5]
    #------------------
    # sample output from learners
    #------------------
    model_trial = Trial()  # model_trail should come from model run
    # phrase
    model_trial[trajector,         (stairs, )] = 0.37
    model_trial[landmark,          (stairs, )] = 0.68
    model_trial[spatial_indicator, (stairs, )] = 0.05
    model_trial[none_entity,       (stairs, )] = 0.20

    model_trial[trajector,         (about_20_kids_, )] = 0.72
    model_trial[landmark,          (about_20_kids_, )] = 0.15
    model_trial[spatial_indicator, (about_20_kids_, )] = 0.03
    model_trial[none_entity,       (about_20_kids_, )] = 0.61

    model_trial[trajector,         (about_20_kids, )] = 0.78
    model_trial[landmark,          (about_20_kids, )] = 0.33
    model_trial[spatial_indicator, (about_20_kids, )] = 0.02
    model_trial[none_entity,       (about_20_kids, )] = 0.48

    model_trial[trajector,         (on, )] = 0.01
    model_trial[landmark,          (on, )] = 0.03
    model_trial[spatial_indicator, (on, )] = 0.93
    model_trial[none_entity,       (on, )] = 0.03

    model_trial[trajector,         (hats, )] = 0.42
    model_trial[landmark,          (hats, )] = 0.43
    model_trial[spatial_indicator, (hats, )] = 0.03
    model_trial[none_entity,       (hats, )] = 0.05

    model_trial[trajector,         (traditional_clothing, )] = 0.22
    model_trial[landmark,          (traditional_clothing, )] = 0.13
    model_trial[spatial_indicator, (traditional_clothing, )] = 0.01
    model_trial[none_entity,       (traditional_clothing, )] = 0.52

    # triplet relation
    for instance in combinations(sprl_input.childInstanceNodes, 3):
        model_trial[triplet, instance] = np.random.rand() * 0.2
    model_trial[triplet, (stairs, about_20_kids_, on)] = 0.85  # GT
    model_trial[triplet, (stairs, about_20_kids, on)] = 0.78  # GT

    # spatial_triplet relation
    for instance in combinations(sprl_input.childInstanceNodes, 3):
        model_trial[spatial_triplet, instance] = np.random.rand() * 0.2
    model_trial[spatial_triplet, (stairs, about_20_kids_, on)] = 0.74  # GT
    model_trial[spatial_triplet, (stairs, about_20_kids, on)] = 0.83  # GT

    # region relation

    for instance in combinations(sprl_input.childInstanceNodes, 3):
        model_trial[region, instance] = np.random.rand() * 0.2
    model_trial[region, (stairs, about_20_kids_, on)] = 0.65  # GT
    model_trial[region, (stairs, about_20_kids, on)] = 0.88  # GT

    return model_trial

