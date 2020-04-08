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
            #ling_graph.ontology = ('http://ontology.ihmc.us/ML/PhraseGraph.owl', './')
            phrase = Concept(name = 'phrase')
            sentence = Concept(name = 'sentence')

        with Graph('application') as app_graph:
            #app_graph.ontology = ('http://ontology.ihmc.us/ML/SPRL.owl', './examples/SpRL/')

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

    test_dataNode = DataNode(instanceID = 0, instanceValue = sentence0, ontologyNode = sentence, \
                             childInstanceNodes = [DataNode(instanceID = 0, instanceValue = phrases[0], ontologyNode = phrase, childInstanceNodes = []),
                                                   DataNode(instanceID = 1, instanceValue = phrases[1], ontologyNode = phrase, childInstanceNodes = []),
                                                   DataNode(instanceID = 2, instanceValue = phrases[2], ontologyNode = phrase, childInstanceNodes = []),
                                                   DataNode(instanceID = 3, instanceValue = phrases[3], ontologyNode = phrase, childInstanceNodes = []),
                                                   DataNode(instanceID = 4, instanceValue = phrases[4], ontologyNode = phrase, childInstanceNodes = []),
                                                   DataNode(instanceID = 5, instanceValue = phrases[5], ontologyNode = phrase, childInstanceNodes = [])]) 
    
    stairs, about_20_kids_, about_20_kids, on, hats, traditional_clothing =  test_dataNode.childInstanceNodes
    
    test_dataNode.childInstanceNodes[0].predictions = {DataNode.PredictionType["Learned"] : { (trajector) : 0.37,
                                                                                              (landmark) : 0.68,
                                                                                              (spatial_indicator) : 0.05,
                                                                                              (none_entity) : 0.20,
                                                                    
                                                                                              (triplet, (about_20_kids_, on)) : 0.85,
                                                                                              (triplet, (about_20_kids, on)) : 0.78,
                                                                                              (spatial_triplet, (about_20_kids_, on)) : 0.74,
                                                                                              (spatial_triplet, (about_20_kids, on)) : 0.83,
                                                                                              (region, (about_20_kids_, on)) : 0.65,
                                                                                              (region, (about_20_kids, on)) : 0.88
                                                                                              # Fill the rest of triple relation with 0.2 below
                                                                                            }
                                                      }
        
    
    test_dataNode.childInstanceNodes[1].predictions = {DataNode.PredictionType["Learned"] : { (trajector) : 0.72,
                                                                                              (landmark) : 0.15,
                                                                                              (spatial_indicator) : 0.03,
                                                                                              (none_entity) : 0.61
                                                                                            }
                                                       }
    
    test_dataNode.childInstanceNodes[2].predictions = {DataNode.PredictionType["Learned"] : { (trajector) : 0.78,
                                                                                              (landmark) : 0.33,
                                                                                              (spatial_indicator) : 0.02,
                                                                                              (none_entity) : 0.48
                                                                                            }
                                                       }

    test_dataNode.childInstanceNodes[3].predictions = {DataNode.PredictionType["Learned"] : { (trajector) : 0.01,
                                                                                              (landmark) : 0.03,
                                                                                              (spatial_indicator) : 0.93,
                                                                                              (none_entity) : 0.03
                                                                                            }
                                                       }
    
    test_dataNode.childInstanceNodes[4].predictions = {DataNode.PredictionType["Learned"] : { (trajector) : 0.42,
                                                                                              (landmark) : 0.43,
                                                                                              (spatial_indicator) : 0.03,
                                                                                              (none_entity) : 0.05
                                                                                            }
                                                       }
    
    test_dataNode.childInstanceNodes[5].predictions = {DataNode.PredictionType["Learned"] : { (trajector) : 0.22,
                                                                                              (landmark) : 0.13,
                                                                                              (spatial_indicator) : 0.01,
                                                                                              (none_entity) : 0.52
                                                                                            }
                                                       }
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
    stairs, about_20_kids_, about_20_kids, on, hats, traditional_clothing =  sentence_node.childInstanceNodes

    #------------------
    # Call solver on data Node for sentence 0
    #------------------
    sentence_node.inferILPConstrains(trajector, landmark, spatial_indicator, none_entity, triplet, spatial_triplet, region)

    #------------------
    # evaluate
    #------------------
    # -- phrase results:
    assert stairs.predictions[DataNode.PredictionType["ILP"]][landmark] == 1  # "stairs" - LANDMARK
    assert about_20_kids_.predictions[DataNode.PredictionType["ILP"]][trajector] == 1  # "About 20 kids " - TRAJECTOR
    assert about_20_kids.predictions[DataNode.PredictionType["ILP"]][trajector] == 1  # "About 20 kids" - TRAJECTOR

    assert on.predictions[DataNode.PredictionType["ILP"]][spatial_indicator] == 1  # "on" - SPATIALINDICATOR

    # -- relation results:
    assert stairs.predictions[DataNode.PredictionType["ILP"]][triplet, (about_20_kids_, on)] == 1
    assert stairs.predictions[DataNode.PredictionType["ILP"]][triplet, (about_20_kids, on)] == 1

    assert stairs.predictions[DataNode.PredictionType["ILP"]][spatial_triplet, (about_20_kids_, on)] == 1
    assert stairs.predictions[DataNode.PredictionType["ILP"]][spatial_triplet, (about_20_kids, on)] == 1