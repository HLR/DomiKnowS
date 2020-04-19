from itertools import combinations
import pytest
import numpy as np
from regr.graph import Graph, Concept, andL, nandL, notL, ifL, existsL, equalA, inSetA


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
    conceptNamesList = ["SPATIAL_ENTITY", "NONE_ENTITY"]
    attributeDictionary = { 'hasSpatialType' : ('trajector', 'landmark', 'spatial_indicator')}
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

            none_entity = Concept(name='NONE_ENTITY')
            spatial_entity = Concept(name='SPATIAL_ENTITY')
            
            none_entity.is_a(phrase)
            spatial_entity.is_a(phrase)
            
            nandL(none_entity, spatial_entity)
            
            #nandL(spatial_entity, equalA('hasSpatialType', 'trajector'), spatial_entity, equalA('hasSpatialType', 'landmark'))
            
            nandL(equalA('hasSpatialType', 'trajector'), equalA('hasSpatialType', 'landmark'))
            
            # Alternative            
            nandL(equalA(spatial_entity, 'hasSpatialType', 'trajector'),  equalA(spatial_entity, 'hasSpatialType', 'landmark'))
            nandL(inSetA(spatial_entity, 'hasSpatialType', ('trajector', 'spatial_indicator')),  equalA(spatial_entity, 'hasSpatialType', 'landmark'))

            nandL(spatial_entity, equalA('hasSpatialType', 'trajector'), spatial_entity, equalA('hasSpatialType', 'spatial_indicator'))
            nandL(spatial_entity, equalA('hasSpatialType', 'landmark'), spatial_entity, equalA('hasSpatialType', 'spatial_indicator'))

            triplet = Concept(name='triplet')
            triplet.has_a(first=phrase, second=phrase, third=phrase)
            
            spatial_triplet = Concept(name='spatial_triplet')
            spatial_triplet.is_a(triplet)
       
            ifL(spatial_triplet, ('x','y', 'z'), equalA('hasSpatialType', 'landmark'), ('x',))
            ifL(spatial_triplet, ('x','y', 'z'), equalA('hasSpatialType', 'trajector'), ('y',))
            ifL(spatial_triplet, ('x','y', 'z'), equalA('hasSpatialType', 'spatial_indicator'), ('z',))

            none_relation= Concept(name='none_relation')
            none_relation.is_a(triplet)
            
            ifL(none_relation, ('x','y', 'z'),  equalA('hasSpatialType', 'landmark'), ('x',))
            ifL(none_relation, ('x','y', 'z'),  equalA('hasSpatialType', 'trajector'), ('y',))
            ifL(none_relation, ('x','y', 'z'),  equalA('hasSpatialType', 'spatial_indicator'), ('z',))
            
            region = Concept(name='region')
            region.is_a(spatial_triplet)
            
            distance = Concept(name='distance')
            distance.is_a(spatial_triplet)
            
            direction = Concept(name='direction')
            direction.is_a(spatial_triplet)

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
    spatial_entity = application_graph['SPATIAL_ENTITY']
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
    
    stairs.predictions = {DataNode.PredictionType["Learned"] : { (spatial_entity,) : 0.93,
                                                                 (none_entity,) : 0.61,
                                        
                                                                 (triplet, (about_20_kids_, on)) : 0.85,
                                                                 (triplet, (about_20_kids, on)) : 0.78,
                                                                 (spatial_triplet, (about_20_kids_, on)) : 0.74,
                                                                 (spatial_triplet, (about_20_kids, on)) : 0.83,
                                                                 (region, (about_20_kids_, on)) : 0.65,
                                                                 (region, (about_20_kids, on)) : 0.88
                                                                }
                                                      }
    
    stairs.attributes = {DataNode.PredictionType["Learned"] :  {'hasSpatialType': {  ('trajector',) : 0.37,
                                                                                    ('landmark',) : 0.68,
                                                                                    ('spatial_indicator',) : 0.05
                                                                                 }
                                                           }
                                                      }
    
    yield test_dataNode

@pytest.mark.skip(reason="Not ready yet - prototype")
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
    assert stairs.predictions[DataNode.PredictionType["ILP"]][spatial_entity] == 1  # "stairs" - LANDMARK
    assert stairs.attributes[DataNode.PredictionType["ILP"]]['hasSpatialType']['landmark'] == 1  # "stairs" - LANDMARK
