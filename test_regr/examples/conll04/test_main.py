import sys
sys.path.append('../..')

import pytest

@pytest.fixture(name='case')
def test_case():
    import torch
    from regr.utils import Namespace

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    case = {
        'sentence': {
            'raw': 'John works for IBM'
        },
        'char': {
            'raw': [['J', 'o', 'h', 'n'], ['w', 'o', 'r', 'k', 's'], ['f', 'o', 'r'], ['I', 'B', 'M']]
        },
        'word': {
            'raw': ['John', 'works', 'for', 'IBM'],
            'emb': torch.randn(4, 2048, device=device),
            #                             John        works       for           IBM
            'people':       torch.tensor([[0.3, 0.7], [0.9, 0.1], [0.98, 0.02], [0.40, 0.6]], device=device),
            'organization': torch.tensor([[0.5, 0.5], [0.8, 0.2], [0.97, 0.03], [0.09, 0.91]], device=device),
            'location':     torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.95, 0.05], [0.50, 0.50]], device=device),
            'other':        torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.90, 0.10], [0.70, 0.30]], device=device),
            'O':            torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.10, 0.90], [0.90, 0.10]], device=device),
        },
        'phrase': {
            'raw': [(0, 0), (1, 2), (3, 3)],  # ['John', 'works for', 'IBM'],
            'emb': torch.randn(3, 2048, device=device),
            'people': torch.tensor([[0.3, 0.7], [0.9, 0.1], [0.40, 0.6]], device=device),

        },
        'pair': {
            'emb': torch.randn(4, 4, 2048, device=device),
            'work_for': torch.rand(4, 4, 2, device=device), # TODO: add examable values
            
            'work_for': torch.tensor([[[0.60, 0.40],         [0.80, 0.20],         [0.80, 0.20], [0.37, 0.63]],      # John
                                      [[1.00, float("nan")], [1.00, float("nan")], [0.60, 0.40], [0.70, 0.30]],  # works
                                      [[0.98, 0.02],         [0.70, 0.03],         [0.95, 0.05], [0.90, 0.10]],  # for
                                      [[0.35, 0.65],         [0.80, 0.20],         [0.90, 0.10], [0.70, 0.30]],  # IBM
                                     ], device=device),
            
            'live_in': torch.mul(torch.rand(4, 4, 2, device=device), 0.5), # TODO: add examable values
            'located_in': torch.mul(torch.rand(4, 4, 2, device=device), 0.5), # TODO: add examable values
            'orgbase_on': torch.mul(torch.rand(4, 4, 2, device=device), 0.5), # TODO: add examable values
            'kill': torch.mul(torch.rand(4, 4, 2, device=device), 0.5), # TODO: add examable values
        }
    }
    case = Namespace(case)
    return case


def model_declaration(config, case):
    from regr.program.program import LearningBasedProgram

    from graph import graph, sentence, word, char, phrase, pair
    from graph import people, organization, location, other, o
    from graph import work_for, located_in, live_in, orgbase_on, kill
    from graph import rel_sentence_contains_word, rel_phrase_contains_word, rel_word_contains_char, rel_pair_word1, rel_pair_word2
    from test_regr.sensor.pytorch.sensors import TestSensor, TestEdgeSensor

    graph.detach()

    sentence['index'] = TestSensor(expected_outputs=case.sentence.raw)

    # Edge: sentence to word forward
    rel_sentence_contains_word['forward'] = TestEdgeSensor(
        'index', mode='forward', to='index',
        expected_outputs=case.word.raw)
    word['emb'] = TestSensor(
        'index', edges=[rel_sentence_contains_word['forward']],
        expected_inputs=(case.word.raw,),
        expected_outputs=case.word.emb)

    # Edge: word to char forward
    rel_word_contains_char['forward'] = TestEdgeSensor(
        'index', mode='forward', to='index',
        edges=[rel_sentence_contains_word['forward']],
        expected_inputs=(case.word.raw,),
        expected_outputs=case.char.raw)
    char['emb'] = TestSensor(
        'index', edges=[rel_word_contains_char['forward']],
        expected_inputs=(case.char.raw,),
        expected_outputs=case.word.emb)
    char['emb'] = TestSensor(label=True)  # just to trigger calculation

    # Edge: word to phrase backward
    rel_phrase_contains_word['backward'] = TestEdgeSensor(
        'index', mode='backward', to='index',
        edges=[rel_sentence_contains_word['forward']],
        expected_inputs=(case.word.raw,),
        expected_outputs=case.phrase.raw)
    phrase['emb'] = TestSensor(
        'index', edges=[rel_phrase_contains_word['backward']],
        expected_inputs=(case.phrase.raw,),
        expected_outputs=case.phrase.emb)
    phrase['emb'] = TestSensor(label=True)  # just to trigger calculation

    # Edge: pair backward
    rel_pair_word1['backward'] = TestEdgeSensor(
        'emb', mode='backward', to='word1_emb',
        expected_inputs=(case.word.emb,),
        expected_outputs=case.word.emb)
    rel_pair_word2['backward'] = TestEdgeSensor(
        'emb', mode='backward', to='word2_emb',
        expected_inputs=(case.word.emb,),
        expected_outputs=case.word.emb)

    pair['emb'] = TestSensor(
        'word1_emb', 'word2_emb',
        edges=[rel_pair_word1['backward'], rel_pair_word2['backward']],
        expected_inputs=(case.word.emb, case.word.emb),
        expected_outputs=case.pair.emb)

    word[people] = TestSensor(
        label=True,
        expected_outputs=case.word.people)
    word[organization] = TestSensor(
        label=True,
        expected_outputs=case.word.organization)
    word[location] = TestSensor(
        label=True,
        expected_outputs=case.word.location)
    word[other] = TestSensor(
        label=True,
        expected_outputs=case.word.other)
    word[o] = TestSensor(
        label=True,
        expected_outputs=case.word.O)

    word[people] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=(case.word.emb,),
        expected_outputs=case.word.people)
    word[organization] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=(case.word.emb,),
        expected_outputs=case.word.organization)
    word[location] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=(case.word.emb,),
        expected_outputs=case.word.location)
    word[other] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=(case.word.emb,),
        expected_outputs=case.word.other)
    word[o] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=(case.word.emb,),
        expected_outputs=case.word.O)

    phrase[people] = TestSensor(
        label=True,
        expected_outputs=case.phrase.people)
    phrase[people] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=(case.phrase.emb,),
        expected_outputs=case.phrase.people)


    pair[work_for] = TestSensor(
        label=True,
        expected_outputs=case.pair.work_for)
    pair[located_in] = TestSensor(
        label=True,
        expected_outputs=case.pair.work_for)
    pair[live_in] = TestSensor(
        label=True,
        expected_outputs=case.pair.work_for)
    pair[orgbase_on] = TestSensor(
        label=True,
        expected_outputs=case.pair.work_for)
    pair[kill] = TestSensor(
        label=True,
        expected_outputs=case.pair.work_for)

    pair[work_for] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=(case.pair.emb,),
        expected_outputs=case.pair.work_for)
    pair[located_in] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=(case.pair.emb,),
        expected_outputs=case.pair.located_in)
    pair[live_in] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=(case.pair.emb,),
        expected_outputs=case.pair.live_in)
    pair[orgbase_on] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=(case.pair.emb,),
        expected_outputs=case.pair.orgbase_on)
    pair[orgbase_on] = TestSensor(
        'emb', input_dim=2048, output_dim=2,
        expected_inputs=(case.pair.emb,),
        expected_outputs=case.pair.kill)

    lbp = LearningBasedProgram(graph, **config)
    return lbp

def test_graph_naming():
    from graph import graph, sentence, word, char, phrase, pair
    from graph import people, organization, location, other, o
    from graph import work_for, located_in, live_in, orgbase_on, kill
    from graph import rel_sentence_contains_word, rel_phrase_contains_word, rel_word_contains_char, rel_pair_word1, rel_pair_word2

    # graph
    assert graph.name == 'global'

    # concepts
    assert sentence.name == 'sentence'
    assert word.name == 'word'
    assert char.name == 'char'
    assert phrase.name == 'phrase'
    assert pair.name == 'pair'

    assert people.name == 'people'
    assert organization.name == 'organization'
    assert location.name == 'location'
    assert other.name == 'other'
    assert o.name == 'O'  # Note: here is different Cap: `o = word(name='O')`

    assert work_for.name == 'work_for'
    assert located_in.name == 'located_in'
    assert live_in.name == 'live_in'
    assert orgbase_on.name == 'orgbase_on'
    assert kill.name == 'kill'

    # relation: default named
    assert rel_sentence_contains_word.name == 'sentence-contains-0-word'
    assert rel_phrase_contains_word.name == 'phrase-contains-0-word'
    assert rel_word_contains_char.name == 'word-contains-0-char'

    # relation: explicitly named
    # `(rel_pair_word1, rel_pair_word2, ) = pair.has_a(arg1=word, arg2=word)`
    assert rel_pair_word1.name == 'arg1'
    assert rel_pair_word2.name == 'arg2'


@pytest.mark.gurobi
def test_main_conll04(case):
    from config import CONFIG
    from graph import graph, sentence, word, char, phrase, pair
    from graph import people, organization, location, other, o
    from graph import work_for, located_in, live_in, orgbase_on, kill

    lbp = model_declaration(CONFIG.Model, case)
    data = {}

    _, _, datanode = lbp.model(data)

    for child_node in datanode.getChildDataNodes():
        if child_node.ontologyNode.name == 'word':
            assert child_node.getAttribute('index') == case.word.raw[child_node.instanceID]
            
            for child_node1 in child_node.getChildDataNodes():
                if child_node1.ontologyNode.name == 'char':
                    assert True
                else:
                    assert False
                       
            assert len(child_node.getChildDataNodes()) == len(case.char.raw[child_node.instanceID])
                    
            assert len(child_node.findDatanodes(select = "pair")) == 4
            
            assert (child_node.getAttribute('emb') == case.word.emb[child_node.instanceID]).all()
            assert (child_node.getAttribute('<people>') == case.word.people[child_node.instanceID]).all()
            assert (child_node.getAttribute('<organization>') == case.word.organization[child_node.instanceID]).all()
            assert (child_node.getAttribute('<location>') == case.word.location[child_node.instanceID]).all()
            assert (child_node.getAttribute('<other>') == case.word.other[child_node.instanceID]).all()
            assert (child_node.getAttribute('<O>') == case.word.O[child_node.instanceID]).all()
        elif child_node.ontologyNode.name == 'phrase':
            assert (child_node.getAttribute('emb') == case.phrase.emb[child_node.instanceID]).all()
            assert (child_node.getAttribute('<people>') == case.phrase.people[child_node.instanceID]).all()
        else:
            assert False, 'There should be only word and phrases. {} is unexpected.'.format(child_node.ontologyNode.name)

    conceptsRelationsStrings = ['people', 'organization', 'location', 'other', 'O', 'work_for']
    conceptsRelationsConcepts = [people, organization, location, other, o, work_for]
    conceptsRelationsMix = ["people", organization, location, other, o, "work_for"]
    conceptsRelationsEmpty = []
    
    conceptsRelationsVariants = [conceptsRelationsEmpty, conceptsRelationsStrings, conceptsRelationsConcepts, conceptsRelationsMix]
    
    for conceptsRelations in conceptsRelationsVariants:
        
        # ------------ Call the ILP Solver
        datanode.inferILPConstrains(*conceptsRelations, fun=None)
        
        # ------------ Concepts Results
        
        # Get value of attribute people/ILP for word 0
        #assert tokenResult['people'][0] == 1
        assert datanode.findDatanodes(select = word)[0].getAttribute(people, 'ILP').item() == 1
        
        assert datanode.findDatanodes(select = word,  indexes = {"contains" : (char, 'index', 'J')})[0].getAttribute(people, 'ILP').item() == 1
        assert datanode.findDatanodes(select = word,  indexes = {"contains" : (char, 'index', 'h')})[0].getAttribute(people, 'ILP').item() == 1
        assert datanode.findDatanodes(select = word,  indexes = {"contains" : (char, 'index', 'I')})[0].getAttribute(people, 'ILP').item() == 0
        
        assert datanode.findDatanodes(select = word,  indexes = {"contains" : ((char, 'index', 'o'), (char, 'index', 'h')) })[0].getAttribute(people, 'ILP').item() == 1
        assert datanode.findDatanodes(select = word,  indexes = {"contains" : ((char, 'index', 'o'), (char, 'index', 'h'), (char, 'index', 'n')) })[0].getAttribute(people, 'ILP').item() == 1

        assert len(datanode.findDatanodes(select = (char, 'index', 'J'))) == 1
        assert datanode.findDatanodes(select = (char, 'index', 'J'))[0].getRootDataNode() == datanode.findDatanodes(select = sentence)[0]
        
        # Sum value of attribute people/ILP for all words
        #assert sum(tokenResult['people']) == 1
        assert sum([dn.getAttribute(people, 'ILP').item() for dn in datanode.findDatanodes(select = word)]) == 1
        
        # Get value of attribute organization/ILP for word 3
        #assert tokenResult['organization'][3] == 1
        datanode.findDatanodes(select = word)[3].getAttribute(organization, 'ILP').item() == 1
        
        # Sum value of attribute organization/ILP for all words
        #assert sum(tokenResult['organization']) == 1
        assert sum([dn.getAttribute(organization, 'ILP').item() for dn in datanode.findDatanodes(select = word)]) == 1
    
        # Sum value of attribute location/ILP for all words
        #assert sum(tokenResult['location']) == 0
        assert sum([dn.getAttribute(location, 'ILP').item() for dn in datanode.findDatanodes(select = word)]) == 0
    
        # Sum value of attribute other/ILP for all words
        #assert sum(tokenResult['other']) == 0
        assert sum([dn.getAttribute(other, 'ILP').item() for dn in datanode.findDatanodes(select = word)]) == 0
    
        # Get value of attribute o/ILP for word 1
        #assert tokenResult['O'][1] == 1
        assert datanode.findDatanodes(select = word)[1].getAttribute(o, 'ILP').item() == 1
        
        JohnDN = datanode.findDatanodes(select = word)[1]
        assert JohnDN.getAttribute(organization)[0] == 0.8
        assert len(JohnDN.getChildDataNodes()) == 5
        
        assert len(JohnDN.getChildDataNodes(conceptName=char)) == 5
        assert len(JohnDN.getChildDataNodes(conceptName=phrase)) == 0
        
        assert len(JohnDN.getRelationLinks()) == 2
        assert len(JohnDN.getRelationLinks(relationName=pair)) == 4

        # Get value of attribute o/ILP for word 2
        #assert tokenResult['O'][2] == 1
        assert datanode.findDatanodes(select = word)[2].getAttribute(o, 'ILP').item() == 1
    
        # Sum value of attribute o/ILP for all words
        #assert sum(tokenResult['O']) == 2
        assert sum([dn.getAttribute(o, 'ILP').item() for dn in datanode.findDatanodes(select = word)]) == 2
        
        # ------------ Relations Results
        
        # Get value of attribute work_for/ILP for pair between 0 and 3
        #assert pairResult['work_for'][0][3] == 1
        assert datanode.findDatanodes(select = pair, indexes = {"arg1" : 0, "arg2": 3})[0].getAttribute(work_for, 'ILP').item() == 1
        
        assert datanode.findDatanodes(select = pair, indexes = {"arg1" : (word, 'index', 'John'), "arg2": (word, 'index', "IBM")})[0].getAttribute(work_for, 'ILP') == 1

        assert datanode.findDatanodes(select = pair, indexes = {"arg1" : ((word,), (word, 'index', 'John')), "arg2": (word, 'index', "IBM")})[0].getAttribute(work_for, 'ILP') == 1
        assert datanode.findDatanodes(select = pair, indexes = {"arg1" : (word, (word, 'index', 'John')), "arg2": (word, 'index', "IBM")})[0].getAttribute(work_for, 'ILP') == 1
         
        assert datanode.findDatanodes(select = pair, indexes = {"arg1" : (0, (word, 'index', 'John')), "arg2": (word, 'index', "IBM")})[0].getAttribute(work_for, 'ILP') == 1
        
        # Sum all value of attribute work_for/ILP  for the pair relation from 0
        #assert sum(pairResult['work_for'][0]) == 1
        assert sum([dn.getAttribute(work_for, 'ILP').item() for dn in datanode.findDatanodes(select = pair, indexes = {"arg1" : 0})]) == 1
        
        # Sum all value of attribute work_for/ILP  for the pair relation from 1
        #assert sum(pairResult['work_for'][1]) == 0
        assert sum([dn.getAttribute(work_for, 'ILP').item() for dn in datanode.findDatanodes(select = pair, indexes = {"arg1" : 1})]) == 0
        
        # Sum all value of attribute work_for/ILP  for the pair relation from 2
        #assert sum(pairResult['work_for'][2]) == 0
        assert sum([dn.getAttribute(work_for, 'ILP').item() for dn in datanode.findDatanodes(select = pair, indexes = {"arg1" : 2})]) == 0
    
        # Sum all value of attribute work_for/ILP  for the pair relation from 3
        #assert sum(pairResult['work_for'][3]) == 0
        assert sum([dn.getAttribute(work_for, 'ILP').item() for dn in datanode.findDatanodes(select = pair, indexes = {"arg1" : 3})]) == 0

if __name__ == '__main__':
    pytest.main([__file__])
