import sys
from itertools import product
sys.path.append('.')
sys.path.append('../../..')

import pytest


@pytest.fixture(name='case')
def test_case():
    import torch
    from regr.utils import Namespace

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word_emb = torch.randn(4, 2048, device=device)
    # pcw = torch.tensor([[1, 0, 0, 0],
    #                     [0, 1, 1, 0],
    #                     [0, 0, 0, 1,]], device=device)
    # pairs_link = torch.randn(4, 4, device=device)
    # pairs_link *= 1 - torch.eye(4, 4, device=device)
    # pairs_link[3, 0] = 1  # make sure John - IBM
    # pairs_link = pairs_link > 0
    # arg1, arg2 = pairs_link.nonzero(as_tuple=True)
    # pa1 = torch.zeros(arg1.shape[0], 4, device=device)
    # pa1.scatter_(1, arg1.unsqueeze(-1), 1)
    # pa2 = torch.zeros(arg2.shape[0], 4, device=device)
    # pa2.scatter_(1, arg2.unsqueeze(-1), 1)
    # pa1_emb = pa1.matmul(word_emb)
    # pa2_emb = pa2.matmul(word_emb)
    # pair_emb = torch.cat((pa1_emb, pa2_emb), dim=-1)

    case = {
        'sentence': {
            'raw': 'John works for IBM'
        },
        'word': {
            'scw': torch.tensor([[1], [1], [1], [1]], device=device),
            'raw': ['John', 'works', 'for', 'IBM'],
            'emb': word_emb,
            'Oword': torch.tensor([[0.6, 0.4],
                                  [0.8, 0.2],
                                  [0.55, 0.45],
                                  [0.7, 0.3]],
                                  device=device),
            'Bword': torch.tensor([[0.3, 0.7],
                                   [0.4, 0.6],
                                   [0.45, 0.55],
                                   [0.2, 0.8]],
                                  device=device),
            'Iword': torch.tensor([[0.1, 0.9],
                                   [0.3, 0.7],
                                   [0.2, 0.8],
                                   [0.1, 0.9]],
                                  device=device),
            'Eword': torch.tensor([[0.6, 0.4],
                                   [0.4, 0.6],
                                   [0.1, 0.9],
                                   [0.5, 0.5]],
                                  device=device),
        },
        'phrase': {
            'scw': torch.tensor([[1], [1], [1]], device=device),
            'raw': ['John', 'works for', 'IBM'],
            'pw1_backward': torch.tensor([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]], device=device),
            'pw2_backward': torch.tensor([[1, 0, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], device=device),
            'emb': torch.stack([
                torch.cat((word_emb[0], word_emb[0]), dim=0),
                torch.cat((word_emb[1], word_emb[2]), dim=0),
                torch.cat((word_emb[3], word_emb[3]), dim=0)], dim=0),
            #                             John        "works for"           IBM
            'people': torch.tensor([[0.3, 0.7], [0.98, 0.02], [0.40, 0.6]], device=device),
            'organization': torch.tensor([[0.5, 0.5],  [0.97, 0.03], [0.09, 0.91]], device=device),
            'location': torch.tensor([[0.7, 0.3], [0.95, 0.05], [0.50, 0.50]], device=device),
            'other': torch.tensor([[0.7, 0.3], [0.90, 0.10], [0.70, 0.30]], device=device),
            'O': torch.tensor([[0.9, 0.1], [0.10, 0.90], [0.90, 0.10]], device=device),
        },
        'pair': {
            #                             John-"works for"   John-IBM   IBM-John   IBM-"works for"
            'pa1_backward': torch.tensor([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]], device=device),
            'pa2_backward': torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]], device=device),
            'emb': torch.stack([
                torch.cat((torch.cat((word_emb[0], word_emb[0]), dim=0), torch.cat((word_emb[1], word_emb[2]), dim=0)), dim=0),
                torch.cat((torch.cat((word_emb[0], word_emb[0]), dim=0), torch.cat((word_emb[3], word_emb[3]), dim=0)), dim=0),
                torch.cat((torch.cat((word_emb[3], word_emb[3]), dim=0), torch.cat((word_emb[0], word_emb[0]), dim=0)), dim=0),
                torch.cat((torch.cat((word_emb[3], word_emb[3]), dim=0), torch.cat((word_emb[1], word_emb[2]), dim=0)), dim=0),
            ]),
            'work_for': torch.tensor([[1.00, float("nan")], [0.70, 0.03], [0.37, 0.63]], device=device),
            'live_in': torch.mul(torch.rand(4, 2, device=device), 0.5), # TODO: add examable values
            'located_in': torch.mul(torch.rand(4, 2, device=device), 0.5), # TODO: add examable values
            'orgbase_on': torch.mul(torch.rand(4, 2, device=device), 0.5), # TODO: add examable values
            'kill': torch.mul(torch.rand(4, 2, device=device), 0.5), # TODO: add examable values
        }, 
        
        # nandL(people,organization)
        #                               John    "works for"     IBM
        'lc0LossTensor' : torch.tensor([0.2000, 0.0000, 0.5100], device=device),
        
        # ifL(work_for, ('x', 'y'), andL(people, ('x',), organization, ('y',)))
        #                                 John           works          for      IBM
        'lc2LossTensor' : torch.tensor([[0.2000,         0.2000,        0.2000,  0.0200],  # John
                                        [float("nan"),  float("nan"),  0.4000,  0.2900],  # works for
                                        [0.5500,         0.2000,        0.1000,  0.0000]], # IBM
                                        device=device)
    
    }
    case = Namespace(case)
    return case


def model_declaration(config, case):
    from regr.program.program import LearningBasedProgram

    from .graph import graph, sentence, word, phrase, pair
    from .graph import Eword, Iword, Bword, Oword
    from .graph import people, organization, location, other, o
    from .graph import work_for, located_in, live_in, orgbase_on, kill
    from .graph import rel_sentence_contains_word, rel_phrase_word1, rel_phrase_word2, rel_pair_phrase1, rel_pair_phrase2, rel_sentence_contains_phrase
    from test_regr.sensor.pytorch.sensors import TestSensor, TestEdgeSensor

    graph.detach()

    sentence['raw'] = TestSensor(expected_outputs=case.sentence.raw)

    # Edge: sentence to word forward
    word[rel_sentence_contains_word.forward, 'raw'] = TestSensor(
        sentence['raw'],
        expected_inputs=(case.sentence.raw,),
        expected_outputs=(case.word.scw, case.word.raw))

    word['emb'] = TestSensor(
        'raw',
        expected_inputs=(case.word.raw,),
        expected_outputs=case.word.emb)

    word[Eword] = TestSensor(
        label=True,
        expected_outputs=case.word.Eword)
    
    word[Iword] = TestSensor(
        label=True,
        expected_outputs=case.word.Iword)
    
    word[Oword] = TestSensor(
        label=True,
        expected_outputs=case.word.Oword)
    
    word[Bword] = TestSensor(
        label=True,
        expected_outputs=case.word.Bword)
    
    word[Eword] = TestSensor(
        'emb',
        expected_inputs=(case.word.emb,),
        expected_outputs=case.word.Eword)

    word[Iword] = TestSensor(
        'emb',
        expected_inputs=(case.word.emb,),
        expected_outputs=case.word.Iword)

    word[Oword] = TestSensor(
        'emb',
        expected_inputs=(case.word.emb,),
        expected_outputs=case.word.Oword)

    word[Bword] = TestSensor(
        'emb',
        expected_inputs=(case.word.emb,),
        expected_outputs=case.word.Bword)

    phrase[rel_phrase_word1.backward, rel_phrase_word2.backward] = TestSensor(
        word['emb'],
        expected_inputs=(case.word.emb,),
        expected_outputs=(case.phrase.pw1_backward, case.phrase.pw2_backward))

    phrase[rel_sentence_contains_phrase.forward, 'raw'] = TestSensor(
        sentence['raw'],
        expected_inputs=(case.sentence.raw,),
        expected_outputs=(case.phrase.scw, case.phrase.raw))
    
    phrase['emb'] = TestSensor(
        rel_phrase_word1.backward('emb'), rel_phrase_word2.backward('emb'),
        expected_inputs=(case.phrase.emb[:, :2048], case.phrase.emb[:, 2048:]),
        expected_outputs=case.phrase.emb)

    pair[rel_pair_phrase1.backward, rel_pair_phrase2.backward] = TestSensor(
        phrase['emb'],
        expected_inputs=(case.phrase.emb,),
        expected_outputs=(case.pair.pa1_backward, case.pair.pa2_backward))

    pair['emb'] = TestSensor(
        rel_pair_phrase1.backward('emb'), rel_pair_phrase2.backward('emb'),
        expected_inputs=(case.pair.emb[:,:4096],case.pair.emb[:,4096:]),
        expected_outputs=case.pair.emb)

    phrase[people] = TestSensor(
        label=True,
        expected_outputs=case.phrase.people)
    phrase[organization] = TestSensor(
        label=True,
        expected_outputs=case.phrase.organization)
    phrase[location] = TestSensor(
        label=True,
        expected_outputs=case.phrase.location)
    phrase[other] = TestSensor(
        label=True,
        expected_outputs=case.phrase.other)
    phrase[o] = TestSensor(
        label=True,
        expected_outputs=case.phrase.O)

    phrase[people] = TestSensor(
        'emb',
        expected_inputs=(case.phrase.emb,),
        expected_outputs=case.phrase.people)
    phrase[organization] = TestSensor(
        'emb',
        expected_inputs=(case.phrase.emb,),
        expected_outputs=case.phrase.organization)
    phrase[location] = TestSensor(
        'emb',
        expected_inputs=(case.phrase.emb,),
        expected_outputs=case.phrase.location)
    phrase[other] = TestSensor(
        'emb',
        expected_inputs=(case.phrase.emb,),
        expected_outputs=case.phrase.other)
    phrase[o] = TestSensor(
        'emb',
        expected_inputs=(case.phrase.emb,),
        expected_outputs=case.phrase.O)


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
        'emb',
        expected_inputs=(case.pair.emb,),
        expected_outputs=case.pair.work_for)
    pair[located_in] = TestSensor(
        'emb',
        expected_inputs=(case.pair.emb,),
        expected_outputs=case.pair.located_in)
    pair[live_in] = TestSensor(
        'emb',
        expected_inputs=(case.pair.emb,),
        expected_outputs=case.pair.live_in)
    pair[orgbase_on] = TestSensor(
        'emb',
        expected_inputs=(case.pair.emb,),
        expected_outputs=case.pair.orgbase_on)
    pair[orgbase_on] = TestSensor(
        'emb',
        expected_inputs=(case.pair.emb,),
        expected_outputs=case.pair.kill)

    lbp = LearningBasedProgram(graph, **config)
    return lbp

def test_graph_naming():
    from .graph import graph, sentence, word, phrase, pair
    from .graph import people, organization, location, other, o
    from .graph import work_for, located_in, live_in, orgbase_on, kill
    from .graph import rel_sentence_contains_word, rel_phrase_word1, rel_phrase_word2, rel_pair_phrase1, rel_pair_phrase2

    # graph
    assert graph.name == 'global'

    # concepts
    assert sentence.name == 'sentence'
    assert word.name == 'word'
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
    #assert rel_phrase_word1.name == 'arg1'
    #assert rel_phrase_word2.name == 'arg2'

    # relation: explicitly named
    # `(rel_pair_word1, rel_pair_word2, ) = pair.has_a(arg1=word, arg2=word)`
    assert rel_pair_phrase1.name == 'arg1'
    assert rel_pair_phrase2.name == 'arg2'


@pytest.mark.gurobi
def test_main_conll04(case):
    from .config import CONFIG
    from .graph import graph, sentence, word, phrase, pair
    from .graph import people, organization, location, other, o
    from .graph import work_for, located_in, live_in, orgbase_on, kill

    lbp = model_declaration(CONFIG.Model, case)
    data = {}

    _, _, datanode = lbp.model(data)

    for child_node in datanode.getChildDataNodes():
        if child_node.ontologyNode.name == 'word':
            assert child_node.getAttribute('raw') == case.word.raw[child_node.instanceID]
            assert len(child_node.findDatanodes(select="phrase")) == 1

        elif child_node.ontologyNode.name == 'phrase':
            assert (child_node.getAttribute('emb') == case.phrase.emb[child_node.instanceID]).all()
            assert (child_node.getAttribute('<people>') == case.phrase.people[child_node.instanceID]).all()
            assert (child_node.getAttribute('<organization>') == case.phrase.organization[child_node.instanceID]).all()
            assert (child_node.getAttribute('<location>') == case.phrase.location[child_node.instanceID]).all()
            assert (child_node.getAttribute('<other>') == case.phrase.other[child_node.instanceID]).all()
            assert (child_node.getAttribute('<O>') == case.phrase.O[child_node.instanceID]).all()
            if child_node.instanceID == 0:
                assert len(child_node.findDatanodes(select="pair")) == 3
            elif child_node.instanceID == 1:
                assert len(child_node.findDatanodes(select="pair")) == 2
            elif child_node.instanceID == 2:
                assert len(child_node.findDatanodes(select="pair")) == 3
                
        else:
            assert False, 'There should be only word and phrases. {} is unexpected.'.format(child_node.ontologyNode.name)

    assert len(datanode.getChildDataNodes(conceptName=word)) == 4 # There are 4 words in sentance
    assert len(datanode.getChildDataNodes(conceptName=phrase)) == 3 # There are 3 phrases build of the words in the sentance

    conceptsRelationsStrings = ['people', 'organization', 'location', 'other', 'O', 'work_for']
    conceptsRelationsConcepts = [people, organization, location, other, o, work_for]
    conceptsRelationsMix = ["people", organization, location, other, o, "work_for"]
    conceptsRelationsEmpty = []
    
    conceptsRelationsVariants = [conceptsRelationsEmpty, conceptsRelationsStrings, conceptsRelationsConcepts, conceptsRelationsMix]
    
    for conceptsRelations in conceptsRelationsVariants:
        
        # ------------ Calculate logical constraints losses 
        lcResult = datanode.calculateLcLoss()
                
        for i in range(3):
            assert round(lcResult['LC0']['lossTensor'][i].item(), 4) == round(case.lc0LossTensor[i].item(), 4)

        for i in product(range(3), repeat = 2):  
            if lcResult['LC2']['lossTensor'][i] != lcResult['LC2']['lossTensor'][i] or case.lc2LossTensor[i] != case.lc2LossTensor[i]:
                if lcResult['LC2']['lossTensor'][i] != lcResult['LC2']['lossTensor'][i] and case.lc2LossTensor[i] != case.lc2LossTensor[i]:
                    assert True
                else:
                    assert False
            else:
                assert round(lcResult['LC2']['lossTensor'][i].item(), 4) == round(case.lc2LossTensor[i].item(), 4)

        # ------------ Call the ILP Solver
        datanode.inferILPConstrains(*conceptsRelations, fun=None)
        
        # ------------ Concepts Results
        
        # Get value of attribute people/ILP for word 0
        #assert tokenResult['people'][0] == 1
        assert datanode.findDatanodes(select = phrase)[0].getAttribute(people, 'ILP').item() == 1

        # Sum value of attribute people/ILP for all words
        #assert sum(tokenResult['people']) == 1
        assert sum([dn.getAttribute(people, 'ILP').item() for dn in datanode.findDatanodes(select = phrase)]) == 1
        
        # Get value of attribute organization/ILP for word 3
        #assert tokenResult['organization'][3] == 1
        assert datanode.findDatanodes(select = phrase)[3].getAttribute(organization, 'ILP').item() == 1
        
        # Sum value of attribute organization/ILP for all words
        #assert sum(tokenResult['organization']) == 1
        assert sum([dn.getAttribute(organization, 'ILP').item() for dn in datanode.findDatanodes(select = phrase)]) == 1
    
        # Sum value of attribute location/ILP for all words
        #assert sum(tokenResult['location']) == 0
        assert sum([dn.getAttribute(location, 'ILP').item() for dn in datanode.findDatanodes(select = phrase)]) == 0
    
        # Sum value of attribute other/ILP for all words
        #assert sum(tokenResult['other']) == 0
        assert sum([dn.getAttribute(other, 'ILP').item() for dn in datanode.findDatanodes(select = phrase)]) == 0
    
        # Get value of attribute o/ILP for word 1
        #assert tokenResult['O'][1] == 1
        assert datanode.findDatanodes(select = phrase)[1].getAttribute(o, 'ILP').item() == 1
        
        JohnDN = datanode.findDatanodes(select = phrase)[1]
        assert JohnDN.getAttribute(organization)[0] == 0.8

        # Get value of attribute o/ILP for word 2
        #assert tokenResult['O'][2] == 1
        assert datanode.findDatanodes(select = phrase)[2].getAttribute(o, 'ILP').item() == 1
    
        # Sum value of attribute o/ILP for all words
        #assert sum(tokenResult['O']) == 2
        assert sum([dn.getAttribute(o, 'ILP').item() for dn in datanode.findDatanodes(select = phrase)]) == 2
        
        # ------------ Relations Results
        
        # Get value of attribute work_for/ILP for pair between 0 and 3
        #assert pairResult['work_for'][0][3] == 1
        # assert datanode.findDatanodes(select = pair, indexes = {"arg1" : 0, "arg2": 3})[0].getAttribute(work_for, 'ILP').item() == 1
        #
        # assert datanode.findDatanodes(select = pair, indexes = {"arg1" : (word, 'raw', 'John'), "arg2": (word, 'raw', "IBM")})[0].getAttribute(work_for, 'ILP') == 1
        #
        # assert datanode.findDatanodes(select = pair, indexes = {"arg1" : ((word,), (word, 'raw', 'John')), "arg2": (word, 'raw', "IBM")})[0].getAttribute(work_for, 'ILP') == 1
        # assert datanode.findDatanodes(select = pair, indexes = {"arg1" : (word, (word, 'raw', 'John')), "arg2": (word, 'raw', "IBM")})[0].getAttribute(work_for, 'ILP') == 1
        #
        # assert datanode.findDatanodes(select = pair, indexes = {"arg1" : (0, (word, 'raw', 'John')), "arg2": (word, 'raw', "IBM")})[0].getAttribute(work_for, 'ILP') == 1
        #
        # # Sum all value of attribute work_for/ILP  for the pair relation from 0
        # #assert sum(pairResult['work_for'][0]) == 1
        # assert sum([dn.getAttribute(work_for, 'ILP').item() for dn in datanode.findDatanodes(select = pair, indexes = {"arg1" : 0})]) == 1
        #
        # # Sum all value of attribute work_for/ILP  for the pair relation from 1
        # #assert sum(pairResult['work_for'][1]) == 0
        # assert sum([dn.getAttribute(work_for, 'ILP').item() for dn in datanode.findDatanodes(select = pair, indexes = {"arg1" : 1})]) == 0
        #
        # # Sum all value of attribute work_for/ILP  for the pair relation from 2
        # #assert sum(pairResult['work_for'][2]) == 0
        # assert sum([dn.getAttribute(work_for, 'ILP').item() for dn in datanode.findDatanodes(select = pair, indexes = {"arg1" : 2})]) == 0
        #
        # # Sum all value of attribute work_for/ILP  for the pair relation from 3
        # #assert sum(pairResult['work_for'][3]) == 0
        # assert sum([dn.getAttribute(work_for, 'ILP').item() for dn in datanode.findDatanodes(select = pair, indexes = {"arg1" : 3})]) == 0


if __name__ == '__main__':
    pytest.main([__file__])
