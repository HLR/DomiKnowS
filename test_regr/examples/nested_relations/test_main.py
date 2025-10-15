import sys
from itertools import product

from test_regr.sensor.pytorch.test_candidate_sensor import case
sys.path.append('.')
sys.path.append('../../..')

import pytest

@pytest.fixture(name='case')
def test_case():
    import torch
    from domiknows.utils import Namespace

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word_emb = torch.randn(4, 2048, device=device)
    
    # Compute phrase embeddings first
    phrase_emb = torch.stack([word_emb[0], word_emb[1]+word_emb[2], word_emb[3]], dim=0)
    
    num_phrases = 3
    pair_indices = [(i, j) for i in range(num_phrases) for j in range(num_phrases)]

    # arg1 / arg2 backward mappings: shape (9,3)
    pa1_backward = torch.zeros(len(pair_indices), num_phrases, device=device)
    pa2_backward = torch.zeros(len(pair_indices), num_phrases, device=device)
    for r, (i, j) in enumerate(pair_indices):
        pa1_backward[r, i] = 1.0
        pa2_backward[r, j] = 1.0

    # Pair embeddings are concatenations of phrase embeddings: shape (9, 4096)
    pair_emb_rows = []
    for (i, j) in pair_indices:
        pair_emb_rows.append(torch.cat((phrase_emb[i], phrase_emb[j]), dim=0))
    pair_emb = torch.stack(pair_emb_rows, dim=0)
    
    # Labels / scores for pair relations: shape (9, 2)
    # Make (John -> IBM) the positive 'work_for' example: that's phrase 0 -> phrase 2 in your fixture
    work_for = torch.tensor([
        [0.50, 0.50],  # (0,0)
        [0.70, 0.30],  # (0,1)
        [0.90, 0.10],  # (0,2)  John -> IBM  (positive)
        [0.40, 0.60],  # (1,0)
        [0.50, 0.50],  # (1,1)
        [0.40, 0.60],  # (1,2)
        [0.60, 0.40],  # (2,0)
        [0.60, 0.40],  # (2,1)
        [0.50, 0.50],  # (2,2)
    ], device=device)

    # You can keep the other relations as low-confidence dummies (or random * 0.5) with the same (9,2) shape:
    live_in    = torch.mul(torch.rand(9, 2, device=device), 0.5)
    located_in = torch.mul(torch.rand(9, 2, device=device), 0.5)
    orgbase_on = torch.mul(torch.rand(9, 2, device=device), 0.5)
    kill       = torch.mul(torch.rand(9, 2, device=device), 0.5)

    case = {
        'sentence': {
            'raw': 'John works for IBM'
        },
        'word': {
            'scw': torch.tensor([[1], [1], [1], [1]], device=device),
            'raw': ['John', 'works', 'for', 'IBM'],
            'emb': word_emb,
            #                             John        works       for           IBM
            'people':       torch.tensor([[0.3, 0.7], [0.9, 0.1], [0.98, 0.02], [0.50, 0.5]], device=device),
            'organization': torch.tensor([[0.5, 0.5], [0.8, 0.2], [0.97, 0.03], [0.09, 0.91]], device=device),
            'location':     torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.95, 0.05], [0.50, 0.50]], device=device),
            'other':        torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.90, 0.10], [0.70, 0.30]], device=device),
            'O':            torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.10, 0.90], [0.90, 0.10]], device=device),
            # Add missing attributes
            'Eword':        torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]], device=device),
            'Iword':        torch.tensor([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5]], device=device),
            'Bword':        torch.tensor([[0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4]], device=device),
            'Oword':        torch.tensor([[0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3]], device=device),
        },
        'char': {
            'wcc': torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], 
                    [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
                    [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0],
                    [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]], device=device),
            'wcc_raw': [['J', 'o', 'h', 'n',], ['w', 'o', 'r', 'k', 's',], ['f', 'o', 'r',], ['I', 'B', 'M']],
            'raw': ['J', 'o', 'h', 'n', 'w', 'o', 'r', 'k', 's', 'f', 'o', 'r', 'I', 'B', 'M']
        },
        'phrase': {
            # ['John', 'works for', 'IBM'],
            # CHANGE: pcw_backward should be (3, 4) not (3, 3)
            'pcw_backward': torch.tensor([[1, 0, 0, 0],
                                        [0, 1, 1, 0],
                                        [0, 0, 0, 1]], device=device),
            'scp': torch.tensor([[1], [1], [1]], device=device),
            'emb': torch.stack([word_emb[0], word_emb[1]+word_emb[2], word_emb[3]], dim=0),
            'people': torch.tensor([[0.3, 0.7], [0.9, 0.1], [0.40, 0.6]], device=device),
            'organization': torch.tensor([[0.5, 0.5], [0.8, 0.2], [0.09, 0.91]], device=device),
            'location': torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.50, 0.50]], device=device),
            'other': torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.70, 0.30]], device=device),
            'O': torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.90, 0.10]], device=device),
            # CHANGE: pw1_backward and pw2_backward should be (3, 4) not (3, 3)
            # Use dtype=torch.float32 to match word_emb dtype
            'pw1_backward': torch.tensor([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 0, 1]], device=device, dtype=torch.float32),
            'pw2_backward': torch.tensor([[1, 0, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], device=device, dtype=torch.float32),
            'raw': ['John', 'works for', 'IBM'],
        },
        'pair': {
            'pa1_backward': pa1_backward,   # (9,3)
            'pa2_backward': pa2_backward,   # (9,3)
            'emb': pair_emb,                # (9,4096)
            'work_for': work_for,           # (9,2)
            'live_in': live_in,             # (9,2)
            'located_in': located_in,       # (9,2)
            'orgbase_on': orgbase_on,       # (9,2)
            'kill': kill,                   # (9,2)
        }, 
        
        # nandL(people,organization)
        #                                       John    works   for     IBM # 0.5987, 0.3100, 0.5498
        'lc0LossTensor' : torch.tensor([0.5987, 0.3100, 0.5498], device=device),
        
        #nandL(people,organization)                   John    works for     IBM
        'lc10LossTensor' : torch.tensor([0.4013, 0.5498, 0.5000], device=device),
    }
    case = Namespace(case)
    return case


def model_declaration(config, case):
    from domiknows.program.program import LearningBasedProgram

    from .graph import graph, sentence, word, phrase, pair
    from .graph import Eword, Iword, Bword, Oword
    from .graph import people, organization, location, other, o
    from .graph import work_for, located_in, live_in, orgbase_on, kill
    from .graph import rel_sentence_contains_word, rel_phrase_word1, rel_phrase_word2, rel_pair_phrase1, rel_pair_phrase2, rel_sentence_contains_phrase
    from test_regr.sensor.pytorch.sensors import TestSensor, TestEdgeSensor

    graph.detach()

    sentence['raw'] = TestSensor(expected_outputs=case.sentence.raw)

    # Edge: sentence to word
    word[rel_sentence_contains_word, 'raw'] = TestSensor(
        sentence['raw'],
        expected_inputs=(case.sentence.raw,),
        expected_outputs=(case.word.scw, case.word.raw))
    
    # Edge: sentence to phrase
    phrase[rel_sentence_contains_phrase, 'raw'] = TestSensor(
        sentence['raw'],
        expected_inputs=(case.sentence.raw,),
        expected_outputs=(case.phrase.scp, case.phrase.raw))

    word['emb'] = TestSensor(
        'raw',
        expected_inputs=(case.word.raw,),
        expected_outputs=case.word.emb)

    # Remove duplicate assignments - only keep one set
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

    phrase[rel_phrase_word1.reversed, rel_phrase_word2.reversed] = TestSensor(
        word['emb'],
        expected_inputs=(case.word.emb,),
        expected_outputs=(case.phrase.pw1_backward, case.phrase.pw2_backward))

    phrase_emb_input1 = case.phrase.pw1_backward @ case.word.emb  # (3, 4) @ (4, 2048) = (3, 2048)
    phrase_emb_input2 = case.phrase.pw2_backward @ case.word.emb  # (3, 4) @ (4, 2048) = (3, 2048)

    phrase['emb'] = TestSensor(
        rel_phrase_word1.reversed('emb'), rel_phrase_word2.reversed('emb'),
        expected_inputs=(phrase_emb_input1, phrase_emb_input2),
        expected_outputs=case.phrase.emb)

    pair[rel_pair_phrase1.reversed, rel_pair_phrase2.reversed] = TestSensor(
        phrase['emb'],
        expected_inputs=(case.phrase.emb,),
        expected_outputs=(case.pair.pa1_backward, case.pair.pa2_backward))

    pair['emb'] = TestSensor(
        rel_pair_phrase1.reversed('emb'), rel_pair_phrase2.reversed('emb'),
        expected_inputs=(case.pair.pa1_backward @ case.phrase.emb, case.pair.pa2_backward @ case.phrase.emb),
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
    pair[kill] = TestSensor(
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

    loss, metric, datanode, builder = lbp.model(data)

    for child_node in datanode.getChildDataNodes():
        if child_node.ontologyNode.name == 'word':
            assert child_node.getAttribute('raw') == case.word.raw[child_node.instanceID]
            #assert len(child_node.findDatanodes(select="phrase")) == 1

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
    assert len(datanode.findDatanodes(select=phrase)) == 3 # There are 3 phrases build of the words in the sentance

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

        for i in range(3):  
            if lcResult['LC10']['lossTensor'][i] != lcResult['LC10']['lossTensor'][i] or case.lc10LossTensor[i] != case.lc10LossTensor[i]:
                if lcResult['LC10']['lossTensor'][i] != lcResult['LC10']['lossTensor'][i] and case.lc10LossTensor[i] != case.lc10LossTensor[i]:
                    assert True
                else:
                    assert False
            else:
                assert round(lcResult['LC10']['lossTensor'][i].item(), 4) == round(case.lc10LossTensor[i].item(), 4)

if __name__ == '__main__':
    pytest.main([__file__])
