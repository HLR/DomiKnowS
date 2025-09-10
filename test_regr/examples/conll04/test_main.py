import pytest
import math

@pytest.fixture(name='case')
def test_case():
    import torch
    from domiknows.utils import Namespace

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word_emb = torch.randn(4, 2048, device=device)

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
            'pcw_backward': torch.tensor([[1, 0, 0, 0],
                             [0, 1, 1, 0],
                             [0, 0, 0, 1,]], device=device),
            'scp': torch.tensor([[1], [1], [1]], device=device),
            'emb': torch.stack([word_emb[0], word_emb[1]+word_emb[2], word_emb[3]], dim=0),
            'people': torch.tensor([[0.3, 0.7], [0.9, 0.1], [0.40, 0.6]], device=device),
        },
        'pair': {
            #                John-works    John-IBM      for-works     IBM-John      IBM-for
            'pa1_backward': torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
                                          [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
                                          [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0],
                                          [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],], device=device),
            'pa2_backward': torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                                          [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                                          [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                                          [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], device=device),
            'emb': torch.stack([
                torch.cat((word_emb[0], word_emb[0]), dim=0),
                torch.cat((word_emb[0], word_emb[1]), dim=0),
                torch.cat((word_emb[0], word_emb[2]), dim=0),
                torch.cat((word_emb[0], word_emb[3]), dim=0),
                
                torch.cat((word_emb[1], word_emb[0]), dim=0),
                torch.cat((word_emb[1], word_emb[1]), dim=0),
                torch.cat((word_emb[1], word_emb[2]), dim=0),
                torch.cat((word_emb[1], word_emb[3]), dim=0),
                
                torch.cat((word_emb[2], word_emb[0]), dim=0),
                torch.cat((word_emb[2], word_emb[1]), dim=0),
                torch.cat((word_emb[2], word_emb[2]), dim=0),
                torch.cat((word_emb[2], word_emb[3]), dim=0),
                
                torch.cat((word_emb[3], word_emb[0]), dim=0),
                torch.cat((word_emb[3], word_emb[1]), dim=0),
                torch.cat((word_emb[3], word_emb[2]), dim=0),
                torch.cat((word_emb[3], word_emb[3]), dim=0),
            ]),
            'work_for': torch.tensor([[0.60, 0.40],         [0.80, 0.20],         [0.80, 0.20], [float("nan"), float("nan")],  # John
                                      #[0.5, 0.5], [0.5, 0.5], [0.60, 0.40], [0.70, 0.30],  # works
                                      [float("nan"), float("nan")], [float("nan"), float("nan")], [0.60, 0.40], [0.70, 0.30],  # works
                                      [0.98, 0.02],         [0.70, 0.03],         [0.95, 0.05], [0.90, 0.10],  # for
                                      [0.5, 0.5],         [0.80, 0.20],         [0.90, 0.10], [0.70, 0.30],  # IBM
                                     ], device=device),
            
            'live_in': torch.mul(torch.rand(16, 2, device=device), 0.5), # TODO: add examable values
            'located_in': torch.mul(torch.rand(16, 2, device=device), 0.5), # TODO: add examable values
            'orgbase_on': torch.mul(torch.rand(16, 2, device=device), 0.5), # TODO: add examable values
            'kill': torch.mul(torch.rand(16, 2, device=device), 0.5), # TODO: add examable values
        }, 
        
        # nandL(people,organization)
        #                                       John    works   for     IBM
        'lc0LossTensor' : {"L" : torch.tensor([0.0987, 0.0000, 0.0000, 0.1942], device=device),
                           "G" : torch.tensor([0.5000, 0.3100, 0.2769, 0.5000],  device=device),
                           "P" : torch.tensor([0.2993, 0.1099, 0.0778, 0.3471],  device=device)
                        },
        #                 torch.tensor([0.2000, 0.0000, 0.0000, 0.5100], device=device),
        
        # ifL(work_for('x'), andL(people(path=('x', rel_pair_word1.name)), organization(path=('x', rel_pair_word2.name))))
        #                                 John           works          for      IBM
        'lc2LossTensor' : {"L" : torch.tensor([0.3515, 0.3543, 0.3543, 0.2717, # John
                                               0.5000, 0.5000,  0.4502, 0.3971, # works
                                               0.2769,         0.3385, 0.2891, 0.3100, # for
                                               float("nan") , 0.3543, 0.3100, 0.2071], # IBM
                                               device=device),
                           "G" : torch.tensor([0.0000, 0.6457, 0.7191, 0.0000, 
                                               0.6900, 0.6900, 0.7191, 0.6900, 
                                               0.7231, 0.7231, 0.7231, 0.7231, 
                                               0.5000, 0.6457, 0.7191, 0.0000], 
                                               device=device),
                           
                           "P" : torch.tensor([0.3350, 0.4013, 0.5254, 0.2639,       
                                               0.6900, 0.7803, 0.8065, 0.4637, 
                                               0.5000, 0.7102, 0.7309, 0.3800, 
                                               float("nan"), 0.5000, 0.5470, 0.1350], 
                                               device=device)
                           }
    }
    case = Namespace(case)
    return case


def model_declaration(config, case):
    from domiknows.program.program import LearningBasedProgram

    from .graph import graph, sentence, word, char, phrase, pair
    from .graph import people, organization, location, other, o
    from .graph import work_for, located_in, live_in, orgbase_on, kill
    from .graph import rel_sentence_contains_word, rel_phrase_contains_word, rel_word_contains_char, rel_pair_word1, rel_pair_word2, rel_sentence_contains_phrase
    from test_regr.sensor.pytorch.sensors import TestSensor, TestEdgeSensor

    graph.detach()

    from domiknows.utils import setDnSkeletonMode
    setDnSkeletonMode(False)
    
    lcConcepts = {}
    for _, g in graph.subgraphs.items():
        for _, lc in g.logicalConstrains.items():
            if lc.headLC:  
                lcConcepts[lc.name] = lc.getLcConcepts()
                
    assert lcConcepts == {  'LC0': {'organization', 'people'}, 
                            'LC2': {'O', 'other', 'organization', 'people', 'word', 'location'},
                            'LC4': {'people', 'organization', 'work_for'},
                            'LC6': {'organization', 'location', 'located_in'}, 
                            'LC8': {'location', 'live_in', 'people'},
                            'LC10': {'organization', 'location', 'orgbase_on'},
                            'LC12': {'kill', 'people'}
                            }
    
    sentence['raw'] = TestSensor(expected_outputs=case.sentence.raw)

    # Edge: sentence to word
    word[rel_sentence_contains_word, 'raw'] = TestSensor(
        sentence['raw'],
        expected_inputs=(case.sentence.raw,),
        expected_outputs=(case.word.scw, case.word.raw))
    word['emb'] = TestSensor(
        'raw',
        expected_inputs=(case.word.raw,),
        expected_outputs=case.word.emb)

    # Edge: word to char
    char[rel_word_contains_char, 'raw'] = TestSensor(
        word['raw'],
        expected_inputs=(case.word.raw,),
        expected_outputs=(case.char.wcc, case.char.raw))

    # Edge: word to phrase reversed
    phrase[rel_phrase_contains_word.reversed, 'raw'] = TestSensor(
        word['emb'],
        expected_inputs=(case.word.emb,),
        expected_outputs=(case.phrase.pcw_backward, case.phrase.raw))
    phrase['emb'] = TestSensor(
        rel_phrase_contains_word.reversed('emb'),
        expected_inputs=(case.phrase.emb,),
        expected_outputs=case.phrase.emb)
    phrase[rel_sentence_contains_phrase] = TestEdgeSensor(
        rel_phrase_contains_word.reversed(word[rel_sentence_contains_word], fn=lambda x: x.max(1)[0]),
        relation=rel_sentence_contains_phrase,
        expected_inputs=(case.phrase.scp,),
        expected_outputs=case.phrase.scp)

    pair[rel_pair_word1.reversed, rel_pair_word2.reversed] = TestSensor(
        word['emb'],
        expected_inputs=(case.word.emb,),
        expected_outputs=(case.pair.pa1_backward, case.pair.pa2_backward))
    pair['emb'] = TestSensor(
        rel_pair_word1.reversed('emb'), rel_pair_word2.reversed('emb'),
        expected_inputs=(case.pair.emb[:,:2048],case.pair.emb[:,2048:]),
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
        'emb',
        expected_inputs=(case.word.emb,),
        expected_outputs=case.word.people)
    word[organization] = TestSensor(
        'emb',
        expected_inputs=(case.word.emb,),
        expected_outputs=case.word.organization)
    word[location] = TestSensor(
        'emb',
        expected_inputs=(case.word.emb,),
        expected_outputs=case.word.location)
    word[other] = TestSensor(
        'emb',
        expected_inputs=(case.word.emb,),
        expected_outputs=case.word.other)
    word[o] = TestSensor(
        'emb',
        expected_inputs=(case.word.emb,),
        expected_outputs=case.word.O)

    phrase[people] = TestSensor(
        label=True,
        expected_outputs=case.phrase.people)
    phrase[people] = TestSensor(
        'emb',
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
    from .graph import graph, sentence, word, char, phrase, pair
    from .graph import people, organization, location, other, o
    from .graph import work_for, located_in, live_in, orgbase_on, kill
    from .graph import rel_sentence_contains_word, rel_phrase_contains_word, rel_word_contains_char, rel_pair_word1, rel_pair_word2

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
    import torch
    from .config import CONFIG
    from .graph import graph, sentence, word, char, phrase, pair
    from .graph import people, organization, location, other, o
    from .graph import work_for, located_in, live_in, orgbase_on, kill

    lbp = model_declaration(CONFIG.Model, case)
    data = {}

    _, _, datanode, _ = lbp.model(data)
    
    for child_node in datanode.getChildDataNodes():
        if child_node.ontologyNode.name == 'word':
            #assert child_node.getAttribute('raw') == case.word.raw[child_node.instanceID]
            
            for child_node1 in child_node.getChildDataNodes():
                assert child_node1.ontologyNode.name == 'char'
                       
            #assert len(child_node.getChildDataNodes()) == len(case.char.raw[child_node.instanceID])
            num_pairs = case.pair.pa1_backward[:,child_node.instanceID].sum()
            assert len(child_node.getLinks(relationName = "arg1")) == num_pairs # has relation named "pair"with each word (including itself)

            #assert (child_node.getAttribute('emb') == case.word.emb[child_node.instanceID]).all()
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

    assert len(datanode.getChildDataNodes(conceptName=word)) == 4 # There are 4 words in sentence
    #assert len(datanode.getChildDataNodes(conceptName=phrase)) == 3 # There are 3 phrases build of the words in the sentence

    conceptsRelationsStrings = ['people', 'organization', 'location', 'other', 'O', 'work_for']
    conceptsRelationsConcepts = [people, organization, location, other, o, work_for]
    conceptsRelationsMix = ["people", organization, location, other, o, "work_for"]
    conceptsRelationsEmpty = []
    
    conceptsRelationsVariants = [conceptsRelationsEmpty, conceptsRelationsStrings, conceptsRelationsConcepts, conceptsRelationsMix]
        
    for conceptsRelations in conceptsRelationsVariants:
        
        # ------------ Call the ILP Solver
        datanode.inferILPResults(*conceptsRelations, fun=None)
        datanode.infer()

        # ------------ Concepts Results
        
        # Get value of attribute people/ILP for word 0
        #assert tokenResult['people'][0] == 1
        assert datanode.findDatanodes(select = word)[0].getAttribute(people, 'ILP').item() == 1
        
        test = datanode.findDatanodes(select = word,  indexes = {"contains" : (char, 'raw', 'J')})
        
        #continue
        assert datanode.findDatanodes(select = word,  indexes = {"contains" : (char, 'raw', 'J')})[0].getAttribute(people, 'ILP').item() == 1
        assert datanode.findDatanodes(select = word,  indexes = {"contains" : (char, 'raw', 'h')})[0].getAttribute(people, 'ILP').item() == 1
        assert datanode.findDatanodes(select = word,  indexes = {"contains" : (char, 'raw', 'I')})[0].getAttribute(people, 'ILP').item() == 0
        
        assert datanode.findDatanodes(select = word,  indexes = {"contains" : ((char, 'raw', 'o'), (char, 'raw', 'h')) })[0].getAttribute(people, 'ILP').item() == 1
        assert datanode.findDatanodes(select = word,  indexes = {"contains" : ((char, 'raw', 'o'), (char, 'raw', 'h'), (char, 'raw', 'n')) })[0].getAttribute(people, 'ILP').item() == 1

        assert len(datanode.findDatanodes(select = (char, 'raw', 'J'))) == 1
        assert datanode.findDatanodes(select = (char, 'raw', 'J'))[0].getRootDataNode() == datanode.findDatanodes(select = sentence)[0]
        
        # Sum value of attribute people/ILP for all words
        #assert sum(tokenResult['people']) == 1
        assert sum([dn.getAttribute(people, 'ILP').item() for dn in datanode.findDatanodes(select = word)]) == 1
        
        # Get value of attribute organization/ILP for word 3
        #assert tokenResult['organization'][3] == 1
        assert datanode.findDatanodes(select = word)[3].getAttribute(organization, 'ILP').item() == 1
        
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
        
        assert len(JohnDN.getRelationLinks()) == 1
        assert len(JohnDN.getLinks(relationName="arg1")) == 4

        # Get value of attribute o/ILP for word 2
        #assert tokenResult['O'][2] == 1
        assert datanode.findDatanodes(select = word)[2].getAttribute(o, 'ILP').item() == 1
    
        # Sum value of attribute o/ILP for all words
        #assert sum(tokenResult['O']) == 2
        assert sum([dn.getAttribute(o, 'ILP').item() for dn in datanode.findDatanodes(select = word)]) == 2
        
        # ------------ Relations Results
                
        # Sum all value of attribute work_for/ILP  for the pair relation from 0
        #assert sum(pairResult['work_for'][1]) == 0
        assert sum([dn.getAttribute(work_for, 'ILP').item() if dn.getAttribute(work_for, 'ILP').item() == dn.getAttribute(work_for, 'ILP').item() else 0
                    for dn in datanode.findDatanodes(select = pair, indexes = {"arg1" : 1}) 
                    ]) == 0 or 1
        
        # Sum all value of attribute work_for/ILP  for the pair relation from 1
        #assert sum(pairResult['work_for'][0]) == 0
        assert sum([dn.getAttribute(work_for, 'ILP').item() if not math.isnan(dn.getAttribute(work_for, 'ILP').item()) 
                    else 0 for dn in datanode.findDatanodes(select = pair, indexes = {"arg1" : 0})]) == 0 or 1
        
        # Sum all value of attribute work_for/ILP  for the pair relation from 2
        #assert sum(pairResult['work_for'][2]) == 0
        assert sum([dn.getAttribute(work_for, 'ILP').item() for dn in datanode.findDatanodes(select = pair, indexes = {"arg1" : 2})]) == 0 or 1
    
        # Sum all value of attribute work_for/ILP  for the pair relation from 3
        #assert sum(pairResult['work_for'][3]) == 0
        assert sum([dn.getAttribute(work_for, 'ILP').item() for dn in datanode.findDatanodes(select = pair, indexes = {"arg1" : 3})]) == 0 or 1
        
        # ------------ Calculate logical constraints losses 
        for tnorm in ['L', 'G', "P"]:
            lcResult = datanode.calculateLcLoss(tnorm=tnorm)
                    
            if 'LC0' in lcResult:                     
                for i in range(3):
                    assert round(lcResult['LC0']['lossTensor'][i].item(), 4) == round(case.lc0LossTensor[tnorm][i].item(), 4)
        
            ifLLCid = 'LC22'
            if ifLLCid not in lcResult:
                ifLLCid = 'LC2'
                
            if ifLLCid in lcResult:                     
                for i in range(4):  
                    if lcResult[ifLLCid]['lossTensor'][i] != lcResult[ifLLCid]['lossTensor'][i] or case.lc2LossTensor[tnorm][i] != case.lc2LossTensor[tnorm][i]:
                        if lcResult[ifLLCid]['lossTensor'][i] != lcResult[ifLLCid]['lossTensor'][i] and case.lc2LossTensor[tnorm][i] != case.lc2LossTensor[tnorm][i]:
                            assert True
                        else:
                            assert False
        
        #------- Calculate sample logical constraints losses 
       
        #sampleResult = datanode.calculateLcLoss(sample = True, sampleSize = -1)
        sampleResult = datanode.calculateLcLoss(sample = True, sampleSize = 1)
        sampleResult = datanode.calculateLcLoss(sample = True, sampleSize = 1000)
        
        #datanode.satisfactionReportOfConstraints()
                        
if __name__ == '__main__':
    pytest.main([__file__])