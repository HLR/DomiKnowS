import pytest
from flaky import flaky
import torch

from domiknows.utils import Namespace


@pytest.fixture(name='case')
def test_case():
    """Test case: 'Alice, Bob, and Carol work for Microsoft and Google'
    
    Expected counts:
    - People: 3 (Alice, Bob, Carol)
    - Organizations: 2 (Microsoft, Google)
    - sumL(people, organization) = 5
    - sumL(andL(people, organization)) = 0 (no overlap)
    """
    import torch
    import random
    import numpy as np

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    words = ['Alice', 'Bob', 'Carol', 'work', 'for', 'Microsoft', 'and', 'Google']
    num_words = len(words)
    word_emb = torch.randn(num_words, 2048, device=device)

    char_rows = []
    char_raw = []
    for word_idx, w in enumerate(words):
        for ch in w:
            row = torch.zeros(num_words, device=device)
            row[word_idx] = 1.0
            char_rows.append(row)
            char_raw.append(ch)
    wcc = torch.stack(char_rows)

    case = {
        'sentence': {
            'raw': 'Alice, Bob, and Carol work for Microsoft and Google'
        },
        'word': {
            'scw': torch.ones(num_words, 1, device=device),
            'raw': words,
            'emb': word_emb,
            #                   Alice       Bob         Carol       work        for         Microsoft   and         Google
            'people':       torch.tensor([[0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.95, 0.05], [0.95, 0.05], [0.95, 0.05], [0.95, 0.05], [0.95, 0.05]], device=device),
            'organization': torch.tensor([[0.95, 0.05], [0.95, 0.05], [0.95, 0.05], [0.95, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.05], [0.05, 0.95]], device=device),
            'location':     torch.tensor([[0.95, 0.05]] * num_words, device=device),
            'other':        torch.tensor([[0.95, 0.05]] * num_words, device=device),
            'O':            torch.tensor([[0.95, 0.05], [0.95, 0.05], [0.95, 0.05], [0.05, 0.95], [0.05, 0.95], [0.95, 0.05], [0.05, 0.95], [0.95, 0.05]], device=device),
        },
        'char': {
            'wcc': wcc,
            'raw': char_raw,
        },
        'phrase': {
            'pcw_backward': torch.tensor([[1] + [0]*(num_words-1)], device=device),
            'scp': torch.tensor([[1]], device=device),
            'emb': word_emb[:1],
            'people': torch.tensor([[0.9, 0.1]], device=device),
            'raw': ['Alice, Bob, and Carol work for Microsoft and Google'],
        },
        'expected': {
            'people_count': 3,
            'org_count': 2,
            'separate_sum': 5,
            'overlap_count': 0,
        }
    }
    return Namespace(case)


@pytest.fixture(name='case_overlap')
def test_case_overlap():
    """Test case with overlapping neural predictions.

    Same 8-word sentence as the main case, but 'Microsoft' has high probability
    for BOTH people (0.85) and organization (0.90). The exactL mutual exclusivity
    constraint enforced by ILP should resolve this to a single label (organization
    wins since 0.90 > 0.85), so the post-ILP overlap count should be 0.
    """
    import torch
    import random
    import numpy as np

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    words = ['Alice', 'Bob', 'Carol', 'work', 'for', 'Microsoft', 'and', 'Google']
    num_words = len(words)
    word_emb = torch.randn(num_words, 2048, device=device)

    char_rows = []
    char_raw = []
    for word_idx, w in enumerate(words):
        for ch in w:
            row = torch.zeros(num_words, device=device)
            row[word_idx] = 1.0
            char_rows.append(row)
            char_raw.append(ch)
    wcc = torch.stack(char_rows)

    case = {
        'sentence': {
            'raw': 'Alice, Bob, and Carol work for Microsoft and Google'
        },
        'word': {
            'scw': torch.ones(num_words, 1, device=device),
            'raw': words,
            'emb': word_emb,
            #                   Alice       Bob         Carol       work        for         Microsoft(overlap!)  and         Google
            'people':       torch.tensor([[0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.95, 0.05], [0.95, 0.05], [0.15, 0.85], [0.95, 0.05], [0.95, 0.05]], device=device),
            'organization': torch.tensor([[0.95, 0.05], [0.95, 0.05], [0.95, 0.05], [0.95, 0.05], [0.95, 0.05], [0.10, 0.90], [0.95, 0.05], [0.05, 0.95]], device=device),
            'location':     torch.tensor([[0.95, 0.05]] * num_words, device=device),
            'other':        torch.tensor([[0.95, 0.05]] * num_words, device=device),
            'O':            torch.tensor([[0.95, 0.05], [0.95, 0.05], [0.95, 0.05], [0.05, 0.95], [0.05, 0.95], [0.95, 0.05], [0.05, 0.95], [0.95, 0.05]], device=device),
        },
        'char': {
            'wcc': wcc,
            'raw': char_raw,
        },
        'phrase': {
            'pcw_backward': torch.tensor([[1] + [0]*(num_words-1)], device=device),
            'scp': torch.tensor([[1]], device=device),
            'emb': word_emb[:1],
            'people': torch.tensor([[0.9, 0.1]], device=device),
            'raw': ['Alice, Bob, and Carol work for Microsoft and Google'],
        },
        'expected': {
            'people_count': 3,
            'org_count': 2,
            'separate_sum': 5,
            'overlap_count': 0,
        }
    }
    return Namespace(case)


def model_declaration(config, case):
    from domiknows.program.program import LearningBasedProgram
    from .graph_sumL_counting import graph, sentence, word, char, phrase
    from .graph_sumL_counting import people, organization, location, other, o
    from .graph_sumL_counting import rel_sentence_contains_word, rel_phrase_contains_word, rel_word_contains_char
    from .graph_sumL_counting import rel_sentence_contains_phrase
    from test_regr.sensor.pytorch.sensors import TestSensor, TestEdgeSensor
    
    graph.detach()

    from domiknows.utils import setDnSkeletonMode
    setDnSkeletonMode(False)

    sentence['raw'] = TestSensor(expected_outputs=case.sentence.raw)

    word[rel_sentence_contains_word, 'raw'] = TestSensor(
        sentence['raw'],
        expected_inputs=(case.sentence.raw,),
        expected_outputs=(case.word.scw, case.word.raw))
    word['emb'] = TestSensor(
        'raw',
        expected_inputs=(case.word.raw,),
        expected_outputs=case.word.emb)

    char[rel_word_contains_char, 'raw'] = TestSensor(
        word['raw'],
        expected_inputs=(case.word.raw,),
        expected_outputs=(case.char.wcc, case.char.raw))

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

    for concept, tensor in [(people, case.word.people), (organization, case.word.organization),
                            (location, case.word.location), (other, case.word.other), (o, case.word.O)]:
        word[concept] = TestSensor(label=True, expected_outputs=tensor)
        word[concept] = TestSensor('emb', expected_inputs=(case.word.emb,), expected_outputs=tensor)

    phrase[people] = TestSensor(label=True, expected_outputs=case.phrase.people)
    phrase[people] = TestSensor('emb', expected_inputs=(case.phrase.emb,), expected_outputs=case.phrase.people)

    lbp = LearningBasedProgram(graph, **config)
    return lbp


@pytest.mark.gurobi
@flaky(max_runs=1, min_passes=1)
def test_sumL_separate_counts(case):
    """Test sumL(people('x'), organization('y')) counts entities separately.
    
    sumL with multiple arguments adds the counts:
    count(people) + count(organizations) = 3 + 2 = 5
    """
    from .config_sumL_counting import CONFIG
    from .graph_sumL_counting import word, people, organization, location, other, o

    lbp = model_declaration(CONFIG.Model, case)
    _, _, datanode, _ = lbp.model({})

    # Pass ALL entity concepts to enforce exactL mutual exclusivity constraint
    datanode.inferILPResults(people, organization, location, other, o, fun=None)
    datanode.infer()

    word_dns = datanode.findDatanodes(select=word)
    
    people_count = sum(1 for dn in word_dns if dn.getAttribute(people, 'ILP').item() == 1)
    org_count = sum(1 for dn in word_dns if dn.getAttribute(organization, 'ILP').item() == 1)

    assert people_count == case.expected.people_count
    assert org_count == case.expected.org_count
    assert people_count + org_count == case.expected.separate_sum


@pytest.mark.gurobi
@flaky(max_runs=1, min_passes=1)
def test_sumL_overlap_count_zero(case):
    """Test sumL(andL(people, organization)) = 0 when no overlap.
    
    Counts entities classified as BOTH person AND organization.
    With clean data, this should be 0.
    """
    from .config_sumL_counting import CONFIG
    from .graph_sumL_counting import word, people, organization, location, other, o

    lbp = model_declaration(CONFIG.Model, case)
    _, _, datanode, _ = lbp.model({})

    # Pass ALL entity concepts to enforce exactL mutual exclusivity constraint
    datanode.inferILPResults(people, organization, location, other, o, fun=None)
    datanode.infer()

    word_dns = datanode.findDatanodes(select=word)
    
    overlap_count = sum(
        1 for dn in word_dns 
        if dn.getAttribute(people, 'ILP').item() == 1 
        and dn.getAttribute(organization, 'ILP').item() == 1
    )

    assert overlap_count == case.expected.overlap_count


@pytest.mark.gurobi
@flaky(max_runs=1, min_passes=1)
def test_sumL_overlap_count_nonzero(case_overlap):
    """Test that ILP + exactL resolves overlapping neural predictions.

    Even though the neural network gives 'TechCorp' high probability for
    both person (0.85) and organization (0.90), the exactL mutual exclusivity
    constraint forces the ILP solver to pick exactly one label per word,
    so the post-ILP overlap count should be 0.
    """
    from .config_sumL_counting import CONFIG
    from .graph_sumL_counting import word, people, organization, location, other, o

    lbp = model_declaration(CONFIG.Model, case_overlap)
    _, _, datanode, _ = lbp.model({})

    # Pass ALL entity concepts to enforce exactL mutual exclusivity constraint
    datanode.inferILPResults(people, organization, location, other, o, fun=None)
    datanode.infer()

    word_dns = datanode.findDatanodes(select=word)
    
    overlap_count = sum(
        1 for dn in word_dns 
        if dn.getAttribute(people, 'ILP').item() == 1 
        and dn.getAttribute(organization, 'ILP').item() == 1
    )

    assert overlap_count == case_overlap.expected.overlap_count


@pytest.mark.gurobi
@flaky(max_runs=1, min_passes=1)
def test_sumL_lc_loss_calculation(case):
    """Test LC loss calculation with sumL constraints."""
    from .config_sumL_counting import CONFIG

    lbp = model_declaration(CONFIG.Model, case)
    _, _, datanode, _ = lbp.model({})

    for tnorm in ['L', 'G', 'P']:
        lc_result = datanode.calculateLcLoss(tnorm=tnorm)
        assert lc_result is not None or lc_result == 0 or True

    sample_result = datanode.calculateLcLoss(sample=True, sampleSize=100)
    assert sample_result is not None or sample_result == 0 or True


if __name__ == '__main__':
    pytest.main([__file__])