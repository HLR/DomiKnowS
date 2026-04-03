import pytest
from flaky import flaky
import torch

from domiknows.utils import Namespace

@pytest.fixture(name='case')
def test_case():
    import torch
    import random
    import numpy as np
    from domiknows.utils import Namespace

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # "John works for IBM and Alice works for Google"
    # words: John(0), works(1), for(2), IBM(3), and(4), Alice(5), works(6), for(7), Google(8)
    word_emb = torch.randn(9, 2048, device=device)
    
    
    words = ['John', 'works', 'for', 'IBM', 'and', 'Alice', 'works', 'for', 'Google']
    num_words = len(words)

    char_rows = []
    char_raw = []

    for word_idx, w in enumerate(words):
        for ch in w:
            # one-hot over words: this char belongs to word_idx
            row = torch.zeros(num_words, device=device)
            row[word_idx] = 1.0
            char_rows.append(row)
            char_raw.append(ch)

    wcc = torch.stack(char_rows)  # shape: (37, 9)
    char_raw = [f'c{i}' for i in range(wcc.size(0))]  # 45 dummy char tokens

    case = {
        'sentence': {
            'raw': 'John works for IBM and Alice works for Google'
        },
        'word': {
            'scw': torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]], device=device),
            'raw': words,
            'emb': word_emb,
            #                   John        works       for         IBM         and         Alice       works       for         Google
            'people':       torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.95, 0.05], [0.8, 0.2], [0.95, 0.05], [0.1, 0.9], [0.9, 0.1], [0.95, 0.05], [0.8, 0.2]], device=device),
            'organization': torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.95, 0.05], [0.1, 0.9], [0.95, 0.05], [0.9, 0.1], [0.9, 0.1], [0.95, 0.05], [0.1, 0.9]], device=device),
            'location':     torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.95, 0.05], [0.9, 0.1], [0.95, 0.05], [0.9, 0.1], [0.9, 0.1], [0.95, 0.05], [0.9, 0.1]], device=device),
            'other':        torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.95, 0.05], [0.9, 0.1], [0.95, 0.05], [0.9, 0.1], [0.9, 0.1], [0.95, 0.05], [0.9, 0.1]], device=device),
            'O':            torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.05, 0.95], [0.9, 0.1], [0.05, 0.95], [0.9, 0.1], [0.1, 0.9], [0.05, 0.95], [0.9, 0.1]], device=device),
        },
        'char': {
            'wcc': wcc,
            'wcc_raw': [list(w) for w in ['John', 'works', 'for', 'IBM', 'and', 'Alice', 'works', 'for', 'Google']],
            'raw': char_raw,
        },
       'phrase': {
            'pcw_backward': torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0]], device=device),
            'scp': torch.tensor([[1]], device=device),
            'emb': word_emb[:1],
            'people': torch.tensor([[0.1, 0.9]], device=device),
            'raw': ['John works for IBM and Alice works for Google'],  # length 1
        },
        'pair': {
            'pa1_backward': _build_pair_matrix(9, device, 'arg1'),
            'pa2_backward': _build_pair_matrix(9, device, 'arg2'),
            'emb': _build_pair_emb(word_emb),
            'work_for': _build_work_for_probs(device),
            'live_in': torch.rand(81, 2, device=device) * 0.3,
            'located_in': torch.rand(81, 2, device=device) * 0.3,
            'orgbase_on': torch.rand(81, 2, device=device) * 0.3,
            'kill': torch.rand(81, 2, device=device) * 0.3,
        },
    }
    return Namespace(case)


def _build_pair_matrix(n, device, arg_type):
    """Build pair backward matrix for n words."""
    matrix = torch.zeros(n * n, n, device=device)
    for i in range(n):
        for j in range(n):
            pair_idx = i * n + j
            if arg_type == 'arg1':
                matrix[pair_idx, i] = 1
            else:
                matrix[pair_idx, j] = 1
    return matrix


def _build_pair_emb(word_emb):
    """Build pair embeddings from word embeddings."""
    n = word_emb.shape[0]
    pairs = []
    for i in range(n):
        for j in range(n):
            pairs.append(torch.cat((word_emb[i], word_emb[j]), dim=0))
    return torch.stack(pairs)


def _build_work_for_probs(device):
    """Build work_for probabilities. High for (John,IBM)=idx 3 and (Alice,Google)=idx 53."""
    n = 9
    probs = torch.rand(n * n, 2, device=device) * 0.3
    probs[:, 0] = 1 - probs[:, 1]
    # (John=0, IBM=3) -> idx = 0*9+3 = 3
    probs[3] = torch.tensor([0.1, 0.9], device=device)
    # (Alice=5, Google=8) -> idx = 5*9+8 = 53
    probs[53] = torch.tensor([0.1, 0.9], device=device)
    return probs


def model_declaration(config, case):
    from domiknows.program.program import LearningBasedProgram
    from .graph_counting_relations import graph, sentence, word, char, phrase, pair
    from .graph_counting_relations import people, organization, location, other, o
    from .graph_counting_relations import work_for, located_in, live_in, orgbase_on, kill
    from .graph_counting_relations import rel_sentence_contains_word, rel_phrase_contains_word, rel_word_contains_char
    from .graph_counting_relations import rel_pair_word1, rel_pair_word2, rel_sentence_contains_phrase
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

    pair[rel_pair_word1.reversed, rel_pair_word2.reversed] = TestSensor(
        word['emb'],
        expected_inputs=(case.word.emb,),
        expected_outputs=(case.pair.pa1_backward, case.pair.pa2_backward))
    pair['emb'] = TestSensor(
        rel_pair_word1.reversed('emb'), rel_pair_word2.reversed('emb'),
        expected_inputs=(case.pair.emb[:, :2048], case.pair.emb[:, 2048:]),
        expected_outputs=case.pair.emb)

    for concept, tensor in [(people, case.word.people), (organization, case.word.organization),
                            (location, case.word.location), (other, case.word.other), (o, case.word.O)]:
        word[concept] = TestSensor(label=True, expected_outputs=tensor)
        word[concept] = TestSensor('emb', expected_inputs=(case.word.emb,), expected_outputs=tensor)

    phrase[people] = TestSensor(label=True, expected_outputs=case.phrase.people)
    phrase[people] = TestSensor('emb', expected_inputs=(case.phrase.emb,), expected_outputs=case.phrase.people)

    for rel, tensor in [(work_for, case.pair.work_for), (located_in, case.pair.located_in),
                        (live_in, case.pair.live_in), (orgbase_on, case.pair.orgbase_on), (kill, case.pair.kill)]:
        pair[rel] = TestSensor(label=True, expected_outputs=tensor)
        pair[rel] = TestSensor('emb', expected_inputs=(case.pair.emb,), expected_outputs=tensor)

    lbp = LearningBasedProgram(graph, **config)
    return lbp


@pytest.mark.gurobi
@flaky(max_runs=3, min_passes=1)
def test_counting_constraints(case):
    from .config_counting_relations import CONFIG
    from .graph_counting_relations import word, pair, people, organization, work_for

    lbp = model_declaration(CONFIG.Model, case)
    _, _, datanode, _ = lbp.model({})

    # ILP inference
    datanode.inferILPResults(people, organization, work_for, fun=None)
    datanode.infer()

    # Verify John and Alice are people
    word_dns = datanode.findDatanodes(select=word)
    assert word_dns[0].getAttribute(people, 'ILP').item() == 1  # John
    assert word_dns[5].getAttribute(people, 'ILP').item() == 1  # Alice

    # Verify IBM and Google are organizations
    assert word_dns[3].getAttribute(organization, 'ILP').item() == 1  # IBM
    assert word_dns[8].getAttribute(organization, 'ILP').item() == 1  # Google

    # Verify work_for relations
    pair_dns = datanode.findDatanodes(select=pair)
    work_for_count = sum(1 for dn in pair_dns if dn.getAttribute(work_for, 'ILP').item() == 1)
    assert 2 <= work_for_count <= 3  # counting constraints: atLeast 2, atMost 3

    # Calculate LC loss
    for tnorm in ['L', 'G', 'P']:
        lc_result = datanode.calculateLcLoss(tnorm=tnorm)
        assert lc_result is not None

    # Sample loss
    sample_result = datanode.calculateLcLoss(sample=True, sampleSize=100)
    assert sample_result is not None


if __name__ == '__main__':
    pytest.main([__file__])