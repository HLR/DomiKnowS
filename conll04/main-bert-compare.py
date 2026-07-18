"""conll04 BERT primal-dual — training-mechanism comparison.

Mirrors ``main-bert-primaldual.py`` but, instead of a single training run,
drives the reusable comparison harness
(``domiknows.program.training_comparison.TrainingComparison``) over several
constraint-training mechanisms and prints a side-by-side table:

    baseline       ascent duals + interpreter LC loss  (pre-R1/R5)
    r1_compiled    R1: compiled (batched-gather) LC loss
    r5a_augmented  R5A: augmented-Lagrangian duals
    r1_r5a         R1 + R5A combined

For each variant it reports total train time, the wall-clock spent building the
constraint loss (R1's target), the mean constraint violation before/after
training (R5's target), and macro-F1 over entities+relations (task quality).

Run (needs the same environment/data as main-bert-primaldual.py — BERT,
en_core_web_sm, and the split corp files under data/):

    python main-bert-compare.py --split 2 --iteration 5 --gpu cpu

This adds one more ``Variant`` per future R mechanism; nothing else changes.
"""

import sys
import itertools
import torch

sys.path.append('.')
sys.path.append('../..')

from domiknows.program.model.pytorch import SolverModelDictLoss
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.loss import NBCrossEntropyDictLoss
from domiknows.program.training_comparison import TrainingComparison, Variant, DEFAULT_VARIANTS
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ModuleSensor, ReaderSensor, FunctionalReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor, CompositionCandidateReaderSensor
from domiknows.utils import setProductionLogMode
from examples.conll04.CallBackModel import CallbackPrimalProgram

from conll.data.data import SingletonDataLoader

import spacy
nlp = spacy.load('en_core_web_sm')

from transformers import BertTokenizerFast, BertModel

TRANSFORMER_MODEL = 'bert-base-uncased'
FEATURE_DIM = 768 + 96


class Tokenizer():
    def __init__(self) -> None:
        self.tokenizer = BertTokenizerFast.from_pretrained(TRANSFORMER_MODEL)

    def __call__(self, text):
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(text, padding=True, return_tensors='pt', return_offsets_mapping=True)
        ids = tokens['input_ids']
        mask = tokens['attention_mask']
        offset = tokens['offset_mapping']
        idx = mask.nonzero()[:, 0].unsqueeze(-1)
        mapping = torch.zeros(idx.shape[0], idx.max() + 1)
        mapping.scatter_(1, idx, 1)
        mask = mask.bool()
        ids = ids.masked_select(mask)
        offset = torch.stack((offset[:, :, 0].masked_select(mask), offset[:, :, 1].masked_select(mask)), dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return mapping, ids, offset, tokens


class BERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = BertModel.from_pretrained(TRANSFORMER_MODEL)
        for param in self.module.base_model.parameters():
            param.requires_grad = False

    def forward(self, input):
        input = input.unsqueeze(0)
        _out = self.module(input)
        out, *_ = _out
        if isinstance(out, str):
            out = _out.last_hidden_state
        assert out.shape[0] == 1
        return out.squeeze(0)


class Classifier(torch.nn.Sequential):
    def __init__(self, in_features) -> None:
        super().__init__(torch.nn.Linear(in_features, 2))


def build_program(variant, device='cpu'):
    """Re-declare the conll04 graph + sensors and build a CallbackPrimalProgram
    whose construction merges ``variant.program_kwargs`` (compile_lc /
    dual_algorithm / …). Faithful to ``main-bert-primaldual.py``'s ``model()``.
    """
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o
    from graph import work_for, located_in, live_in, orgbase_on, kill
    from graph import (rel_sentence_contains_word, rel_phrase_contains_word,
                       rel_pair_phrase1, rel_pair_phrase2, rel_sentence_contains_phrase)

    graph.detach()

    phrase['text'] = ReaderSensor(keyword='tokens')
    phrase['postag'] = ReaderSensor(keyword='postag')

    def word2vec(text):
        texts = list(map(lambda x: ' '.join(x.split('/')), text))
        tokens_list = list(nlp.pipe(texts))
        return torch.tensor([tokens.vector for tokens in tokens_list])
    phrase['w2v'] = FunctionalSensor('text', forward=word2vec)

    def merge_phrase(phrase_text):
        return [' '.join(phrase_text)], torch.ones((1, len(phrase_text)))
    sentence['text', rel_sentence_contains_phrase.reversed] = JointSensor(phrase['text'], forward=merge_phrase)

    word[rel_sentence_contains_word, 'ids', 'offset', 'text'] = JointSensor(sentence['text'], forward=Tokenizer())
    word['bert'] = ModuleSensor('ids', module=BERT())

    def match_phrase(phrase_, word_offset):
        def overlap(a_s, a_e, b_s, b_e):
            return (a_s <= b_s and b_s <= a_e) or (a_s <= b_e and b_e <= a_e)
        ph_offset = 0
        ph_word_overlap = []
        for ph in phrase_:
            ph_len = len(ph)
            word_overlap = []
            for word_s, word_e in word_offset:
                if word_e - word_s <= 0:
                    word_overlap.append(False)
                else:
                    word_overlap.append(overlap(ph_offset, ph_offset + ph_len, word_s, word_e))
            ph_word_overlap.append(word_overlap)
            ph_offset += ph_len + 1
        return torch.tensor(ph_word_overlap)
    phrase[rel_phrase_contains_word.reversed] = EdgeSensor(
        phrase['text'], word['offset'], relation=rel_phrase_contains_word.reversed, forward=match_phrase)
    phrase['bert'] = FunctionalSensor(rel_phrase_contains_word.reversed(word['bert']), forward=lambda bert: bert)
    phrase['emb'] = FunctionalSensor('bert', 'w2v', forward=lambda bert, w2v: torch.cat((bert, w2v), dim=-1))

    phrase[people] = ModuleLearner('emb', module=Classifier(FEATURE_DIM))
    phrase[organization] = ModuleLearner('emb', module=Classifier(FEATURE_DIM))
    phrase[location] = ModuleLearner('emb', module=Classifier(FEATURE_DIM))
    phrase[other] = ModuleLearner('emb', module=Classifier(FEATURE_DIM))
    phrase[o] = ModuleLearner('emb', module=Classifier(FEATURE_DIM))

    def find_label(label_type):
        def find(data):
            return torch.tensor([item == label_type for item in data])
        return find
    phrase[people] = FunctionalReaderSensor(keyword='label', forward=find_label('Peop'), label=True)
    phrase[organization] = FunctionalReaderSensor(keyword='label', forward=find_label('Org'), label=True)
    phrase[location] = FunctionalReaderSensor(keyword='label', forward=find_label('Loc'), label=True)
    phrase[other] = FunctionalReaderSensor(keyword='label', forward=find_label('Other'), label=True)
    phrase[o] = FunctionalReaderSensor(keyword='label', forward=find_label('O'), label=True)

    def filter_pairs(phrase_text, arg1, arg2, data):
        for rel, (rel_arg1, *_), (rel_arg2, *_) in data:
            if arg1.instanceID == rel_arg1 and arg2.instanceID == rel_arg2:
                return True
        return False
    pair[rel_pair_phrase1.reversed, rel_pair_phrase2.reversed] = CompositionCandidateReaderSensor(
        phrase['text'], relations=(rel_pair_phrase1.reversed, rel_pair_phrase2.reversed),
        keyword='relation', forward=filter_pairs)
    pair['emb'] = FunctionalSensor(
        rel_pair_phrase1.reversed('emb'), rel_pair_phrase2.reversed('emb'),
        forward=lambda arg1, arg2: torch.cat((arg1, arg2), dim=-1))

    pair[work_for] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    pair[located_in] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    pair[live_in] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    pair[orgbase_on] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    pair[kill] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))

    def find_relation(relation_type):
        def find(arg1m, arg2m, data):
            label = torch.zeros(arg1m.shape[0], dtype=torch.bool)
            for rel, (arg1, *_), (arg2, *_) in data:
                if rel == relation_type:
                    i, = (arg1m[:, arg1] * arg2m[:, arg2]).nonzero(as_tuple=True)
                    label[i] = True
            return label
        return find
    pair[work_for] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Work_For'), label=True)
    pair[located_in] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Located_In'), label=True)
    pair[live_in] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Live_In'), label=True)
    pair[orgbase_on] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('OrgBased_In'), label=True)
    pair[kill] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Kill'), label=True)

    program = CallbackPrimalProgram(
        graph, Model=SolverModelDictLoss, poi=(sentence, phrase, pair), inferTypes=['local/argmax'],
        dictloss={
            str(o.name): NBCrossEntropyDictLoss(weight=torch.tensor([4.5341, 0.5620]).to(device)),
            str(location.name): NBCrossEntropyDictLoss(weight=torch.tensor([0.5194, 13.3925]).to(device)),
            str(people.name): NBCrossEntropyDictLoss(weight=torch.tensor([0.5156, 22.5134]).to(device)),
            str(other.name): NBCrossEntropyDictLoss(weight=torch.tensor([0.5120, 21.4100]).to(device)),
            str(organization.name): NBCrossEntropyDictLoss(weight=torch.tensor([0.5098, 25.8953]).to(device)),
            str(work_for.name): NBCrossEntropyDictLoss(weight=torch.tensor([0.6277, 2.4578]).to(device)),
            str(located_in.name): NBCrossEntropyDictLoss(weight=torch.tensor([0.6270, 2.4677]).to(device)),
            str(live_in.name): NBCrossEntropyDictLoss(weight=torch.tensor([0.6748, 2.1306]).to(device)),
            str(orgbase_on.name): NBCrossEntropyDictLoss(weight=torch.tensor([0.6309, 2.4094]).to(device)),
            str(kill.name): NBCrossEntropyDictLoss(weight=torch.tensor([0.5730, 4.3231]).to(device)),
            "default": NBCrossEntropyDictLoss()},
        tnorm='G',
        metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
        **variant.program_kwargs)

    return program


def compute_scores(item, criteria="F1"):
    entities = ["location", "people", "organization", "other"]
    relations = ["work_for", "located_in", "live_in", "orgbase_on", "kill"]
    n = 0.0
    s = 0.0
    for key in entities + relations:
        if key in item:
            n += 1
            s += item[key][criteria]
    return s / n if n else float('nan')


def make_evaluate(test_reader, device):
    def evaluate(program):
        program.test(test_reader, device=device)
        metrics = program.model.metric['argmax'].value()
        return {'macro_F1': compute_scores(metrics, criteria="F1")}
    return evaluate


def main(args):
    device = args.gpu
    split_id = args.split
    train_reader = SingletonDataLoader(f'data/conll04.corp_{split_id}_train.corp')
    test_reader = SingletonDataLoader(f'data/conll04.corp_{split_id}_test.corp')
    valid_reader = SingletonDataLoader(f'data/conll04.corp_{split_id}_valid.corp')

    print('\n' + '#' * 72)
    print(f'# conll04 BERT primal-dual — R-mechanism training comparison')
    print(f'# split={split_id}  epochs/variant={args.iteration}  seed={args.seed}  device={device}')
    print('#' * 72)

    # Measure constraint violation on a capped subset — a full BERT pass over
    # the training set per variant (x before/after) would dominate runtime.
    violation_subset = list(itertools.islice(iter(train_reader), args.violation_items))

    comparison = TrainingComparison(
        build_program=lambda v: build_program(v, device=device),
        dataset=train_reader,
        evaluate=make_evaluate(test_reader, device),
        variants=DEFAULT_VARIANTS,
        epochs=args.iteration,
        seed=args.seed,
        device=device,
        violation_tnorm='G',  # match the program's tnorm for a comparable metric
        violation_dataset=violation_subset,
        optim=lambda param: torch.optim.SGD(param, lr=.001),
        train_kwargs=dict(valid_set=valid_reader, test_set=test_reader, c_warmup_iters=10),
        verbose=True,
        print_table=True,
    )
    result = comparison.run()

    # Explicit final summary to the screen (survives setProductionLogMode()).
    print('\n' + '#' * 72)
    print('# FINAL COMPARISON (macro_F1 higher is better; viol_after lower is better)')
    print('#' * 72)
    print(result.render())
    print('#' * 72 + '\n')
    return result


import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="conll04 R-mechanism training comparison")
    parser.add_argument("-s", "--split", type=int, default=2, choices=[1, 2, 3, 4, 5])
    parser.add_argument("-i", "--iteration", type=int, default=5, help="epochs per variant")
    parser.add_argument("--violation-items", type=int, default=30,
                        help="how many train items to measure constraint violation on")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-g", "--gpu", type=str, default="cpu",
                        choices=["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    setProductionLogMode()
    main(args)
