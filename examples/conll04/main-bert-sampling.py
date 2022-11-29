import imp
import sys
import torch

sys.path.append('.')
sys.path.append('../..')

from regr.program import POIProgram, SolverPOIProgram, IMLProgram, CallbackProgram
from regr.program.callbackprogram import ProgramStorageCallback
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from regr.program.lossprogram import PrimalDualProgram
# from regr.program.primaldualprogram import PrimalDualProgram
from regr.program.model.pytorch import SolverModel, SolverModelDictLoss
from regr.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss, NBCrossEntropyDictLoss
from regr.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ModuleSensor, ReaderSensor, FunctionalReaderSensor, cache, TorchCache
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor, EdgeSensor, CompositionCandidateReaderSensor
from regr.sensor.pytorch.query_sensor import DataNodeReaderSensor
from regr.utils import setProductionLogMode
from examples.conll04.CallBackModel import CallbackPrimalProgram, CallbackSamplingProgram

from conll.data.data import SingletonDataLoader



import spacy
# from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm') #English()


import logging

logging.basicConfig(level=logging.INFO)


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
        mapping = torch.zeros(idx.shape[0], idx.max()+1)
        mapping.scatter_(1, idx, 1)

        mask = mask.bool()
        ids = ids.masked_select(mask)
        offset = torch.stack((offset[:,:,0].masked_select(mask), offset[:,:,1].masked_select(mask)), dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return mapping, ids, offset, tokens

class BERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = BertModel.from_pretrained(TRANSFORMER_MODEL)
        # to freeze BERT, uncomment the following
        for param in self.module.base_model.parameters():
            param.requires_grad = False

    def forward(self, input):
        input = input.unsqueeze(0)
        _out = self.module(input)
        
        out, *_ = _out
        
        if (isinstance(out, str)): # Update for new transformers
            out = _out.last_hidden_state
            
        assert out.shape[0] == 1
        out = out.squeeze(0)
        return out

class Classifier(torch.nn.Sequential):
    def __init__(self, in_features) -> None:
        linear = torch.nn.Linear(in_features, 2)
        super().__init__(linear)


def model(device='auto'):
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o
    from graph import work_for, located_in, live_in, orgbase_on, kill
    from graph import rel_sentence_contains_word, rel_phrase_contains_word, rel_pair_phrase1, rel_pair_phrase2, rel_sentence_contains_phrase

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

    def match_phrase(phrase, word_offset):
        def overlap(a_s, a_e, b_s, b_e):
            return (a_s <= b_s and b_s <= a_e) or (a_s <= b_e and b_e <= a_e)
        ph_offset = 0
        ph_word_overlap = []
        for ph in phrase:
            ph_len = len(ph)
            word_overlap = []
            for word_s, word_e in word_offset:
                if word_e - word_s <= 0:
                    # empty string / special tokens
                    word_overlap.append(False)
                else:
                    # other tokens, do compare offset
                    word_overlap.append(overlap(ph_offset, ph_offset+ph_len, word_s, word_e))
            ph_word_overlap.append(word_overlap)
            ph_offset += ph_len + 1
        return torch.tensor(ph_word_overlap)
    phrase[rel_phrase_contains_word.reversed] = EdgeSensor(phrase['text'], word['offset'], relation=rel_phrase_contains_word.reversed, forward=match_phrase)
    def phrase_bert(bert):
        return bert
    phrase['bert'] = FunctionalSensor(rel_phrase_contains_word.reversed(word['bert']), forward=phrase_bert)
    phrase['emb'] = FunctionalSensor('bert', 'w2v', forward=lambda bert, w2v: torch.cat((bert, w2v), dim=-1))

    phrase[people] = ModuleLearner('emb', module=Classifier(FEATURE_DIM))
    phrase[organization] = ModuleLearner('emb', module=Classifier(FEATURE_DIM))
    phrase[location] = ModuleLearner('emb', module=Classifier(FEATURE_DIM))
    phrase[other] = ModuleLearner('emb', module=Classifier(FEATURE_DIM))
    phrase[o] = ModuleLearner('emb', module=Classifier(FEATURE_DIM))

    def find_label(label_type):
        def find(data):
            label = torch.tensor([item==label_type for item in data])
            return label # torch.stack((~label, label), dim=1)
        return find
    phrase[people] = FunctionalReaderSensor(keyword='label', forward=find_label('Peop'), label=True)
    phrase[organization] = FunctionalReaderSensor(keyword='label', forward=find_label('Org'), label=True)
    phrase[location] = FunctionalReaderSensor(keyword='label', forward=find_label('Loc'), label=True)
    phrase[other] = FunctionalReaderSensor(keyword='label', forward=find_label('Other'), label=True)
    phrase[o] = FunctionalReaderSensor(keyword='label', forward=find_label('O'), label=True)

    # pair[rel_pair_phrase1.reversed, rel_pair_phrase2.reversed] = CompositionCandidateSensor(
    #     phrase['text'],
    #     relations=(rel_pair_phrase1.reversed, rel_pair_phrase2.reversed),
    #     forward=lambda *_, **__: True)
    def filter_pairs(phrase_text, arg1, arg2, data):
        for rel, (rel_arg1, *_), (rel_arg2, *_) in data:
            if arg1.instanceID == rel_arg1 and arg2.instanceID == rel_arg2:
                return True
        return False
    pair[rel_pair_phrase1.reversed, rel_pair_phrase2.reversed] = CompositionCandidateReaderSensor(
        phrase['text'],
        relations=(rel_pair_phrase1.reversed, rel_pair_phrase2.reversed),
        keyword='relation',
        forward=filter_pairs)
    pair['emb'] = FunctionalSensor(
        rel_pair_phrase1.reversed('emb'), rel_pair_phrase2.reversed('emb'),
        forward=lambda arg1, arg2: torch.cat((arg1, arg2), dim=-1))

    pair[work_for] = ModuleLearner('emb', module=Classifier(FEATURE_DIM*2))
    pair[located_in] = ModuleLearner('emb', module=Classifier(FEATURE_DIM*2))
    pair[live_in] = ModuleLearner('emb', module=Classifier(FEATURE_DIM*2))
    pair[orgbase_on] = ModuleLearner('emb', module=Classifier(FEATURE_DIM*2))
    pair[kill] = ModuleLearner('emb', module=Classifier(FEATURE_DIM*2))

    def find_relation(relation_type):
        def find(arg1m, arg2m, data):
            label = torch.zeros(arg1m.shape[0], dtype=torch.bool)
            for rel, (arg1,*_), (arg2,*_) in data:
                if rel == relation_type:
                    i, = (arg1m[:, arg1] * arg2m[:, arg2]).nonzero(as_tuple=True)
                    label[i] = True
            return label # torch.stack((~label, label), dim=1)
        return find
    pair[work_for] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Work_For'), label=True)
    pair[located_in] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Located_In'), label=True)
    pair[live_in] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Live_In'), label=True)
    pair[orgbase_on] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('OrgBased_In'), label=True)
    pair[kill] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Kill'), label=True)

    
#     class DictCallBackProgram(CallbackProgram, PrimalDualProgram):
#         pass
    
    lbp1 = CallbackSamplingProgram(
        graph, Model=SolverModel, poi=(sentence, phrase, pair), inferTypes=['local/argmax', "ILP"],
        loss = MacroAverageTracker(NBCrossEntropyLoss()),
        sample = True,
        sampleSize=500, 
        sampleGlobalLoss = False,
        beta=1,
        metric={
            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax')),
            'ILP': PRF1Tracker(DatanodeCMMetric('ILP'))})
    
    lbp = CallbackSamplingProgram(
        graph, Model=SolverModel, poi=(sentence, phrase, pair), inferTypes=['local/argmax'],
#         dictloss={
#             str(o.name): NBCrossEntropyDictLoss(weight=torch.tensor([ 4.5341,  0.5620]).to(device)),
#             str(location.name): NBCrossEntropyDictLoss(weight=torch.tensor([ 0.5194, 13.3925]).to(device)),
#             str(people.name): NBCrossEntropyDictLoss(weight=torch.tensor([ 0.5156, 16.5134]).to(device)), 
#             str(other.name): NBCrossEntropyDictLoss(weight=torch.tensor([ 0.5120, 21.4100]).to(device)),             
#             str(organization.name): NBCrossEntropyDictLoss(weight=torch.tensor([ 0.5098, 25.8953]).to(device)), 
#             str(work_for.name): NBCrossEntropyDictLoss(weight=torch.tensor([0.6277, 2.4578]).to(device)), 
#             str(located_in.name): NBCrossEntropyDictLoss(weight=torch.tensor([0.6270, 2.4677]).to(device)), 
#             str(live_in.name): NBCrossEntropyDictLoss(weight=torch.tensor([0.6748, 2.1306]).to(device)), 
#             str(orgbase_on.name): NBCrossEntropyDictLoss(weight=torch.tensor([0.6309, 2.4094]).to(device)), 
#             str(kill.name): NBCrossEntropyDictLoss(weight=torch.tensor([0.5730, 4.3231]).to(device)), 
#             "default": NBCrossEntropyDictLoss()},
        loss = MacroAverageTracker(NBCrossEntropyLoss()),
        sample = True,
        sampleSize=500, 
        sampleGlobalLoss = False,
        beta=1,
        metric={
            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

    return lbp, lbp1


def main(args):
    from graph import graph, sentence, word, phrase, pair
    program, program1 = model(device=args.gpu)

    split_id = args.split
    if args.number == 1:
        train_reader = SingletonDataLoader(f'data/conll04.corp_{split_id}_train.corp')
    else:
        train_reader = SingletonDataLoader(f'data/conll04.corp_{split_id}_train.corp_subsample_{args.number}.corp')
        
    test_reader = SingletonDataLoader(f'data/conll04.corp_{split_id}_test.corp')
    valid_reader = SingletonDataLoader(f'data/conll04.corp_{split_id}_valid.corp')
    
    def save_epoch(program, epoch=1):
        if args.number == 1:
            program.save(f'saves/conll04-bert-sample-{split_id}-{epoch}.pt')
        else:
            program.save(f'saves/conll04-bert-sample-{split_id}-{epoch}-size-{args.number}.pt')
        return epoch + 1

    def compute_scores(item, criteria="P"):
        entities = ["location", "people", "organization", "other"]
        relations = ["work_for", "located_in", "live_in", "orgbase_on", "kill"]
        instances = {"location": 937, "people": 774, "organization": 512, "other": 610, "work_for": 71, "located_in": 75, "live_in": 103, 
                     "orgbase_on": 97, "kill": 55}
        sum_entity = 0.0
        sum_relations = 0.0
        precision_entity = 0.0
        precision_relations = 0.0
        normal_precision_entity = 0.0
        normal_precision_relations = 0.0
        sum_all = 0.0
        precision_all = 0.0
        normal_precision_all = 0.0
        for key in entities:
            sum_entity += instances[key]
            precision_entity += instances[key] * item[key][criteria]
            normal_precision_entity += item[key][criteria]

        for key in relations:
            sum_relations += instances[key]
            precision_relations += instances[key] * item[key][criteria]
            normal_precision_relations += item[key][criteria]

        sum_all = sum_relations + sum_entity
        precision_all = precision_entity + precision_relations
        normal_precision_all = normal_precision_relations + normal_precision_entity

        outputs = {}
        
        if criteria == "P":
            outputs["micro_" + str(criteria) + "_entities"] = precision_entity / sum_entity
            outputs["micro_" + str(criteria) + "_relations"] = precision_relations / sum_relations
            outputs["micro_" + str(criteria) + "_all"] = precision_all / sum_all

        outputs["macro_" + str(criteria) + "_entities"] = normal_precision_entity / len(entities)
        outputs["macro_" + str(criteria) + "_relations"] = normal_precision_relations / len(relations)
        outputs["macro_" + str(criteria) + "_all"] = normal_precision_all / (len(entities) + len(relations))
        
        return outputs
    
    def save_best(program, epoch=1, best_epoch=-1, best_macro_f1=0):
        import logging
        logger = logging.getLogger(__name__)
        metrics = program.model.metric['argmax'].value()
        results = compute_scores(metrics, criteria="F1")
        score = results["macro_F1_all"]
        if score > best_macro_f1:
            logger.info(f'New Best Score {score} achieved at Epoch {epoch}.')
            best_epoch = epoch
            best_macro_f1 = score
            if args.number == 1:
                program.save(f'saves/conll04-bert-sample-{split_id}-best-macro-f1.pt')
            else:
                program.save(f'saves/conll04-bert-sample-{split_id}-size-{args.number}-best_macro-f1.pt')
        return epoch + 1, best_epoch, best_macro_f1
    
    if not args.load:
        program.after_train_epoch = [ProgramStorageCallback(program, save_epoch)]
        program.after_test_epoch = [ProgramStorageCallback(program, save_best)]
        program.train(train_reader, valid_set=valid_reader, test_set=test_reader, train_epoch_num=args.iteration, Optim=lambda param: torch.optim.SGD(param, lr=.001), device=args.gpu, c_warmup_iters=10)
#         train_epoch_callbacks=[ProgramStorageCallback(program, save_epoch)], valid_epoch_callbacks=[ProgramStorageCallback(program, save_best)])
    else:
        program.load(args.path)
        
    if not args.load:
        if args.number == 1:
            program1.load(f'saves/conll04-bert-sample-{split_id}-best-macro-f1.pt')
        else:
            program1.load(f'saves/conll04-bert-sample-{split_id}-size-{args.number}-best_macro-f1.pt')
        
    program1.test(test_reader, device=args.gpu)
    
    from datetime import datetime
    now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    if args.number == 1:
        program.save(f'saves/conll04-bert-sample-{split_id}-{now}.pt')
    else:
        program.save(f'saves/conll04-bert-sample-{split_id}-{now}_size_{args.number}.pt')
        

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Getting the arguments passed")
    parser.add_argument(
        "-s",
        "--split",
        help="The split",
        required=False,
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
    )
    parser.add_argument(
        "-n",
        "--number",
        help="Number of examples",
        type=float,
        required=False,
        choices=[1, 0.25, 0.1, 0.2],
        default=1,
    )
    parser.add_argument(
        "-i",
        "--iteration",
        help="Number of iterations",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "-l",
        "--load",
        help="Load?",
        type=bool,
        required=False,
        default=False,
    )
    
    parser.add_argument(
        "-p",
        "--path",
        help="Loading path",
        type=str,
        required=False,
        default=None,
    )
    
    parser.add_argument(
        "-g",
        "--gpu",
        help="GPU option",
        type=str,
        required=False,
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
            "cuda:1",
            "cuda:0",
            "cuda:2",
            "cuda:3",
            "cuda:4",
            "cuda:5",
            "cuda:6",
            "cuda:7",
        ],
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arguments()
    setProductionLogMode()
    main(args)
