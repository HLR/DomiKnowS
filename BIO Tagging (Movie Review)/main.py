from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor
from graph import graph,NERSentence,tag,NER_contains,Generated_label,next_word,first_word, second_word
from reader import parse_corpus,encode_examples
from transformers import T5ForConditionalGeneration, AdamW
from transformers import T5Tokenizer
from domiknows.sensor.pytorch.learners import ModuleLearner
import torch
from domiknows.program import SolverPOIProgram
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss
import logging
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
logging.basicConfig(level=logging.INFO)

from model import FilteredT5Model
model_name='t5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
new_tokens = ['<B>', '<I>', '<O>']
tokenizer.add_tokens(new_tokens)
print(tokenizer.convert_tokens_to_ids(new_tokens))
reader_data=parse_corpus(file_path="trivia10k13train.bio",for_domiknows=True,samplenum=100)

device='cuda:1'
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

NERSentence['sentence'] = ReaderSensor(keyword='sentence', device=device)
NERSentence['taglabels'] = ReaderSensor(keyword='taglabels', device=device)
NERSentence['sentence_ids']=FunctionalSensor("sentence", forward=lambda input: tokenizer(f"ner: {input}", return_tensors="pt")['input_ids'])

tag[NER_contains, "taglabels_"] = JointSensor(NERSentence["taglabels"],forward=lambda taglabels:(torch.ones((len(taglabels.split(" ")), 1)), taglabels.split(" ")))

tag["Generated_label_"] = FunctionalSensor(NER_contains, "taglabels_", forward=lambda x,y:torch.LongTensor([[{'<B>':0, '<I>':1, '<O>':2}.get(i)] for i in y]))
tag[Generated_label] = FunctionalSensor(NER_contains,"Generated_label_", forward=lambda x,y:y, label=True)
tag[Generated_label] = ModuleLearner(NER_contains,NERSentence['sentence_ids'],tag["Generated_label_"], module=FilteredT5Model(model_name, tokenizer))


def link_words(tags):
    n=len(tags)
    return torch.eye(n,dtype=torch.long),torch.nn.functional.pad(torch.eye(n-1), (1, 0, 0, 1), "constant", 0)

next_word[first_word.reversed, second_word.reversed] = JointSensor(tag, forward=link_words)


program = SolverPOIProgram(graph,poi=[tag[Generated_label],next_word], inferTypes=['local/argmax',"ILP"], loss=MacroAverageTracker(NBCrossEntropyLoss()),
                           metric={'ILP': PRF1Tracker(DatanodeCMMetric()),'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))})

program.train(reader_data,valid_set=reader_data, train_epoch_num=10, Optim=lambda param: AdamW(param, lr=1e-4),device=device)
