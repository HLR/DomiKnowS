import torch
from torch.optim import SGD
from regr.program import SolverPOIProgram
from regr.graph import Graph, Concept, EnumConcept, Relation
from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor
from regr.sensor.pytorch.learners import ModuleLearner
from regr.program.loss import NBCrossEntropyLoss
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from models import POSLSTM, HeadLayer
import logging
from argparse import Namespace
from utils import LabelReader, make_words, create_reader_and_vocuabulary
import random
args = Namespace(
cuda_number=0,
epoch=10,
learning_rate=2e-3,
samplenum=10,
batch_size=14,
beta=0.1,
embedding_dim=300,
hidden_dim=300,
top_pos=2,
)

logging.basicConfig(level=logging.INFO)
args.device="cuda:"+str(args.cuda_number) if torch.cuda.is_available() else 'cpu'
print("device is :",args.device)
vocabulary,tag_list,reader=create_reader_and_vocuabulary(samplenum=args.samplenum,top_pos=args.top_pos)

random.seed(2021)
random.shuffle(reader)
train_reader,dev_reader,test_reader=reader[:len(reader)//10*6],reader[len(reader)//10*6:len(reader)//10*8],reader[len(reader)//10*8:]

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('pos_graph') as graph:
    Sentence = Concept(name='sentence')
    Word = Concept(name='word')
    sent_word_contains, = Sentence.contains(Word)
    Tag = Word(name="tag", ConceptClass=EnumConcept, values=tag_list)

Sentence['tokenized_text'] = ReaderSensor(keyword='sentecne')
Sentence['tokenized_pos'] = ReaderSensor(keyword='tags')
Word[sent_word_contains, "token", 'pos'] = JointSensor(Sentence['tokenized_text'],Sentence['tokenized_pos'],forward=make_words)
Word["hidden_layer"]=ModuleLearner("token", module=POSLSTM(embedding_dim=args.embedding_dim,\
                hidden_dim=args.hidden_dim, vocab_size=len(vocabulary),vocabulary=vocabulary,device=args.device))
Word[Tag] = FunctionalSensor(sent_word_contains, "pos", forward=LabelReader(device=args.device,tag_list=tag_list), label=True)
Word[Tag] = ModuleLearner("hidden_layer", module=HeadLayer( hidden_dim=args.hidden_dim, target_size=len(tag_list)))

program = SolverPOIProgram(graph,inferTypes=['local/argmax'],loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker(DatanodeCMMetric('local/argmax')))
program.train(reader[:len(reader)//10*8],valid_set=reader[len(reader)//10*8:], train_epoch_num=args.epoch, Optim=lambda param: SGD(param, lr=args.learning_rate),device=args.device)
program.train(reader[:len(reader)//10*8])

word_results=[]
word_labels=[]
for pic_num, sentence in enumerate(program.populate(reader[:len(reader)//10*8], device=args.device)):
    for word in sentence.getChildDataNodes():
        word_results.append(int(torch.argmax(word.getAttribute(Tag, "local/argmax"))))
        word_labels.append(int(word.getAttribute(Tag, "label").item()))

print("Final model accuracy is :",sum([i==j for i,j in zip(word_results,word_labels)])/len(word_results))