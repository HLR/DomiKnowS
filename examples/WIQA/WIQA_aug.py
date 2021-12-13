
import torch
from transformers import AdamW
from regr.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss
from regr.program.metric import MacroAverageTracker, PRF1Tracker, MetricTracker, CMWithLogitsMetric, DatanodeCMMetric
import logging
from transformers import get_linear_schedule_with_warmup
from regr.program.primaldualprogram import PrimalDualProgram
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor, FunctionalReaderSensor
from regr.graph.logicalConstrain import nandL, ifL, V, orL, andL, existsL, notL, atLeastL, atMostL, eqL, xorL, exactL
from regr.graph import Graph, Concept, Relation
from WIQA_reader import make_reader
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from regr.program import LearningBasedProgram, IMLProgram, SolverPOIProgram
from regr.program.model.pytorch import model_helper, PoiModel, SolverModel
from WIQA_utils import RobertaTokenizer,test_inference_results,join_model
from WIQA_models import WIQA_Robert, RobertaClassificationHead
import argparse
from WIQA_utils import guess_pair, guess_triple

parser = argparse.ArgumentParser(description='Run Wiqa Main Learning Code')
parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on',type=int)
parser.add_argument('--epoch', dest='cur_epoch', default=1, help='number of epochs you want your model to train on',type=int)
parser.add_argument('--lr', dest='learning_rate', default=2e-6, help='learning rate of the adamW optimiser',type=float)
parser.add_argument('--pd', dest='primaldual', default=False, help='whether or not to use primaldual constriant learning',type=bool)
parser.add_argument('--iml', dest='IML', default=False, help='whether or not to use IML constriant learning',type=bool)
parser.add_argument('--samplenum', dest='samplenum', default=100000000, help='number of samples to train the model on',type=int)
parser.add_argument('--batch', dest='batch_size', default=10, help='batch size for neural network training',type=int)
parser.add_argument('--beta', dest='beta', default=0.5, help='primal dual or IML multiplier',type=float)
parser.add_argument('--num_warmup_steps', dest='num_warmup_steps', default=5000, help='warmup steps for the transformer',type=int)
parser.add_argument('--num_training_steps', dest='num_training_steps', default=20000, help='total number of training steps for the transformer',type=int)
parser.add_argument('--verbose', dest='verbose', default=1, help='print the errors',type=int)
args = parser.parse_args()


# here we set the cuda we want to use and the number of maximum epochs we want to train our model
cuda_number= args.cuda_number
cur_device = "cuda:"+str(cuda_number) if torch.cuda.is_available() else 'cpu'

# our reader is a list of dictionaries and each dictionary has the attributes for the root node to read
reader_train_aug = make_reader(file_address="data/WIQA_AUG/train.jsonl", sample_num=args.samplenum,batch_size=args.batch_size)
reader_dev_aug = make_reader(file_address="data/WIQA_AUG/dev.jsonl", sample_num=args.samplenum,batch_size=args.batch_size)[:1]
#reader_test_aug = make_reader(file_address="data/WIQA_AUG/test.jsonl", sample_num=args.samplenum,batch_size=args.batch_size)

print("Graph Declaration:")
# reseting the graph
Graph.clear()
Concept.clear()
Relation.clear()

with Graph('WIQA_graph') as graph:
    #first we define paragrapg, then we define questions and add a constains relation from paragraph to question
    paragraph = Concept(name='paragraph')
    question = Concept(name='question')
    para_quest_contains, = paragraph.contains(question)

    # each question could be one the following three concepts:
    is_more = question(name='is_more')
    is_less = question(name='is_less')
    no_effect = question(name='no_effect')

    USE_LC_exactL = True
    USE_LC_atMostL = True

    USE_LC_symmetric  = True
    USE_LC_transitiveIsMore  = True
    USE_LC_transitiveIsLess = True
    
    # Only one of the labels to be true
    exactL(is_more, is_less, no_effect, active=USE_LC_exactL, name="exactL")
    atMostL(is_more, is_less, no_effect, active=USE_LC_atMostL, name="atMostL") # breakpoint in WIQA line 126

    # the symmetric relation is between questions that are opposite of each other and have opposing values
    symmetric = Concept(name='symmetric')
    s_arg1, s_arg2 = symmetric.has_a(arg1=question, arg2=question)

    # If a question is is_more and it has a symmetric relation with another question, then the second question should be is_less
    ifL(is_more('x'), is_less(path=('x', symmetric, s_arg2)), active=USE_LC_symmetric, name="symetric_is_more")
    
    # If a question is is_less and it has a symmetric relation with another question, then the second question should be is_more
    ifL(is_less('x'), is_more(path=('x', symmetric, s_arg2)), active=USE_LC_symmetric, name="symetric_is_less")

    # the transitive relation is between questions that have a transitive relation between them
    # meaning that the effect of the first question if the cause of the second question and the
    # third question is made of the cause of the first and the effect of the second question
    transitive = Concept(name='transitive')
    t_arg1, t_arg2, t_arg3 = transitive.has_a(arg11=question, arg22=question, arg33=question)

    # The transitive relation implies that if the first and the second question are is_more, so should be the third question. 
    ifL(andL(is_more('x'), is_more(path=('x', transitive, t_arg2))), is_more(path=('x', transitive, t_arg3)), active=USE_LC_transitiveIsMore, name="transitive_is_more")

    # If the first question is is_more and the second question is is_less, then the third question should also be is_less
    ifL(andL(is_more('x'), is_less(path=('x', transitive, t_arg2))), is_less(path=('x', transitive, t_arg3)), active=USE_LC_transitiveIsLess, name="transitive_is_less")

from IPython.display import Image
#graph.visualize("./image")
#Image(filename='image.png')

print("Sensor Part:")

# the first sensor reads the text property of the paragraph
paragraph['paragraph_intext'] = ReaderSensor(keyword='paragraph_intext')

# the following sensors read the concatenated properties of the questions related to this paragraph
# these concatenated properties will be splitted and put into their respective question later
paragraph['question_list'] = ReaderSensor(keyword='question_list')
paragraph['less_list'] = ReaderSensor(keyword='less_list')
paragraph['more_list'] = ReaderSensor(keyword='more_list')
paragraph['no_effect_list'] = ReaderSensor(keyword='no_effect_list')
paragraph['quest_ids'] = ReaderSensor(keyword='quest_ids')

def str_to_int_list(x):
    return torch.LongTensor([[int(i)] for i in x])

def make_questions(paragraph, question_list, less_list, more_list, no_effect_list, quest_ids):
    return torch.ones((len(question_list.split("@@")), 1)), [paragraph for i in range(len(question_list.split("@@")))], \
           question_list.split("@@"), str_to_int_list(less_list.split("@@")), str_to_int_list(more_list.split("@@")), \
           str_to_int_list(no_effect_list.split("@@")), quest_ids.split("@@")

# the joint sensor here reads the concatenated question properties from the paragraph and put them into questions
# the first property is the contain relation between paragraph and its questions
# the output of the joint sensor is a tuple with 7 members. each member correspond to its respective property in question
# for example the first element of the output tuple is a tensor of the shape (number of questions,1) meaning that each question
# has a contain relation with the ith question.
# the third element of the tuple is a list of str elements each a text of the a question related to the paragraph.
question[para_quest_contains, "question_paragraph", 'text', "is_more_", "is_less_", "no_effect_", "quest_id"] = JointSensor(
    paragraph['paragraph_intext'], paragraph['question_list'], paragraph['less_list'], paragraph['more_list'],
    paragraph['no_effect_list'], paragraph['quest_ids'],forward=make_questions)

# we use another joint sensor, so given a roberta transformer tokenizer, we use the paragraph and question text, concat them and feed them
# to the tokenizer and get the token_ids and masks for the <s>paragraph</s></s>question</s>
question["token_ids", "Mask"] = JointSensor(para_quest_contains, "question_paragraph", 'text',forward=RobertaTokenizer())

def label_reader(_, label):
    return label

# we load the is_more, is_less and no_effect attributes with label=True to tell the program
# about the real label of these questions for training
question[is_more] = FunctionalSensor(para_quest_contains, "is_more_", forward=label_reader, label=True)
question[is_less] = FunctionalSensor(para_quest_contains, "is_less_", forward=label_reader, label=True)
question[no_effect] = FunctionalSensor(para_quest_contains, "no_effect_", forward=label_reader, label=True)

roberta_model = WIQA_Robert()

# we calculate the encoding for each <s>paragraph</s></s>question</s> by feeding it to roberta model
# this is a ModuleLearner meaning that the parameters of roberta will be trained during learning
question["robert_emb"] = ModuleLearner("token_ids", "Mask", module=roberta_model)

# the CompositionCandidateSensor takes two or three questions and return True or false if they are symmetric or transitive respectively

symmetric[s_arg1.reversed, s_arg2.reversed] = CompositionCandidateSensor(question['quest_id'],relations=(s_arg1.reversed, s_arg2.reversed),forward=guess_pair)
transitive[t_arg1.reversed, t_arg2.reversed, t_arg3.reversed] = CompositionCandidateSensor(question['quest_id'],relations=(t_arg1.reversed,t_arg2.reversed,t_arg3.reversed),forward=guess_triple)

# finally the embedding are used to learn the label of each question by defining a linear model on top of roberta


question[is_more] = ModuleLearner("robert_emb", module=RobertaClassificationHead(roberta_model.last_layer_size))
question[is_less] = ModuleLearner("robert_emb", module=RobertaClassificationHead(roberta_model.last_layer_size))
question[no_effect] = ModuleLearner("robert_emb", module=RobertaClassificationHead(roberta_model.last_layer_size))

# in our program we define POI ( points of interest) that are the final Concepts we want to be calculated
# other inputs are graph, loss function and the metric

if not args.primaldual and not args.IML:
    print("simple program")
    program = SolverPOIProgram(graph, poi=[question[is_less], question[is_more], question[no_effect],\
                                    symmetric, transitive],loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())
if args.primaldual:
    print("primal dual program")
    program = PrimalDualProgram(graph, SolverModel, poi=[question[is_less], question[is_more], question[no_effect],\
                                    symmetric, transitive],inferTypes=['local/argmax'],loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=args.beta)),beta=args.beta)
if args.IML:
    print("IML program")
    program = IMLProgram(graph, poi=[question[is_less], question[is_more], question[no_effect],\
                                    symmetric, transitive],loss=MacroAverageTracker(BCEWithLogitsIMLoss(lmbd=args.beta)), metric=PRF1Tracker())

logging.basicConfig(level=logging.INFO)

from os import path
if not path.exists("new_domi_1"):
    join_model("domi_1_20","new_domi_1")

# at the end we run our program for each epoch and test the results each time

for i in range(args.cur_epoch):
    print("this epoch is number:",i,"&"*10)
    class SchCB():
        def __init__(self, program) -> None:
            self.program = program
            self._sch = None
        @property
        def sch(self):
            if self._sch is None:
                self._sch = get_linear_schedule_with_warmup(self.program.opt, num_warmup_steps = args.num_warmup_steps, num_training_steps = args.num_training_steps)
            return self._sch

        def __call__(self) -> None:
            self.sch.step()

    program.load("new_domi_1", map_location={'cuda:5':'cpu'})# in case we want to load the model instead of training
    #program.train(reader_train_aug, train_epoch_num=1, Optim=lambda param: AdamW(param, lr = args.learning_rate,eps = 1e-8 ), device=cur_device)#, train_step_callbacks=[SchCB(program)])
    #program.save("domi_"+str(i)) in case of saving the parameters of the model

    print('-' * 40,"\n",'Training result:')
    print(program.model.loss)
    if args.primaldual:
        print(program.cmodel.loss)

    print("***** dev aug *****")
    test_inference_results(program, reader_dev_aug, cur_device, is_more, is_less, no_effect, transitive, symmetric, args.verbose)
    #test_inference_results(program, reader_train_aug, cur_device, is_more, is_less, no_effect, transitive, symmetric,args.verbose)
    print("***** test aug *****")
    #test_inference_results(program,reader_test_aug,cur_device,is_more,is_less,no_effect,args.verbose)
