import argparse
import sys

sys.path.append('.')
sys.path.append('../..')
from domiknows.program.lossprogram import SampleLossProgram, PrimalDualProgram
from domiknows.program.model.pytorch import SolverModel


import torch
from torchvision.models import resnet18, resnet50, resnet101, resnet152

from domiknows.program import SolverPOIProgram, IMLProgram
from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from reader import create_readers
import torch.nn as nn
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss


# Enable skeleton DataNode


class ImageNetwork(torch.nn.Module):
    def __init__(self):
        super(ImageNetwork, self).__init__()
        self.conv = resnet18(pretrained=False)

    def forward(self, x):
        x = self.conv(x)
        return x
    
def main():
    from domiknows.utils import setProductionLogMode
    productionMode = True
    if productionMode:
        setProductionLogMode(no_UseTimeLog=True)
    from domiknows.utils import setDnSkeletonMode
    setDnSkeletonMode(True)
    import logging
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument('--nameload', dest='nameload', default="none", help='model name to load', type=str)
    parser.add_argument('--nameloadprogram', dest='nameloadprogram', default="none", help='model name to load', type=str)
    parser.add_argument('--namesave', dest='namesave', default="none", help='model name to save', type=str)
    parser.add_argument('--cuda', dest='cuda_number', default=0, help='cuda number to train the models on', type=int)
    parser.add_argument('--ilp', dest='ilp', default=True, help='whether or not to use ilp', type=bool)
    parser.add_argument('--pd', dest='primaldual', default=False,help='whether or not to use primaldual constriant learning', type=bool)
    parser.add_argument('--iml', dest='IML', default=False, help='whether or not to use IML constriant learning',type=bool)
    parser.add_argument('--sam', dest='sam', default=False, help='whether or not to use sampling learning', type=bool)

    parser.add_argument('--test', dest='test', default=False, help='dont train just test', type=bool)
    parser.add_argument('--verbose', dest='verbose', default=False, help='print improved and damaged examples', type=bool)

    parser.add_argument('--resnet', dest='resnet', default=50, help='value of learning rate', type=int)

    parser.add_argument('--samplenum', dest='samplenum', default=999999999,help='number of samples to choose from the dataset',type=int)
    parser.add_argument('--epochs', dest='epochs', default=19, help='number of training epoch', type=int)
    parser.add_argument('--lambdaValue', dest='lambdaValue', default=0.5, help='value of learning rate', type=float)
    parser.add_argument('--lr', dest='learning_rate', default=2e-4, help='learning rate of the adam optimiser',type=float)
    parser.add_argument('--beta', dest='beta', default=0.1, help='primal dual or IML multiplier', type=float)

    parser.add_argument('--graph_type', dest='graph_type', default="exactL_ifLorLtopdown", help='type of constraints to be defined', type=str)
    args = parser.parse_args()

    if args.graph_type=="nothing":
        from graph import graph, image_group_contains, image, category, Label, image_group,parent_names,children_names,structure
    elif args.graph_type=="only_exactL":
        from graph_only_exactL import graph, image_group_contains, image, category, Label, image_group,parent_names,children_names,structure
    elif args.graph_type == "exactL_ifLorLtopdown":
        from graph_exactL_ifLorLtopdown import graph, image_group_contains, image, category, Label, image_group,parent_names,children_names,structure
    elif args.graph_type == "exactL_ifLorLbottomup":
        from graph_exactL_ifLorLbottomup import graph, image_group_contains, image, category, Label, image_group,parent_names,children_names,structure
    elif args.graph_type == "exactL_ifLorLbothways":
        from graph_exactL_ifLorLbothways import graph, image_group_contains, image, category, Label, image_group,parent_names,children_names,structure
    elif args.graph_type == "exactL_nandL":
        from graph_exactL_nandL import graph, image_group_contains, image, category, Label, image_group,parent_names,children_names,structure

    children_names_reverse={i:j for j,i in children_names.items()}
    parent_names_reverse = {i: j for j, i in parent_names.items()}

    child_to_parent_dict=dict()
    for parent in structure.keys():
       for child in list(structure[parent]):
           child_to_parent_dict[children_names_reverse[child]]=parent_names_reverse[parent]

    device = "cuda:"+str(args.cuda_number)

    image_group['pixels_group'] = ReaderSensor(keyword='pixels', device=device)
    image_group['category_group'] = ReaderSensor(keyword='corase_label', device=device)
    image_group['tag_group'] = ReaderSensor(keyword='fine_label', device=device)

    def str_to_int_list(x):
        return torch.LongTensor([[int(i)] for i in x])

    def make_images(pixels_group, category_group, tag_group):
        return torch.ones((len(category_group.split("@@")), 1)), torch.squeeze(pixels_group, 0),["parent" for i in range(len(category_group.split("@@")))],\
               ["child" for i in range(len(category_group.split("@@")))], str_to_int_list(category_group.split("@@")), str_to_int_list(tag_group.split("@@"))

    image[image_group_contains, "pixels","parent","child", 'category_', "tag_"] = JointSensor(image_group['pixels_group'],
                                                                             image_group["category_group"],
                                                                             image_group["tag_group"],
                                                                             forward=make_images)
    def label_reader(_, label):
        return label

    image[category] = FunctionalSensor(image_group_contains, "category_", forward=label_reader, label=True)
    image[Label] = FunctionalSensor(image_group_contains, "tag_", forward=label_reader, label=True)

    class CIFAR100Model(torch.nn.Module):

        def __init__(self):
            super(CIFAR100Model, self).__init__()
            # resnet18, resnet50, resnet101, resnet152
            if args.resnet == 18:
                self.res_p = resnet18(pretrained=True)
                self.res_c = resnet18(pretrained=True)
            elif args.resnet == 50:
                self.res_p = resnet50(pretrained=True)
                self.res_c = resnet50(pretrained=True)
            elif args.resnet == 101:
                self.res_p = resnet101(pretrained=True)
                self.res_c = resnet101(pretrained=True)
            elif args.resnet == 152:
                self.res_p = resnet152(pretrained=True)
                self.res_c = resnet152(pretrained=True)

            self.l1 = torch.nn.Linear(1000, 20)
            self.l2 = torch.nn.Linear(1000, 100)


        def forward(self, input, mode="parent"):
            if mode[0] == "parent":
                return self.l1(self.res_p(input))
            return self.l2(self.res_c(input))

    model=CIFAR100Model()
    #image['emb'] = ModuleLearner('pixels', module=resnet18(pretrained=True))

    image[category] = ModuleLearner('pixels',"parent", module=model)
    image[Label] = ModuleLearner('pixels',"child", module=model)

    f = open(str(args.ilp)+"_"+str(args.samplenum)+"_"+"_"+str(args.beta)+".txt", "w")
    print("POI")
    program = SolverPOIProgram(graph, inferTypes=['local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()),
                               metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))}, f=f)

    if args.ilp:
        print("ILP")
        program = SolverPOIProgram(graph, inferTypes=['ILP', 'local/argmax'],
                                   loss=MacroAverageTracker(NBCrossEntropyLoss()) \
                                   , metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                             'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))}, f=f)

    if args.primaldual:
        print("PrimalDualProgram")
        program = PrimalDualProgram(graph, SolverModel, poi=(image,), inferTypes=['local/argmax'],
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},beta=args.lambdaValue,f=f)
    if args.primaldual and args.ilp:

        print("PrimalDualProgram ILP")
        program = PrimalDualProgram(graph, SolverModel, poi=(image,), inferTypes=['ILP', 'local/argmax'],
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()),
                                    metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},beta=args.lambdaValue,f=f)
    if args.sam:

        print("sam")
        program = SampleLossProgram(graph, SolverModel,inferTypes=['local/argmax'],metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},loss=MacroAverageTracker(NBCrossEntropyLoss()),sample=True,sampleSize=250,sampleGlobalLoss=False,beta=args.beta,device=device)
    if args.sam and args.ilp:

        print("sam ILP")
        program = SampleLossProgram(graph, SolverModel, inferTypes=['ILP', 'local/argmax'],
                                    metric={'ILP': PRF1Tracker(DatanodeCMMetric()),
                                            'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))},
                                    loss=MacroAverageTracker(NBCrossEntropyLoss()), sample=True, sampleSize=250,
                                    sampleGlobalLoss=False, beta=args.beta, device=device)

    train_reader,test_reader=create_readers(train_num=min(args.samplenum,50000),test_num= min(args.samplenum,10000//4))

    if not args.nameload=="none":
        model.load_state_dict(torch.load(args.nameload))
        model.to(device)

    if len(test_reader) > len(train_reader):
        test_reader = test_reader[:len(train_reader)]

    if not args.test:
        for i in range(args.epochs):
            program.train(train_reader,valid_set=test_reader, train_epoch_num=2, Optim=lambda param: torch.optim.Adam(param, lr=args.learning_rate),device=device)
            program.save(args.namesave + "_" + str(i))
    f.close()
    guessed_tag = {
        "local/softmax": [],
        "ILP": []
    }
    real_tag = []
    guessed_category = {
        "local/softmax": [],
        "ILP": []
    }
    counter_list=[0,0,0,0]
    program.load(args.nameloadprogram)
    #program.test(test_reader)
    program.verifyResultsLC(test_reader, constraint_names=None, device=device)
    real_category = []
    ac_,t_=0,0

    for pic_num, picture_group in enumerate(program.populate(test_reader, device=device)):
        picture_group.inferILPResults()
        verifyResult = picture_group.verifyResultsLC()
        verifyResultILP = picture_group.verifyResultsLC()
        ac_ += sum([verifyResultILP[lc]['satisfied'] for lc in verifyResultILP])
        t_ += len(verifyResultILP.keys())
        for image_ in picture_group.getChildDataNodes():
            for key in ["local/softmax", "ILP"]:
                if key == "ILP":
                    guessed_tag[key].append(int(torch.argmax(image_.getAttribute(Label, key))))
                else:
                    guessed_tag[key].append(int(torch.argmax(image_.getAttribute(Label, key))))

                if key == "ILP":
                    guessed_category[key].append(int(torch.argmax(image_.getAttribute(category, key))))
                else:
                    guessed_category[key].append(int(torch.argmax(image_.getAttribute(category, key))))
            real_tag.append(int(image_.getAttribute(Label, "label")[0]))
            real_category.append(int(image_.getAttribute(category, "label")))
            flag=False
            if counter_list[0]<5 and guessed_tag["local/softmax"][-1]==real_tag[-1] and not guessed_tag["ILP"][-1]==real_tag[-1] and args.verbose:
                counter_list[0]+=1
                print("tag messed up:")
                flag=True

            if counter_list[1]<5 and guessed_category["local/softmax"][-1]==real_category[-1] and not guessed_category["ILP"][-1]==real_category[-1]  and args.verbose:
                counter_list[1] += 1
                print("cathegory messed up:")
                flag = True

            if counter_list[2]<5 and guessed_tag["ILP"][-1]==real_tag[-1] and not guessed_tag["local/softmax"][-1]==real_tag[-1]  and args.verbose:
                counter_list[2] += 1
                print("tag improved:")
                flag = True

            if counter_list[3]<5 and guessed_category["ILP"][-1]==real_category[-1] and not guessed_category["local/softmax"][-1]==real_category[-1]  and args.verbose:
                counter_list[3] += 1
                print("cathegory improved")
                flag = True

            if flag  and args.verbose:
                print("cathegory(parent) label, softmax, ILP :",real_category[-1],guessed_category["local/softmax"][-1],guessed_category["ILP"][-1])
                print("cathegory(parent) label, softmax, ILP :", parent_names[real_category[-1]],parent_names[guessed_category["local/softmax"][-1]], parent_names[guessed_category["ILP"][-1]])
                print("cathegory(parent) label, softmax, ILP :",image_.getAttribute(category, "local/softmax")[real_category[-1]].item(),\
                      image_.getAttribute(category, "local/softmax")[guessed_category["local/softmax"][-1]].item(),image_.getAttribute(category, "local/softmax")[guessed_category["ILP"][-1]].item())

                print("tag(child) label, softmax, ILP :", real_tag[-1],guessed_tag["local/softmax"][-1], guessed_tag["ILP"][-1])
                print("tag(child) label, softmax, ILP :", children_names[real_tag[-1]],children_names[guessed_tag["local/softmax"][-1]], children_names[guessed_tag["ILP"][-1]])
                print("tag(child) label, softmax, ILP :",image_.getAttribute(Label, "local/softmax")[real_tag[-1]].item(), \
                      image_.getAttribute(Label, "local/softmax")[guessed_tag["local/softmax"][-1]].item(), image_.getAttribute(Label, "local/softmax")[guessed_tag["ILP"][-1]].item())

                print("probability of parents of tag(child) label, softmax, ILP :",
                      image_.getAttribute(category, "local/softmax")[child_to_parent_dict[real_tag[-1]]].item(), \
                      image_.getAttribute(category, "local/softmax")[child_to_parent_dict[guessed_tag["local/softmax"][-1]]].item(),
                      image_.getAttribute(category, "local/softmax")[child_to_parent_dict[guessed_tag["ILP"][-1]]].item())

                softmax_children=[]
                names_softmax_children=[]
                for i in structure[parent_names[guessed_category["local/softmax"][-1]]]:
                    softmax_children.append(image_.getAttribute(Label, "local/softmax")[children_names_reverse[i]].item())
                    names_softmax_children.append(i)

                ILP_children = []
                names_ILP_children=[]
                for i in structure[parent_names[guessed_category["ILP"][-1]]]:
                    ILP_children.append(image_.getAttribute(Label, "local/softmax")[children_names_reverse[i]].item())
                    names_ILP_children.append(i)

                print("names of the children of softmax parent:", names_softmax_children)
                print("Probabilities of the children of softmax parent:",softmax_children)
                print("names of the children of ILP parent:", names_ILP_children)
                print("Probabilities of the children of ILP parent:", ILP_children)


    for key in ["local/softmax", "ILP"]:
        print(f"##############################{key}#########################")
        guessed_labels = guessed_tag[key]
        real_labels = real_tag
        #print(guessed_labels)
        #print(real_labels)
        correct = sum(1 if x == y else 0 for x, y in zip(real_labels, guessed_labels))
        total = len(real_labels)+1
        print("tags accuracy", correct / total)
        guessed_labels = guessed_category[key]
        real_labels = real_category
        #print(guessed_labels)
        #print(real_labels)
        correct = sum(1 if x == y else 0 for x, y in zip(real_labels, guessed_labels))
        total = len(real_labels)+1
        print("category accuracy", correct / total)
    print("constraint accuracy: ", ac_ / t_ )

if __name__ == '__main__':
    main()