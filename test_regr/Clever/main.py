import sys
import os
os.environ["GRB_LICENSE_FILE"] = "/home/premsrit/gurobi.lic"
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')
# export GRB_LICENSE_FILE=/full/path/to/gurobi.lic
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor, FunctionalReaderSensor, ModuleSensor
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from preprocess import preprocess_dataset, preprocess_folders_and_files
from graph import create_graph
from pathlib import Path
from modules import ResNetPatcher, DummyLinearLearner, LEFTObjectEMB, LEFTRelationEMB, ResnetLEFT, LinearLayer
from dataset import g_relational_concepts
import argparse, torch, logging

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Logic-guided VQA training / evaluation")
parser.add_argument("--train-size", type=int, default=None,help="Number of training examples to sample (default: use full set)")
parser.add_argument("--test-size", type=int, default=None,help="Number of test examples to sample (default: use full set)")
parser.add_argument("--epochs", type=int, default=4,help="Number of training epochs")
parser.add_argument("--lr", "--learning-rate", type=float, default=1e-6,help="Learning rate")
parser.add_argument("--batch-size", type=int, default=1,help="Mini-batch size")
parser.add_argument("--eval-only", action="store_true",help="Skip training; just load a checkpoint and evaluate")
parser.add_argument("--dummy", action="store_true",help="Use the lightweight dummy configuration")
parser.add_argument("--tnorm", choices=["G", "P", "L"], default="G",help="T-norm used inside InferenceProgram")
args=parser.parse_args()

CACHE_DIR = preprocess_folders_and_files(args.dummy)
NUM_INSTANCES  = 10
device = "cuda:4"

def filter_relation(property, arg1, arg2):
    # This is default of LEFT framework that perform all pair relation
    return arg1.getAttribute("image_id") == arg2.getAttribute("image_id")

dataset = preprocess_dataset(args,NUM_INSTANCES,CACHE_DIR)
questions_executions, graph,image,object,image_object_contains,obj1,obj2,relaton_2_obj,attribute_names_dict  = create_graph(dataset)

for i in range(len(dataset)):
    dataset[i]["logic_str"] = questions_executions[i]
    dataset[i]["logic_label"] = torch.LongTensor([bool(dataset[i]['answer'])]).to(device)

image["pil_image"] = FunctionalReaderSensor(keyword="pil_image",forward=lambda data: [data])
image["image_id"] = FunctionalReaderSensor(keyword='image_index',forward=lambda data: [data])

object["bounding_boxes"]= FunctionalReaderSensor(keyword="objects_raw", forward= lambda data: torch.Tensor(data).to(device))
object["properties"]= ReaderSensor(keyword="all_objects")
object["image_id"]= FunctionalSensor(image["image_id"], "bounding_boxes", forward= lambda data, data2: data * len(data2))

if not args.dummy:
    resnet_model = ResnetLEFT(device=device)
    image["emb"] = ModuleSensor("image_id", "pil_image", module=resnet_model, device=device)
    # model = ResNetPatcher(resnet_model_name='resnet50', pretrained=True, device=device)
    # object["emb"] = FunctionalSensor(image["image_id"],image["pil_image"],"bounding_boxes", forward=model)
    object_feature_extraction_model = LEFTObjectEMB(device=device)
    object["feature_emb"] = ModuleLearner(image["emb"], "bounding_boxes", module=object_feature_extraction_model, device=device)
    object_feature_fc = LinearLayer(128 * 32 * 32, 128, device=device)
    object["emb"] = ModuleLearner("feature_emb", "bounding_boxes", module=object_feature_fc, device=device)

object[image_object_contains] = EdgeSensor(object["bounding_boxes"], image["pil_image"], relation=image_object_contains, forward=lambda b, _: torch.ones(len(b)).unsqueeze(-1))

relaton_2_obj[obj1.reversed, obj2.reversed] = CompositionCandidateSensor(
        object['image_id'],
        relations=(obj1.reversed, obj2.reversed),
        forward=filter_relation)

object_relation_extraction = LEFTRelationEMB(device=device)
relaton_2_obj["emb"] = ModuleLearner(image["emb"], object["bounding_boxes"], object["feature_emb"], module=object_relation_extraction, device=device)

for attr_name,attr_variable in attribute_names_dict.items():
    # print(attr_name)
    attribute_org = attr_name.split("_")[1]
    # TODO: Change this to generlize way
    if attribute_org in g_relational_concepts["spatial_relation"]:
        # scene, box, object_features
        relaton_2_obj[attr_variable] = ModuleLearner("emb", module=torch.nn.Linear(128,2).to(device),device=device)
    elif args.dummy:
        object[attr_variable] = DummyLinearLearner(image_object_contains,"properties",current_attribute=attr_name)
        object[attr_variable] = FunctionalSensor(image_object_contains, forward=lambda label: torch.ones(len(label)).unsqueeze(-1), label=True)
    else:
        object[attr_variable] = ModuleLearner("emb", module=torch.nn.Linear(128,2).to(device),device=device)



# pair[relation_variable] = ModuleLearner


dataset = graph.compile_logic(dataset, logic_keyword='logic_str',logic_label_keyword='logic_label')
program = InferenceProgram(graph,SolverModel,poi=[image,object,*attribute_names_dict.values(), graph.constraint, relaton_2_obj],device=device,tnorm=args.tnorm)
save_file = Path(f"models/program{args.lr}_{args.epochs}_{args.batch_size}_{args.train_size}_{args.tnorm}.pth")
if not args.eval_only:
    acc = program.evaluate_condition(dataset, device=device)
    print("Accuracy before training: {:.2f}".format(acc * 100))
    for i in range(args.epochs):
        print(f"Training epoch {i+1}/{args.epochs}")
        save_file = Path(f"models/program{args.lr}_{i+1}_{args.batch_size}_{args.train_size}_{args.tnorm}.pth")
        program.train(dataset,Optim=torch.optim.Adam,train_epoch_num=1,lr=args.lr,c_warmup_iters=0,batch_size=args.batch_size,device=device,print_loss=False)
        program.save(save_file)
    acc = program.evaluate_condition(dataset, device=device)
    print("Accuracy on Test: {:.2f}".format(acc * 100))
else:
    print("Loading program from checkpoint...")
    program.load(save_file)
    acc = program.evaluate_condition(dataset, device=device)
    print("Accuracy on Test: {:.2f}".format(acc * 100))