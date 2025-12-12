import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
try:
    from monitor.constraint_monitor import enable_monitoring # type: ignore
    MONITORING_AVAILABLE = True
    # Enable in slave mode - will post data to master at localhost:8080
    enable_monitoring(slave_mode=True, master_url="http://localhost:8080")
except ImportError:
    MONITORING_AVAILABLE = False

from internVL_vLLM import InternVLShared as InternVL
# export GRB_LICENSE_FILE=/full/path/to/gurobi.lic
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor, FunctionalReaderSensor, ModuleSensor
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from preprocess import preprocess_dataset, preprocess_folders_and_files
from graph import create_graph
from pathlib import Path
from modules import  LEFTObjectEMB, LEFTRelationEMB, ResnetLEFT, LinearLayer
from dataset import g_relational_concepts
import argparse, torch, logging

from pathlib import Path

RUN_DIR = Path(__file__).parent.resolve()
MODEL_DIR = RUN_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def ckpt_path(lr, epoch_idx, load_epoch_tag, batch, tnorm, subset):
    return MODEL_DIR / f"program{lr}_{epoch_idx}_{load_epoch_tag}__{batch}_6000_{tnorm}_{subset}.pth"

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Logic-guided VQA training / evaluation")
parser.add_argument("--train-size", type=int, default=None,help="Number of training examples to sample (default: use full set)")
parser.add_argument("--test-size", type=int, default=None,help="Number of test examples to sample (default: use full set)")
parser.add_argument("--epochs", type=int, default=4,help="Number of training epochs")
parser.add_argument("--lr", "--learning-rate", type=float, default=1e-6,help="Learning rate")
parser.add_argument("--batch-size", type=int, default=1,help="Mini-batch size")
parser.add_argument("--subset", type=int, default=-1,help="Mini sub-set")
parser.add_argument("--load-epoch", type=int, default=0,help="Load previous epoch")
parser.add_argument("--eval-only", action="store_true",help="Skip training; just load a checkpoint and evaluate")
parser.add_argument("--use-vlm", action="store_true",help="use InternVL for predictions")
parser.add_argument("--dummy", action="store_true",help="Use the lightweight dummy configuration")
parser.add_argument("--tnorm", choices=["G", "P", "L"], default="G",help="T-norm used inside InferenceProgram")
parser.add_argument("--load_previous_save", action="store_true",help="Whether to use previous save")
args=parser.parse_args()

CACHE_DIR = preprocess_folders_and_files(args.dummy)
NUM_INSTANCES  = 10
device = "cpu"

def filter_relation(property, arg1, arg2):
    # This is default of LEFT framework that perform all pair relation
    return arg1.getAttribute("image_id") == arg2.getAttribute("image_id")

dataset = preprocess_dataset(args,NUM_INSTANCES,CACHE_DIR)
questions_executions, graph,image,object,image_object_contains,obj1,obj2,relaton_2_obj,attribute_names_dict  = create_graph(dataset)

print(len(dataset))
for i in range(len(dataset)):
    dataset[i]["logic_str"] = questions_executions[i]
    dataset[i]["logic_label"] = torch.LongTensor([bool(dataset[i]['answer'])]).to(device)

image["pil_image"] = FunctionalReaderSensor(keyword="pil_image",forward=lambda data: [data])
image["image_id"] = FunctionalReaderSensor(keyword='image_index',forward=lambda data: [data])

object["bounding_boxes"]= FunctionalReaderSensor(keyword="objects_raw", forward= lambda data: torch.Tensor(data).to(device))
object["properties"]= ReaderSensor(keyword="all_objects")
object["image_id"]= FunctionalSensor(image["image_id"], "bounding_boxes", forward= lambda data, data2: data * len(data2))

# if not args.dummy:
if not args.use_vlm:
    resnet_model = ResnetLEFT(device=device)
    image["emb"] = ModuleSensor("image_id", "pil_image", module=resnet_model, device=device)
    # model = ResNetPatcher(resnet_model_name='resnet50', pretrained=True, device=device)
    # object["emb"] = FunctionalSensor(image["image_id"],image["pil_image"],"bounding_boxes", forward=model)
    object_feature_extraction_model = LEFTObjectEMB(device=device)
    object["feature_emb"] = ModuleLearner(image["emb"], "bounding_boxes", module=object_feature_extraction_model, device=device)
    object_feature_fc = LinearLayer(128 * 32 * 32, 1024, device=device)
    object["emb"] = ModuleLearner("feature_emb", "bounding_boxes", module=object_feature_fc, device=device)


object[image_object_contains] = EdgeSensor(object["bounding_boxes"], image["pil_image"], relation=image_object_contains, forward=lambda b, _: torch.ones(len(b)).unsqueeze(-1))

relaton_2_obj[obj1.reversed, obj2.reversed] = CompositionCandidateSensor(object['image_id'],relations=(obj1.reversed, obj2.reversed),forward=filter_relation)

if not args.use_vlm:
    object_relation_extraction = LEFTRelationEMB(input_size=256, output_size=1024, device=device)
    relaton_2_obj["emb"] = ModuleLearner(image["emb"], object["bounding_boxes"], object["feature_emb"], module=object_relation_extraction, device=device)

for attr_name,attr_variable in attribute_names_dict.items():
    attribute_org = attr_name.split("_")[1]
    if not args.use_vlm:
        if attribute_org in g_relational_concepts["spatial_relation"]:
            # scene, box, object_features
            relaton_2_obj[attr_variable] = ModuleLearner("emb", module=torch.nn.Linear(1024,2).to(device),device=device)
        else:
            object[attr_variable] = ModuleLearner("emb", module=torch.nn.Linear(1024,2).to(device),device=device)
    else:
        MODEL_PATH = "OpenGVLab/InternVL3_5-1B"
        if attribute_org in g_relational_concepts["spatial_relation"]:
            relaton_2_obj[attr_variable] = ModuleLearner(image["pil_image"], object["bounding_boxes"], module=InternVL(model_path=MODEL_PATH, device=device,relation = 2,attr = attr_name),device=device)
        else:
            object[attr_variable] = ModuleLearner(image["pil_image"], object["bounding_boxes"], module=InternVL(model_path=MODEL_PATH, device=device, relation = 1,attr = attr_name),device=device)

dataset = graph.compile_logic(dataset, logic_keyword='logic_str',logic_label_keyword='logic_label')
program = InferenceProgram(graph,SolverModel,poi=[image,object,*attribute_names_dict.values(), graph.constraint, relaton_2_obj],device=device,tnorm=args.tnorm)

save_file = ckpt_path(args.lr, 1, args.load_epoch, args.batch_size, args.tnorm, args.subset)

if not args.eval_only:
    if args.load_previous_save and args.subset > 1:
        previous_save = ckpt_path(args.lr, 1, args.load_epoch, args.batch_size, args.tnorm, args.subset - 1)
        assert previous_save.exists(), f"Missing checkpoint: {previous_save}"
        program.load(previous_save)
    elif args.load_previous_save and args.load_epoch > 0:
        previous_save = ckpt_path(args.lr, 1, args.load_epoch - 1, args.batch_size, args.tnorm, args.subset)
        assert previous_save.exists(), f"Missing checkpoint: {previous_save}"
        program.load(previous_save)

    for i in range(args.epochs):
        print(f"Training epoch {i+1}/{args.epochs}")
        save_file = ckpt_path(args.lr, i+1, args.load_epoch, args.batch_size, args.tnorm, args.subset)
        program.train(dataset, Optim=torch.optim.Adam, train_epoch_num=1, c_lr=args.lr,
                      c_warmup_iters=0, batch_size=args.batch_size, device=device, print_loss=False)
        program.save(save_file)
        if args.load_previous_save:
            # Optionally clean the *older* file
            try:
                previous_save.unlink()
            except FileNotFoundError:
                pass
            previous_save = save_file  # keep pointer fresh if you want to delete next time
        print("Saving result at", save_file)
else:
    # Choose which epoch index to evaluate; often the last one:
    epoch_to_eval = args.epochs if args.epochs > 0 else 1
    save_file = ckpt_path(args.lr, epoch_to_eval, args.load_epoch, args.batch_size, args.tnorm, args.subset)
    assert save_file.exists(), f"Missing checkpoint: {save_file} (cwd={Path.cwd()})"
    print("Loading program from checkpoint...", save_file)
    program.load(save_file)
    acc = program.evaluate_condition(dataset, device=device)
    save_results_f= open('results.txt', 'a')
    print(save_file, file=save_results_f)
    print("Epoch: ", args.load_epoch, file=save_results_f)
    print("Learning rate: ", args.lr, file=save_results_f)
    print("Accuracy on Test: {:.2f}".format(acc * 100), file=save_results_f)
    save_results_f.close()