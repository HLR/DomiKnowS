import torch
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner, TorchLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor, FunctionalReaderSensor
from graph import create_graph
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.program import SolverPOIProgram
from domiknows.program.loss import NBCrossEntropyLoss, BCEWithLogitsIMLoss
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, MetricTracker, CMWithLogitsMetric, DatanodeCMMetric
import pickle
from pathlib import Path
from dataset import make_dataset, default_image_transform
import os.path as osp

MAIN_PATH      = "train"
CACHE_DIR      = Path("dataset_cache")
NUM_INSTANCES  = 10
PICKLE_PROTO   = pickle.HIGHEST_PROTOCOL

CACHE_DIR.mkdir(exist_ok=True)

def build_dataset():
    ds = make_dataset(
        scenes_json      = osp.join(MAIN_PATH, "scenes.json"),
        questions_json   = osp.join(MAIN_PATH, "questions.json"),
        image_root       = osp.join(MAIN_PATH, "images"),
        image_transform  = default_image_transform,
        vocab_json       = osp.join(MAIN_PATH, "vocab.json"),
        output_vocab_json= osp.join(MAIN_PATH, "output-vocab.json"),
        incl_scene       = True,
        incl_raw_scene   = True,
    )
    ds.filter_relational_type()
    return ds

dataset = []
for idx in range(NUM_INSTANCES):
    cache_file = CACHE_DIR / f"instance_{idx}.pkl"

    if cache_file.exists():
        with cache_file.open("rb") as f:
            instance = pickle.load(f)
            dataset.append(instance)
            print(f"re-loaded {cache_file}")
    else:
        ds = build_dataset()
        for idx_ in range(NUM_INSTANCES):
            cache_file = CACHE_DIR / f"instance_{idx_}.pkl"
            with cache_file.open("wb") as f:
                pickle.dump(ds[idx_], f, protocol=PICKLE_PROTO)
                print(f"cached to {cache_file}")

questions_executions, graph,image,object,image_object_contains,attribute_names_dict  = create_graph(dataset,NUM_INSTANCES)

for i in range(len(dataset)):
    dataset[i]["logic_str"] = questions_executions[i]
    dataset[i]["logic_label"] = torch.tensor([1.0])

image["image"]= FunctionalReaderSensor(keyword="image",forward=lambda data: [data])
object["location"]= ReaderSensor(keyword="objects")

def return_contain(b, _):
    return torch.ones(len(b)).unsqueeze(-1)
object[image_object_contains] = EdgeSensor(object["location"], image["image"], relation=image_object_contains, forward=return_contain)


class DummyLinearLearner(TorchLearner):
    def __init__(self, *pre):
        TorchLearner.__init__(self, *pre)

    def forward(self, x):
        result = torch.zeros(len(x), 2)
        result[:, 1] = 1000
        return result

def label_reader(label):
    return torch.ones(len(label)).unsqueeze(-1)

for attr_name,attr_variable in attribute_names_dict.items():
    object[attr_variable] = DummyLinearLearner(image_object_contains)
    #object[attr_variable] = FunctionalSensor(image_object_contains, forward=label_reader, label=True)

dataset = graph.compile_logic(dataset, logic_keyword='logic_str',logic_label_keyword='logic_label',)
program = InferenceProgram(graph,SolverModel,poi=[image,object,*attribute_names_dict.values(), graph.constraint],device="cpu",tnorm='G')
#program = SolverPOIProgram(graph,poi=[image,object,*attribute_names_dict.values(), graph.constraint],device="cpu",inferTypes=['local/argmax'],loss=MacroAverageTracker(NBCrossEntropyLoss()))
program.train(dataset,epochs=10,lr=1e-4,c_warmup_iters=0,device="cpu")
