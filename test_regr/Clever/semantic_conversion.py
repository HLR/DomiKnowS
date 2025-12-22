import sys

sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')
import json

# export GRB_LICENSE_FILE=/full/path/to/gurobi.lic
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor, FunctionalReaderSensor, ModuleSensor
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from preprocess import preprocess_dataset, preprocess_folders_and_files
from graph import create_graph
from pathlib import Path
from modules import LEFTObjectEMB, LEFTRelationEMB, ResnetLEFT, LinearLayer
from dataset import g_relational_concepts
import argparse, torch, logging

from pathlib import Path

RUN_DIR = Path(__file__).parent.resolve()
MODEL_DIR = RUN_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def ckpt_path(lr, epoch_idx, load_epoch_tag, batch, tnorm, subset):
    return MODEL_DIR / f"program{lr}_{epoch_idx}_{load_epoch_tag}__{batch}_6000_{tnorm}_{subset}.pth"


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Dummy Run of CLEVR")
parser.add_argument("--logic_str", type=str, default="")
args = parser.parse_args()

device = "cpu"


def filter_relation(property, arg1, arg2):
    # This is default of LEFT framework that perform all pair relation
    return arg1.getAttribute("image_id") == arg2.getAttribute("image_id")


with open("convert_CLEVR_program_manual_10_first_translation.json", 'rb') as file:
    raw_data = json.load(file)[:]
    dataset = [data["input"] for data in raw_data]

results = create_graph(dataset)

questions_executions = results[0]
graph = results[1]
image = results[2]
object = results[3]
image_object_contains = results[4]
obj1 = results[5]
obj2 = results[6]
relaton_2_obj = results[7]
attribute_names_dict = results[8]
query_types = results[9] if len(results) > 9 else [None] * len(dataset)

for i in range(len(dataset)):
    dataset[i]["logic_label"] = torch.LongTensor([int(dataset[i]["logic_label"])]).to(device)


class DummyLearner(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, input):
        output = input
        # if self.name == "front":
        #     output_test = output.tolist()
        #     for i in range(6):
        #         for j in range(6):
        #             print(i, j, output_test[i * 6 + j])
        return torch.softmax(output, dim=-1)


image["pil_image"] = FunctionalReaderSensor(keyword="pil_image", forward=lambda data: data)
image["image_id"] = FunctionalReaderSensor(keyword='image_id', forward=lambda data: [data])

object["bounding_boxes"] = FunctionalReaderSensor(keyword="objects_raw",
                                                  forward=lambda data: torch.Tensor(data).to(device))

object["image_id"] = FunctionalSensor(image["image_id"], "bounding_boxes",
                                      forward=lambda data, data2: data * len(data2))

object[image_object_contains] = EdgeSensor(object["bounding_boxes"], image["pil_image"],
                                           relation=image_object_contains,
                                           forward=lambda b, _: torch.ones(len(b)).unsqueeze(-1))

relaton_2_obj[obj1.reversed, obj2.reversed] = CompositionCandidateSensor(
    object['image_id'],
    relations=(obj1.reversed, obj2.reversed),
    forward=filter_relation)

spatial_relations = g_relational_concepts.get("spatial_relation", [])

for attr_name, attr_variable in attribute_names_dict.items():
    if attr_name in spatial_relations:
        # scene, box, object_features
        relaton_2_obj[f"{attr_variable}_label"] = FunctionalReaderSensor(keyword=f"is_{attr_name}",
                                                                         forward=lambda data: torch.Tensor(data).to(
                                                                             device))
        relaton_2_obj[attr_variable] = ModuleLearner(f"{attr_name}_label", module=DummyLearner(attr_name),
                                                     device=device)
    else:
        object[f"{attr_variable}_label"] = FunctionalReaderSensor(keyword=f"is_{attr_name}",
                                                                  forward=lambda data: torch.Tensor(data).to(device))
        object[attr_variable] = ModuleLearner(f"{attr_name}_label", module=DummyLearner(attr_name), device=device)

dataset = graph.compile_logic(dataset, logic_keyword='logic_str', logic_label_keyword='logic_label')

poi = [image, object, *attribute_names_dict.values(), graph.constraint, relaton_2_obj]
program = InferenceProgram(graph, SolverModel,
                           poi=poi,
                           device=device, tnorm="G",
                           inferTypes=["argmax"])

acc = program.evaluate_condition(dataset)
print(acc)

print("-" * 100)

print("Manually Check Concept")

brown = attribute_names_dict["brown"]

cylinder = attribute_names_dict["cylinder"]

index = 0
for data in program.populate(dataset):
    brown_objs = [int(child.getAttribute(brown, 'argmax')) for child in data.getChildDataNodes()]
    print("expected brown objects", [1 if obj["color"] == "brown" else 0 for obj in raw_data[index]["scene"]])
    print("brown objects:", brown_objs)
    cylinder_objs = [int(child.getAttribute(cylinder, 'argmax')) for child in data.getChildDataNodes()]
    print("cylinder objects:", cylinder_objs)
    print("expected cylinder objects", [1 if obj["shape"] == "cylinder" else 0 for obj in raw_data[index]["scene"]])
    index += 1

    print("-" * 10)
print(acc)
