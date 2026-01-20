import sys
from typing import List
import pytest

sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')
import json

# export GRB_LICENSE_FILE=/full/path/to/gurobi.lic
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.sensors import FunctionalSensor, FunctionalReaderSensor, ModuleSensor
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from graph import create_graph
from pathlib import Path
from dataset import g_relational_concepts
import torch

from pathlib import Path

RUN_DIR = Path(__file__).parent.resolve()
MODEL_DIR = RUN_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def filter_relation(property, arg1, arg2):
    # This is default of LEFT framework that perform all pair relation
    return arg1.getAttribute("image_id") == arg2.getAttribute("image_id")


def program_declaration(dataset, device="cpu"):
    results = create_graph(dataset)

    questions_executions = results[0]
    graph = results[1]
    image = results[2]
    objects = results[3]
    image_object_contains = results[4]
    obj1 = results[5]
    obj2 = results[6]
    relaton_2_obj = results[7]
    attribute_names_dict = results[8]

    class DummyLearner(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, input):
            # Input is binary label(s) with values 0 or 1
            # Output must be [prob_negative, prob_positive] per instance
            # Convert: label=1 -> [0, 1], label=0 -> [1, 0]
            output = torch.stack([1 - input, input], dim=-1)
            return output

    image["pil_image"] = FunctionalReaderSensor(keyword="pil_image", forward=lambda data: data)
    image["image_id"] = FunctionalReaderSensor(keyword='image_id', forward=lambda data: [data])

    objects["bounding_boxes"] = FunctionalReaderSensor(keyword="objects_raw",
                                                       forward=lambda data: torch.Tensor(data).to(device))

    objects["image_id"] = FunctionalSensor(image["image_id"], "bounding_boxes",
                                           forward=lambda data, data2: data * len(data2))

    objects[image_object_contains] = EdgeSensor(objects["bounding_boxes"], image["pil_image"],
                                                relation=image_object_contains,
                                                forward=lambda b, _: torch.ones(len(b)).unsqueeze(-1))

    relaton_2_obj[obj1.reversed, obj2.reversed] = CompositionCandidateSensor(
        objects['image_id'],
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
            objects[f"{attr_variable}_label"] = FunctionalReaderSensor(keyword=f"is_{attr_name}",
                                                                       forward=lambda data: torch.Tensor(data).to(
                                                                           device))
            objects[attr_variable] = ModuleLearner(f"{attr_name}_label", module=DummyLearner(attr_name), device=device)

    for i in range(len(dataset)):
        dataset[i]["logic_label"] = torch.LongTensor([int(dataset[i]["logic_label"])]).to(device)
    dataset = graph.compile_executable(dataset, logic_keyword='logic_str', logic_label_keyword='logic_label')

    poi = [image, objects, *attribute_names_dict.values(), graph.constraint, relaton_2_obj]
    program = InferenceProgram(graph, SolverModel,
                               poi=poi,
                               device=device, tnorm="G",
                               inferTypes=["local/argmax"])

    return dataset, attribute_names_dict, program


def check_concept(program, original_dataset, domiknows_dataset, concept_name, concept_attribute, concept_domiknows):
        for i, data in enumerate(program.populate(domiknows_dataset)):
            argmax_results = data.collectInferredResults(concept_domiknows, 'local/argmax')
            domiknows_concept = [int(v[1]) for v in argmax_results.tolist()]
            predicted_concept = [1 if obj[concept_attribute] == concept_name else 0 for obj in original_dataset[i]["scene"]]
            assert domiknows_concept == predicted_concept, f"Expected {concept_name} to be exactly same as ground-truth"

class TestArgmaxConceptInferenceProgram:
    """Tests for single-relation CLEVR program conversion."""

    @pytest.fixture(scope="class")
    def create_program(self):
        with open("convert_CLEVR_program_manual_10_first_translation.json", "rb") as f:
            raw_data = json.load(f)[:]
            dataset = [data["input"] for data in raw_data]

        domiknows_dataset, attribute_names_dict, program = program_declaration(dataset, device="cpu")
        return raw_data, domiknows_dataset, attribute_names_dict, program

    @pytest.mark.parametrize("concept_name", ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"])
    def test_argmax_color_concepts(self, concept_name, create_program):
        raw_data, dataset, attribute_names_dict, program = create_program
        check_concept(program, raw_data, dataset, concept_name, "color", attribute_names_dict[concept_name])

    @pytest.mark.parametrize("concept_name", ["rubber", "metal"])
    def test_argmax_material_concepts(self, concept_name, create_program):
        raw_data, dataset, attribute_names_dict, program = create_program
        check_concept(program, raw_data, dataset, concept_name, "material", attribute_names_dict[concept_name])

    @pytest.mark.parametrize("concept_name", ["cube", "sphere", "cylinder"])
    def test_argmax_shape_concepts(self, concept_name, create_program):
        raw_data, dataset, attribute_names_dict, program = create_program
        check_concept(program, raw_data, dataset, concept_name, "shape", attribute_names_dict[concept_name])

    @pytest.mark.parametrize("concept_name", ["small", "large"])
    def test_argmax_size_concepts(self, concept_name, create_program):
        raw_data, dataset, attribute_names_dict, program = create_program
        check_concept(program, raw_data, dataset, concept_name, "size", attribute_names_dict[concept_name])
