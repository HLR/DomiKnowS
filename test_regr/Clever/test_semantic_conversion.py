import pytest
import json
import torch
from pathlib import Path

from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.sensors import FunctionalSensor, FunctionalReaderSensor
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from .graph import create_graph
from .dataset import g_relational_concepts

device = "cpu"
DATA_DIR = Path(__file__).parent


def filter_relation(property, arg1, arg2):
    return arg1.getAttribute("image_id") == arg2.getAttribute("image_id")


class DummyLearner(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, input):
        return torch.softmax(input, dim=-1)


def run_semantic_conversion(input_file: str):
    """Run semantic conversion pipeline and return accuracy."""
    file_path = DATA_DIR / input_file
    with open(file_path, 'r') as file:
        raw_data = json.load(file)[:]
        dataset = [data["input"] for data in raw_data]

    results = create_graph(dataset)

    graph = results[1]
    image = results[2]
    object = results[3]
    image_object_contains = results[4]
    obj1 = results[5]
    obj2 = results[6]
    relaton_2_obj = results[7]
    attribute_names_dict = results[8]

    for i in range(len(dataset)):
        dataset[i]["logic_label"] = torch.LongTensor([int(dataset[i]["logic_label"])]).to(device)

    image["pil_image"] = FunctionalReaderSensor(keyword="pil_image", forward=lambda data: data)
    image["image_id"] = FunctionalReaderSensor(keyword='image_id', forward=lambda data: [data])

    object["bounding_boxes"] = FunctionalReaderSensor(
        keyword="objects_raw",
        forward=lambda data: torch.Tensor(data).to(device)
    )

    object["image_id"] = FunctionalSensor(
        image["image_id"], "bounding_boxes",
        forward=lambda data, data2: data * len(data2)
    )

    object[image_object_contains] = EdgeSensor(
        object["bounding_boxes"], image["pil_image"],
        relation=image_object_contains,
        forward=lambda b, _: torch.ones(len(b)).unsqueeze(-1)
    )

    relaton_2_obj[obj1.reversed, obj2.reversed] = CompositionCandidateSensor(
        object['image_id'],
        relations=(obj1.reversed, obj2.reversed),
        forward=filter_relation
    )

    spatial_relations = g_relational_concepts.get("spatial_relation", [])

    for attr_name, attr_variable in attribute_names_dict.items():
        if attr_name in spatial_relations:
            relaton_2_obj[f"{attr_variable}_label"] = FunctionalReaderSensor(
                keyword=f"is_{attr_name}",
                forward=lambda data: torch.Tensor(data).to(device)
            )
            relaton_2_obj[attr_variable] = ModuleLearner(
                f"{attr_name}_label", module=DummyLearner(attr_name), device=device
            )
        else:
            object[f"{attr_variable}_label"] = FunctionalReaderSensor(
                keyword=f"is_{attr_name}",
                forward=lambda data: torch.Tensor(data).to(device)
            )
            object[attr_variable] = ModuleLearner(
                f"{attr_name}_label", module=DummyLearner(attr_name), device=device
            )

    dataset = graph.compile_executable(dataset, logic_keyword='logic_str', logic_label_keyword='logic_label')

    poi = [image, object, *attribute_names_dict.values(), graph.constraint, relaton_2_obj]
    program = InferenceProgram(
        graph, SolverModel,
        poi=poi,
        device=device, tnorm="G",
        inferTypes=["argmax"]
    )

    acc = program.evaluate_condition(dataset)
    return acc, program, dataset, raw_data, attribute_names_dict


class TestSemanticConversionTwoRelations:
    """Tests for two-relations CLEVR program conversion."""

    @pytest.fixture(scope="class")
    def conversion_result(self):
        return run_semantic_conversion("convert_CLEVR_program_two_relations.json")

    def test_evaluation_completes(self, conversion_result):
        acc, _, _, _, _ = conversion_result
        assert acc is not None

    def test_accuracy_is_valid(self, conversion_result):
        acc, _, _, _, _ = conversion_result
        if isinstance(acc, (int, float)):
            assert acc == 1.0, f"Expected 100% accuracy, got {acc * 100}%"
        elif isinstance(acc, dict):
            assert acc.get("accuracy", 0) == 1.0, f"Expected 100% accuracy, got {acc}"


class TestSemanticConversionSingleRelation:
    """Tests for single-relation CLEVR program conversion."""

    @pytest.fixture(scope="class")
    def conversion_result(self):
        return run_semantic_conversion("convert_CLEVR_program_manual_10_first_translation.json")

    def test_evaluation_completes(self, conversion_result):
        acc, _, _, _, _ = conversion_result
        assert acc is not None

    def test_accuracy_is_valid(self, conversion_result):
        acc, _, _, _, _ = conversion_result
        if isinstance(acc, (int, float)):
            assert acc == 1.0, f"Expected 100% accuracy, got {acc * 100}%"
        elif isinstance(acc, dict):
            assert acc.get("accuracy", 0) == 1.0, f"Expected 100% accuracy, got {acc}"