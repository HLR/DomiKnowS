"""CLEVR-style object-centered GraphQA executable smoke."""

import functools
from dataclasses import dataclass
import torch

from domiknows.graph import Concept, Graph, Relation
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.sensors import FunctionalReaderSensor, FunctionalSensor, JointReaderSensor


class BinaryVisualPredicate(torch.nn.Module):
    """VLM-compatible binary [no, yes] predicate interface."""

    def __init__(self, size):
        super().__init__()
        self.classifier = torch.nn.Linear(size, 2)

    def forward(self, visual_features):
        return self.classifier(visual_features.float())


class FixedBinaryPredicate(torch.nn.Module):
    def forward(self, labels):
        labels = labels.long().view(-1)
        return torch.stack([(1 - labels).float() * 8, labels.float() * 8], dim=-1)


@dataclass
class SmokeContext:
    graph: Graph
    image: Concept
    obj: Concept
    answer: Concept
    pair: Concept
    van: Concept
    red: Concept
    car: Concept
    right_of: Concept
    pair_src: Relation
    pair_dst: Relation
    object_ids: dict
    modules: dict
    namespace: dict


def build_smoke(device="cpu"):
    Graph.clear()
    Concept.clear()
    Relation.clear()
    from domiknows.graph.dataNode import DataNode, DataNodeBuilder
    from domiknows.solver.ilpOntSolverFactory import ilpOntSolverFactory
    DataNode.clear()
    DataNodeBuilder.clear()
    ilpOntSolverFactory.clear()
    with Graph("graphqa_object_centered_smoke") as graph:
        image = Concept(name="image")
        obj = Concept(name="object")
        image_contains_obj, = image.contains(obj)
        answer = obj(name="answer_object")
        object_ids = {
            value: answer(name=f"answer_{value}") for value in ("o1", "o2", "o3")
        }
        van = obj(name="van")
        red = obj(name="red")
        car = obj(name="car")
        pair = Concept(name="object_pair")
        pair_src, pair_dst = pair.has_a(src_arg=obj, dst_arg=obj)
        right_of = pair(name="right_of")

    image["index"] = FunctionalReaderSensor(
        keyword="image_indices", forward=lambda data: torch.as_tensor(data, device=device)
    )
    obj["index"] = FunctionalReaderSensor(
        keyword="object_indices", forward=lambda data: torch.as_tensor(data, device=device)
    )
    obj["ids"] = FunctionalReaderSensor(keyword="object_ids", forward=lambda data: list(data))
    obj["visual"] = FunctionalReaderSensor(
        keyword="object_visual", forward=lambda data: torch.as_tensor(data, dtype=torch.float32, device=device)
    )
    obj[image_contains_obj] = EdgeSensor(
        obj["index"], image["index"], relation=image_contains_obj,
        forward=lambda x, _y: torch.ones_like(x).unsqueeze(-1),
    )

    modules = {
        name: BinaryVisualPredicate(4).to(device)
        for name in ("van", "red", "car", "right_of")
    }
    obj[van] = ModuleLearner("visual", module=modules["van"], device=device)
    obj[red] = ModuleLearner("visual", module=modules["red"], device=device)
    obj[car] = ModuleLearner("visual", module=modules["car"], device=device)
    for value, concept in object_ids.items():
        key = f"identity_{value}"
        obj[key] = FunctionalSensor(
            obj["ids"],
            forward=lambda ids, expected=value: torch.tensor(
                [str(x) == expected for x in ids], dtype=torch.long, device=device
            ),
        )
        obj[concept] = ModuleLearner(key, module=FixedBinaryPredicate(), device=device)

    pair["index"] = FunctionalReaderSensor(
        keyword="pair_indices", forward=lambda data: torch.as_tensor(data, device=device)
    )
    pair["visual"] = FunctionalReaderSensor(
        keyword="pair_visual", forward=lambda data: torch.as_tensor(data, dtype=torch.float32, device=device)
    )
    pair[pair_src.reversed, pair_dst.reversed] = JointReaderSensor(
        pair["index"], keyword="pair_maps", forward=lambda *_args, data: data
    )
    pair[right_of] = ModuleLearner("visual", module=modules["right_of"], device=device)

    namespace = {
        "answer_object": answer, "van": van, "red": red, "car": car,
        "right_of": right_of, "pair_src": pair_src, "pair_dst": pair_dst,
    }
    return SmokeContext(
        graph, image, obj, answer, pair, van, red, car, right_of,
        pair_src, pair_dst, object_ids, modules, namespace
    )


def smoke_logic():
    return """queryL(
    answer_object,
    iotaL(
        andL(
            van("o"),
            red(path="o"),
            right_of("r", path=("o", pair_src.reversed)),
            car(path=("r", pair_dst))
        )
    )
)"""


def smoke_example(device="cpu"):
    endpoints = [(s, d) for s in range(3) for d in range(3) if s != d]
    src = torch.zeros(len(endpoints), 3, device=device)
    dst = torch.zeros(len(endpoints), 3, device=device)
    for row, (s, d) in enumerate(endpoints):
        src[row, s] = 1
        dst[row, d] = 1
    return {
        "image_indices": torch.tensor([0], device=device),
        "object_indices": torch.arange(3, device=device),
        "object_ids": ["o1", "o2", "o3"],
        "object_visual": torch.tensor(
            [[1., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], device=device
        ),
        "pair_indices": torch.arange(len(endpoints), device=device),
        "pair_visual": torch.tensor(
            [[s == 0, d == 1, s, d] for s, d in endpoints], dtype=torch.float32, device=device
        ),
        "pair_maps": (src, dst),
        "logic_str": smoke_logic(),
        "logic_label": torch.tensor([0], device=device),
    }


def run_smoke(device="cpu", epochs=1, lr=1e-2):
    torch.manual_seed(7)
    ctx = build_smoke(device)
    dataset = ctx.graph.compile_executable(
        [smoke_example(device)], logic_keyword="logic_str",
        logic_label_keyword="logic_label", extra_namespace_values=ctx.namespace,
    )
    program = InferenceProgram(
        ctx.graph, SolverModel,
        poi=[ctx.image, ctx.obj, ctx.answer, ctx.pair, ctx.van, ctx.red, ctx.car,
             ctx.right_of, *ctx.object_ids.values(), ctx.graph.constraint],
        device=device, tnorm="P", inferTypes=["local/argmax"], beta=1.0,
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in program.model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = functools.partial(torch.optim.AdamW, lr=lr)
    program.opt = optimizer(program.model.parameters())
    program.train(
        dataset,
        warmup_epochs=0, constraint_epochs=epochs, device=device, c_lr=lr,
    )
    after = dict(program.model.named_parameters())
    updated = {
        name: not torch.equal(old, after[name].detach())
        for name, old in before.items()
    }
    return {
        "logic": smoke_logic(),
        "updated": updated,
        "updated_count": sum(updated.values()),
        "trainable_count": len(updated),
        "all_updated": bool(updated) and all(updated.values()),
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_smoke(device=device)
    print(result["logic"])
    print(f"updated={result['updated']}")
    print(f"all_updated={result['all_updated']}")
