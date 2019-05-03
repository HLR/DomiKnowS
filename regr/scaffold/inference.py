from .. import Graph
from typing import Dict
from torch import Tensor


DataInstance = Dict[str, Tensor]


def inference(
    graph: Graph,
    data: DataInstance
) -> DataInstance:
    return data
    