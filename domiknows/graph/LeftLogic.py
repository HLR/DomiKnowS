import torch
from domiknows.graph import Graph, Concept
from typing import Literal
from dataclasses import dataclass

# class LearnableUnaryMapping(torch.nn.Module):
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Maps from a input_dim dimensional vector to a output_dim dimensional vector.
#         """
#         raise NotImplementedError()

#     def get_params(self):
#         """
#         Get the parameters of the mapping.
#         """
#         raise NotImplementedError()

# class MLP(LearnableUnaryMapping):
#     def __init__(
#             self,
#             input_dim: int,
#             output_dim: int,
#             hidden_dims: list[int],
#         ):
#         super().__init__()

#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.hidden_dims = hidden_dims

#         self.layers = []
#         prev_dim = input_dim
#         for h_dim in hidden_dims:
#             self.layers.append(torch.nn.Linear(prev_dim, h_dim))
#             self.layers.append(torch.nn.ReLU())
#             prev_dim = h_dim
        
#         self.layers.append(torch.nn.Linear(prev_dim, output_dim))
#         self.layers.append(torch.nn.Softmax(dim=-1))
    
#         self.model = torch.nn.Sequential(*self.layers)

#     def forward(self, x):
#         return self.model(x)

#     def get_params(self):
#         return self.model.parameters()

# def _concept_to_tensor(graph: Graph, concept: Concept) -> torch.Tensor:
#     raise NotImplementedError()

# class LeftLogicElement():
#     def __init__(
#             self,
#             graph: Graph,
#             name: str
#         ):
#         self.graph = graph
#         self.name = name

#         self.mapping: LearnableUnaryMapping = MLP(
#             input_dim,
#             output_dim,
#             [16]
#         )
    
#     def make_name(self, *input_names: str) -> str:
#         return f"{self.name}({','.join(input_names)})"

#     def __call__(self, *input_concepts: Concept) -> Concept:
#         input_tensors = [
#             _concept_to_tensor(self.graph, concept)
#             for concept in input_concepts
#         ]

#         # flatten inputs & merge
#         input_tensor = torch.cat(input_tensors, dim=-1)

# class LeftLogicElement():
#     def __init__(
#             self,
#             graph: Graph,
#             name: str
#         ):
#         """
#         Initialize a learnable operation on concepts in the graph.
#         """

#         self.graph = graph
#         self.name = name

#     def make_output_name(self, *input_names: str) -> str:
#         """
#         Creates the name of the output concept.

#         e.g., sumL(a, b) -> "sumL:a,b"
#         """

#         assert ':' not in self.name
#         assert all(',' not in input_name for input_name in input_names)
#         assert all(':' not in input_name for input_name in input_names)

#         return f"{self.name}:{','.join(input_names)}"
    
#     @staticmethod
#     def parse_output_name(output_name: str) -> tuple[str, list[str]]:
#         """
#         Parses the output name into the operation name and input names.
#         First item in the tuple is the operation name, second item is a list of input concept names.
        
#         e.g., "sumL:a,b" -> "sumL", ["a", "b"]
#         """
#         assert ':' in output_name

#         operation_name, input_names_str = output_name.split(':')
#         input_names = input_names_str.split(',')

#         return operation_name, input_names

#     def __call__(self, *input_concepts: Concept) -> Concept:
#         """
#         For a set of input concepts, create a new concept representing the output of a learnable operation on the inputs.
#         """

#         # check that input concepts are in graph
#         for concept in input_concepts:
#             if concept.name not in self.graph.varNameReversedMap:
#                 raise ValueError(f"LeftLogicElement: Concept {concept.name} not in graph.")

#         # create new concept mapping inputs to outputs
#         input_names = [concept.name for concept in input_concepts]
#         output_name = self.make_output_name(*input_names)

#         output_concept = Concept(name=output_name, input_names=input_names)

#         # register new concept with graph
#         self.graph.varNameReversedMap[output_name] = output_concept

#         return output_concept

# xcon = Concept(name='xcon')
# ycon1 = xcon(name='ycon1')
# ycon2 = xcon(name='ycon2')
# sumL = LeftLogicElement(graph, "sumL")

# problem: need to create a separate concept for each call of e.g., sumL(a, b)
# (b/c they would have separate populated values in the graph)
# but we want to share the same parameters across each sumL

# right now, I just do a dumb string-parsing thing to let you know what operation is being used
# for a given output concept

@dataclass(frozen=True)
class LeftLogicElementOutput():
    """
    LeftLogicElement creates a new concept representing the output of a learnable operation on the inputs.
    LeftLogicElementOutput is a wrapper around that concept to keep some additional metadata.
    """
    input_concepts: list[Concept]
    output_concept: Concept
    operation_name: str

class LeftLogicElement():
    def __init__(
            self,
            graph: Graph,
            name: str
        ):
        """
        Initialize a learnable operation on concepts in the graph.
        """

        self.graph = graph
        self.name = name

    def make_output_name(self, *input_names: str) -> str:
        """
        Creates the name of the output concept.

        e.g., sumL(a, b) -> "sumL(a,b)"
        """

        return f"{self.name}({','.join(input_names)})"

    def __call__(self, *inputs: Concept | LeftLogicElementOutput) -> LeftLogicElementOutput:
        """
        For a set of input concepts, create a new concept representing the output of a learnable operation on the inputs.
        """

        # map list of Concept | LeftLogicElementOutput to list of Concept
        input_concepts = []
        for inp in inputs:
            if isinstance(inp, Concept):
                if inp.name not in self.graph.varNameReversedMap:
                    raise ValueError(f"LeftLogicElement: Concept {inp.name} not in graph.")

                input_concepts.append(inp)

            elif isinstance(inp, LeftLogicElementOutput):
                if inp.output_concept.name not in self.graph.varNameReversedMap:
                    raise ValueError(f"LeftLogicElement: Concept {inp.output_concept.name} not in graph.")

                input_concepts.append(inp.output_concept)

        # create new concept mapping inputs to outputs
        input_names = [concept.name for concept in input_concepts]
        output_name = self.make_output_name(*input_names)

        output_concept = Concept(name=output_name, input_names=input_names)

        # register new concept with graph
        self.graph.varNameReversedMap[output_name] = output_concept

        return LeftLogicElementOutput(
            input_concepts=input_concepts,
            output_concept=output_concept,
            operation_name=self.name
        )
