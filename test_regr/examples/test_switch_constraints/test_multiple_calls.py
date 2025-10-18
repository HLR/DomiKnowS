import pytest
import torch
from torch import nn
from typing import Any

from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import orL, existsL, ifL, notL, andL, atMostL, atLeastL, exactL
from domiknows.graph import EnumConcept
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleSensor, FunctionalSensor
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows import setProductionLogMode
setProductionLogMode(no_UseTimeLog=True)

from test_main import setup_graph, dataset


def test_graph_compilation_and_training(setup_graph, dataset):
    """Test graph compilation and model training"""
    graph, root, x, y = setup_graph
    
    # Pass x and y in extra_namespace_values since they're not in the graph's varContext
    transformed_dataset = graph.compile_logic(
        dataset,
        logic_keyword='logic_str',
        logic_label_keyword='logic_label',
        extra_namespace_values={'x': x, 'y': y}
    )

    transformed_dataset_2 = graph.compile_logic(
        dataset,
        logic_keyword='logic_str',
        logic_label_keyword='logic_label',
        extra_namespace_values={'x': x, 'y': y}
    )

    for x, llbl in zip(transformed_dataset, transformed_dataset.lc_name_list):
        print(x.keys(), x['logic_str'], x[f'_constraint_{llbl}'], x['logic_label'])
    
    for x, llbl in zip(transformed_dataset_2, transformed_dataset_2.lc_name_list):
        print(x.keys(), x['logic_str'], x[f'_constraint_{llbl}'], x['logic_label'])

    print(graph.constraint.items())
    
    # TODO: test that the keyword (f'_constraint_{llbl}') goes to the correct logic concept
    
    # assert transformed_dataset is not None
    # assert len(transformed_dataset) == len(dataset)
    
    # program = InferenceProgram(
    #     graph,
    #     SolverModel,
    #     poi=[root, x, y, graph.constraint],
    #     device="cpu",
    #     tnorm='G'
    # )
    
    # program.train(
    #     transformed_dataset,
    #     epochs=2,
    #     lr=1e-4,
    #     c_warmup_iters=0,
    #     device="cpu"
    # )
    
    # assert program is not None

