import pytest
import torch
from torch import nn
from typing import Any

from domiknows.graph.dataNode import DataNode, DataNodeBuilder
from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.executable import LogicDataset
from domiknows.graph.logicalConstrain import orL, existsL, ifL, notL, andL, atMostL, atLeastL, exactL
from domiknows.graph import EnumConcept
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.sensor.pytorch import ModuleSensor, FunctionalSensor
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows import setProductionLogMode
setProductionLogMode(no_UseTimeLog=True)

from test_main import setup_graph


def parameterized_dataset(seed: int, size: int):
    """Generate test dataset"""
    rng = torch.Generator()
    rng.manual_seed(seed)

    random_constraints = [
        "andL(x, y)",
        "orL(x, y)",
        "ifL(x, y)",
    ]

    data = [
        {
            'input_vec': torch.randn((1, 2), generator=rng),
            'logic_str': random_constraints[
                torch.randint(len(random_constraints), size=(1,), generator=rng)
            ],
            'logic_label': torch.rand(size=(1,), generator=rng)
        }
        for _ in range(size)
    ]
    
    return data

def test_graph_compilation_and_training(setup_graph, n_datasets: int = 3):
    """Test compiling logic w/ variable number of datasets"""
    graph, root, x, y = setup_graph

    datasets = [
        parameterized_dataset(seed=idx, size=(10 * (idx % 3 + 1)))
        for idx in range(n_datasets)
    ]

    # Compile & transform each of the logic datasets
    transformed_datasets = [
        graph.compile_logic(
            dset,
            logic_keyword='logic_str',
            logic_label_keyword='logic_label',
            extra_namespace_values={'x': x, 'y': y}
        )
        for dset in datasets
    ]

    def _get_rs_keyword(tr_sample: dict[str, Any]) -> str:
        """The ReaderSensor keyword in the transformed dataset (that stores the label)
        changes from sample to sample. This function gets the keyword given a sample
        in the transformed dataset."""

        found_keys = [
            x
            for x in tr_sample.keys()
            if isinstance(x, str) and x.startswith('_constraint')
        ]
        assert len(found_keys) == 1

        return found_keys[0]

    # Basic check of transformed dataset
    for dset, tr_dset in zip(datasets, transformed_datasets, strict=True):
        assert len(dset) == len(tr_dset)

        for orig_sample, tr_sample in zip(dset, tr_dset, strict=True):
            # Make sure that the other values (set by the user) are unchanged
            for k in ['input_vec', 'logic_str', 'logic_label']:
                assert orig_sample[k] is tr_sample[k]
            
            # Make sure that the value given to the ReaderSensor (from key rs_keyword)
            # matches the label given by the user (from key logic_label)
            rs_keyword = _get_rs_keyword(tr_sample)
            assert tr_sample[rs_keyword] is orig_sample['logic_label']

    # Check that the expected number of sensors were added to the graph
    total_num_samples = sum(len(dset) for dset in datasets)
    added_sensors = len(graph.constraint.items())

    assert total_num_samples == added_sensors, 'Number of added sensors must match total number of samples'

    # We expect that graph.constraint has LCs w/ ReaderSensors attached
    # We need to check that the label that gets read by the ReaderSensors
    # match the expected label of the LC

    # Populate expected values (based on the dataset)
    keyword_to_lc_name: dict[str, str] = {} # rs key -> LC{n}
    keyword_to_parent_lc: dict[str, str] = {} # rs key -> {andL, orL, ifL}
    for dset, tr_dset in zip(datasets, transformed_datasets):
        for sample in tr_dset:
            rs_keyword = _get_rs_keyword(sample)

            assert rs_keyword not in keyword_to_lc_name, f'ReaderSensor keyword has already been used: {rs_keyword}'
            assert rs_keyword not in keyword_to_parent_lc, f'ReaderSensor keyword has already been used: {rs_keyword}'

            keyword_to_lc_name[rs_keyword] = sample[LogicDataset.curr_lc_key]
            keyword_to_parent_lc[rs_keyword] = sample['logic_str'].split('(')[0]

    # Test that the sensors added to the graph match the expected values
    for lc, lc_property in graph.constraint.items():
        rs_found = [
            v
            for k, v in lc_property.items()
            if k.startswith('readersensor')
        ]
        assert len(rs_found) == 1, 'Each LC must only have a single ReaderSensor as its property'
        
        rs = rs_found[0]

        assert keyword_to_lc_name[rs.keyword] == str(lc), 'Unexpected LC name'
        assert keyword_to_parent_lc[rs.keyword] == type(lc).__name__, 'Unexpected LC class'

    # Initialize program
    program = InferenceProgram(
        graph,
        SolverModel,
        poi=[root, x, y, graph.constraint],
        device="cpu",
        tnorm='G'
    )

    # Test that populating results in the correct input values, label values, and active LCs
    for i, (dset, tr_dset) in enumerate(zip(datasets, transformed_datasets)):
        for orig_sample, tr_sample in zip(dset, tr_dset, strict=True):
            _, _, dn, dn_builder = program.model(tr_sample)

            assert isinstance(dn, DataNode)
            assert isinstance(dn_builder, DataNodeBuilder)

            # Get the populated input
            found_input_vec = dn.getAttribute('input')
            assert isinstance(found_input_vec, torch.Tensor)

            # Get the constraint datanode and the populated LC label value in it
            constraint_dn_search = dn_builder.findDataNodesInBuilder(select=graph.constraint.name)
            constraint_attr = constraint_dn_search[0].getAttributes() # {'LC{x}/label': tensor(...)}
            
            lbl_key = [
                k
                for k in constraint_attr.keys()
                if k.endswith('/label')
            ][0]
            found_lc_name = lbl_key.split('/')[0] # LC{x}
            found_lc_label = constraint_attr[lbl_key] # tensor(...)
            
            # Get the active LCs
            active_LCs = []
            for lc in graph.logicalConstrains.values():
                if lc.active:
                    active_LCs.append(lc)
            
            assert len(active_LCs) == 1, 'Must have only one active LC'
            found_active_lc = active_LCs[0]

            assert str(found_active_lc) == found_lc_name

            # Check the found values against the original dataset values
            # Check correct input value
            assert torch.allclose(orig_sample['input_vec'].cpu(), found_input_vec.cpu())
            
            # Check correct LC label value
            assert torch.isclose(orig_sample['logic_label'].cpu(), found_lc_label.cpu())
            
            # Check correct active LC (only parent LC class is checked)
            assert orig_sample['logic_str'].split('(')[0] == type(found_active_lc).__name__

    # Test that you can train w/o error
    for i, tr_dset in enumerate(transformed_datasets):
        print(f'Training on dataset #{i + 1}')
        program.train(
            tr_dset,
            epochs=2,
            lr=1e-4,
            c_warmup_iters=0,
            device="cpu"
        )
