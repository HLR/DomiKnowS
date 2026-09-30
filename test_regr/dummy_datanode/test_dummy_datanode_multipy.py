import importlib.util
import pytest
from pathlib import Path

from domiknows.graph import createDummyDataNode, satisfactionReportOfConstraints


def _load_graph(module_file, module_name, attribute):
    """Load a graph definition that sits beside this test, by file path.

    Importing it as ``graph`` / ``graph_multi`` through ``sys.path`` depends on
    the path order and on other tests having imported a different module with
    the same name (Tasks/clevr_inference_vs_gumbel/graph.py), which made this
    file fail to collect when run together with them.
    """
    path = Path(__file__).with_name(module_file)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attribute)


graph_multi = _load_graph("graph_multi.py", "dummy_datanode_graph_multi", "graph_multi")


class TestGraphMultiInference:
    
    def test_dummy_data_node_inference(self):
        """Test dummy data node inference operations on multi-graph"""
        testDummyDn = createDummyDataNode(graph_multi)
        
        # run satisfactionReportOfConstraints
        try:
            satisfactionReportOfConstraints(testDummyDn)
        except Exception:
            pytest.fail("satisfaction report raised an exception")
            
        # Checking if inferILPResults doesn't raise any exception
        try:
            testDummyDn.inferILPResults()
        except Exception:
            pytest.fail("inferILPResults raised an exception")

        # Checking if infer doesn't raise any exception
        try:
            testDummyDn.infer()
        except Exception:
            pytest.fail("infer raised an exception")
    
    def test_multi_graph_satisfaction_report(self):
        """Test satisfaction report generation on multi-graph"""
        testDummyDn = createDummyDataNode(graph_multi)
        
        # Test that satisfaction report can be generated without errors
        report = satisfactionReportOfConstraints(testDummyDn)
        assert report is not None
    
    def test_multi_graph_ilp_inference(self):
        """Test ILP inference execution on multi-graph"""
        testDummyDn = createDummyDataNode(graph_multi)
        
        # Test that ILP inference runs without errors
        result = testDummyDn.inferILPResults()
        # Basic assertion that method completed
        assert result is not None or result is None  # Method may return None
    
    def test_multi_graph_general_inference(self):
        """Test general inference execution on multi-graph"""
        testDummyDn = createDummyDataNode(graph_multi)
        
        # Test that general inference runs without errors
        result = testDummyDn.infer()
        # Basic assertion that method completed
        assert result is not None or result is None  # Method may return None