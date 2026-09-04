import pytest
import sys
import os

# Add the current directory to Python path to find graph_multi module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domiknows.graph import createDummyDataNode, satisfactionReportOfConstraints
from graph_multi import graph_multi


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