import pytest

import sys
import os

# Add the current directory to Python path to find graph_multi module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domiknows.graph import createDummyDataNode, satisfactionReportOfConstraints
from graph import graph

def test_dummy_data_node_inference():
    testDummyDn = createDummyDataNode(graph)
    
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