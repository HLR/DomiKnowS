import pytest
from domiknows.graph import createDummyDataNode, instanceReportOfConstraints
from graph import graph

def test_dummy_data_node_inference():
    testDummyDn = createDummyDataNode(graph)
    
    # run instanceReportOfConstraints
    try:
        instanceReportOfConstraints(testDummyDn)
    except Exception:
        pytest.fail("infer raised an exception")
        
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