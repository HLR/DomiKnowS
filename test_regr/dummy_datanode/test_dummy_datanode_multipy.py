import pytest
from domiknows.graph import createDummyDataNode, satisfactionReportOfConstraints
from graph_multi import graph_multi

def test_dummy_data_node_inference():
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