import pytest
from graph_nli_2 import setup_graph
import re

def test_setup_graph_exception():
    try:
        setup_graph()
    except Exception as e:
        sanitized_error_message = re.sub(r'[^\x20-\x7E]', '', str(e)).replace(" ", "")
        #### fill this, this graph has multiple errors in paths
        sanitized_pattern = re.sub(r'[^\x20-\x7E]', '', 
                                   "LC0 andL constraint: the path x rel_pair_premise is not valid in pair(x, rel_pair_premise,). This is because x is of type pair, and x.rel_pair_premise is of type premise. You should either continue the path to get back to the pair with another reversed option, or use the x variable alone",
                                   ).replace(" ", "")
        
        print(repr(sanitized_error_message))
        print(repr(sanitized_pattern))
        
        assert sanitized_error_message == sanitized_pattern, f"Exception message did not match: got {sanitized_error_message}"
    else:
        pytest.fail("Expected an exception but none was raised.")
        
def test_setup_graph_no_exception():
    try:
        setup_graph(fix_constraint=True)
    except Exception as e:
        pytest.fail(f"Unexpected Exception raised: {e}")