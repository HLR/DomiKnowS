import pytest
from graph_nli_3 import setup_graph
import re

def test_setup_graph_exception():
    try:
        setup_graph()
    except Exception as e:
        sanitized_error_message = re.sub(r'[^\x20-\x7E]', '', str(e)).replace(" ", "")

        sanitized_pattern = re.sub(
            r'[^\x20-\x7E]', 
            '', 
            ("The Path rel_pair_premise from the variable x, defined in pair_symmetry_constraint is not valid"
             "and the required destination type of the last element of the path is a pair."
             "The used variable rel_pair_premise is a relationship defined between a pair and a premise, which is not correctly used here.")
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