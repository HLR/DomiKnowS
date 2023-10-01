import pytest
from graph_nli_1 import setup_graph
import re

def test_setup_graph_exception():
    try:
        setup_graph()
    except Exception as e:
        sanitized_error_message = re.sub(r'[^\x20-\x7E]', '', str(e)).replace(" ", "")

        sanitized_pattern = re.sub(
            r'[^\x20-\x7E]', 
            '', 
            ("The Path 'rel_pair_premise rel_pair_hypothesis.reversed' from the variable rel_pair_premise, defined in pair_symmetry_constraint is not valid."
             "The required source type in this place of the path is a premise,"
             "but the used variable rel_pair_hypothesis.reversed is a relationship defined between a hypothesis and a pair, which is not correctly used here.")
        ).replace(" ", "")
        
        print(sanitized_pattern)

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