import pytest
from graph_cardinality import setup_graph
import re

def test_setup_graph_exception():
    try:
        setup_graph()
    except Exception as e:
        sanitized_error_message = re.sub(r'[^\x20-\x7E]', '', str(e)).replace(" ", "")
        sanitized_pattern = re.sub(r'[^\x20-\x7E]', '', 
                                   "Logical Constraint constraint_only_one_entity has incorrect cardinality definition in nested constraint LC0 - "
                                   "integer 1 has to be last element in the Logical Constraint").replace(" ", "")
        
        print(repr(sanitized_error_message))
        print(repr(sanitized_pattern))

        ### fixed with ChatGPT
        ##### Logical Constraint constraint_only_one_entity has incorrect cardinality definition in nested atMostL logical operator - integer 1 has to be last element in the same Logical operator for counting or existing logical operators!
        
        assert sanitized_error_message == sanitized_pattern, f"Exception message did not match: got {sanitized_error_message}"
    else:
        pytest.fail("Expected an exception but none was raised.")
        
def test_setup_graph_no_exception():
    try:
        setup_graph(fix_constraint=True)
    except Exception as e:
        pytest.fail(f"Unexpected Exception raised: {e}")