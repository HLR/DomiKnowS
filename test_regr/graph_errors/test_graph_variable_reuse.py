import pytest
from graph_variable_resue import setup_graph
import re

def test_setup_graph_exception():
    try:
        setup_graph()
    except Exception as e:
        sanitized_error_message = re.sub(r'[^\x20-\x7E]', '', str(e)).replace(" ", "")
        sanitized_pattern = re.sub(r'[^\x20-\x7E]', '', 
                                   "For constraint constraint_only_one_entity_fixed: You have used the variable x, which is already defined in `organization('x')` which is not allowed. Fix this by:\n"
                                   "- If you meant that the same variable 'x' defined before should be of type organization: organization(path=('x'))"
                                   "- If you meant another instance 'y' should be of type organization: organization('y')").replace(" ", "")
        
        ### you have used the notion person(path=('x', entity)) which is incorrect.
        ### entity is a concept and cannot be used as part of the path. 
        ### - If you meant that 'x' should be of type person: person(path=('x'))
        ### - if you meant another entity 'y' should be of type person which is somehow related to 'x': person(path=('x', edge1, edge2, ...)) 
        ###   where edge1, edge2, ... are relations that connect 'x' to 'y'
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