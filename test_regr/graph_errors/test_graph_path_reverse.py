import pytest
from graph_path_reverse import setup_graph
import re

def test_setup_graph_exception():
    try:
        setup_graph()
    except Exception as e:
        sanitized_error_message = re.sub(r'[^\x20-\x7E]', '', str(e)).replace(" ", "")
        sanitized_pattern = re.sub(r'[^\x20-\x7E]', 
                                   '', 
                                   "The Path 'rel_pair_entity1' from the variable x, defined in LC_person_attendance is not valid"
                                   "The relation rel_pair_entity1 is from a pair to a named_entity, but you have used it from a named_entity to a pair." 
                                   "You can change 'rel_pair_entity1' to 'rel_pair_entity1.reversed' to go from named_entity to the pair, which is what is required here.").replace(" ", "")
        
        ### The Path 'rel_pair_entity1' from the variable x, defined in LC_person_attendance is not valid.
        ### The relation rel_pair_entity1 is from a pair to a named_entity, but you have used it from a named_entity to a pair. 
        ### You can change `rel_pair_entity1` to `rel_pair_entity1.reversed` to go from named_entity to the pair, which is what is required here.
        
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