import pytest
from graph_variable_type import setup_graph
import re

def test_setup_graph_exception():
    try:
        setup_graph()
    except Exception as e:
        sanitized_error_message = re.sub(r'[^\x20-\x7E]', '', str(e)).replace(" ", "")

        sanitized_pattern = re.sub(
            r'[^\x20-\x7E]', 
            '', 
            ("In constraint testLC, the variable x is defined to have the type accident_details but is later reused in weather_condition.__getattr__('clear')(path=('x')) which is incorrect!"
             "The variable used inside the path should match its type with weather_details.")
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