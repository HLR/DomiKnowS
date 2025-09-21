import pytest
from domiknows.graph import Graph, Concept, Relation


@pytest.fixture(autouse=True)
def clear_domiknows_state():
    """Automatically clear DomiKnows state before each test"""
    Graph.clear()
    Concept.clear()
    Relation.clear()
    yield
    # Cleanup after test if needed
    Graph.clear()
    Concept.clear()
    Relation.clear()