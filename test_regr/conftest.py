import pytest
from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.dataNode import DataNode, DataNodeBuilder


@pytest.fixture(autouse=True)
def clear_domiknows_state():
    """Automatically clear DomiKnows state before each test"""
    Graph.clear()
    Concept.clear()
    Relation.clear()
    DataNode.clear()
    yield
    # Cleanup after test if needed
    Graph.clear()
    Concept.clear()
    Relation.clear()
    DataNode.clear()
