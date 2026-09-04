import pytest
import os
from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.dataNode import DataNode, DataNodeBuilder
from domiknows.solver.ilpOntSolverFactory import ilpOntSolverFactory


def pytest_addoption(parser):
    parser.addoption(
        "--use-subprocess",
        action="store_true",
        default=False,
        help="Use subprocess for running tests instead of direct calls"
    )


def pytest_configure(config):
    # Set environment variable based on config
    if config.getoption("--use-subprocess"):
        os.environ['USE_SUBPROCESS'] = 'true'
    else:
        os.environ.setdefault('USE_SUBPROCESS', 'false')


@pytest.fixture(autouse=True)
def clear_domiknows_state():
    """Automatically clear DomiKnows state before each test"""
    Graph.clear()
    Concept.clear()
    Relation.clear()
    DataNode.clear()
    DataNodeBuilder.clear()
    ilpOntSolverFactory.clear()
    yield
    # Cleanup after test if needed
    Graph.clear()
    Concept.clear()
    Relation.clear()
    DataNode.clear()
    DataNodeBuilder.clear()
    ilpOntSolverFactory.clear()