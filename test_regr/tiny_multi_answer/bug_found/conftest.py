"""Standalone repro suite. No library patches; native solves require opt-in."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def reset_graph_state():
    from domiknows.graph import Graph, Concept, Relation
    from domiknows.graph.dataNode import DataNode
    from domiknows.solver import ilpOntSolverFactory
    from domiknows.utils import setDnSkeletonMode
    Graph.clear(); Concept.clear(); Relation.clear()
    ilpOntSolverFactory.clear()
    DataNode.collectedConceptsAndRelations = None
    setDnSkeletonMode(False)
    yield
    Graph.clear(); Concept.clear(); Relation.clear()
    ilpOntSolverFactory.clear()


@pytest.fixture(scope='session')
def native_license():
    if os.environ.get('DOMIKNOWS_REPRO_NATIVE') != '1':
        pytest.skip('Native solver disabled: configured license check expired; set DOMIKNOWS_REPRO_NATIVE=1 after renewal')
    import gurobipy as gp
    try:
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as model:
            model.addVar(vtype=gp.GRB.BINARY)
            model.optimize()
            assert model.Status == gp.GRB.OPTIMAL
        env.dispose()
    except gp.GurobiError as error:
        reason='expired license' if 'expired' in str(error).lower() else 'license/environment unavailable'
        pytest.skip(f'Native Gurobi blocked: {reason} (code {error.errno}); not a logical test failure')
