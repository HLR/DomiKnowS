import pytest
from itertools import combinations
import numpy as np

@pytest.fixture()
def emr_graph(request):
    from regr.graph import Graph, Concept, Relation

    Graph.clear()
    Concept.clear()
    Relation.clear()

    with Graph('global') as graph:
        graph.ontology = ('http://ontology.ihmc.us/ML/EMR.owl', './examples/emr')

        with Graph('linguistic') as ling_graph:
            word = Concept(name='word')
            phrase = Concept(name='phrase')
            sentence = Concept(name='sentence')
            phrase.has_many(word)
            sentence.has_many(phrase)

            pair = Concept(name='pair')
            pair.has_a(phrase, phrase)

        with Graph('application') as app_graph:
            entity = Concept(name='entity')
            entity.is_a(phrase)

            people = Concept(name='people')
            organization = Concept(name='organization')
            location = Concept(name='location')
            other = Concept(name='other')
            o = Concept(name='O')

            people.is_a(entity)
            organization.is_a(entity)
            location.is_a(entity)
            other.is_a(entity)
            o.is_a(entity)

            people.not_a(organization)
            people.not_a(location)
            people.not_a(other)
            people.not_a(o)
            organization.not_a(people)
            organization.not_a(location)
            organization.not_a(other)
            organization.not_a(o)
            location.not_a(people)
            location.not_a(organization)
            location.not_a(other)
            location.not_a(o)
            other.not_a(people)
            other.not_a(organization)
            other.not_a(location)
            other.not_a(o)
            o.not_a(people)
            o.not_a(organization)
            o.not_a(location)
            o.not_a(other)

            work_for = Concept(name='work_for')
            work_for.is_a(pair)
            work_for.has_a(people, organization)

            located_in = Concept(name='located_in')
            located_in.is_a(pair)
            located_in.has_a(location, location)

            live_in = Concept(name='live_in')
            live_in.is_a(pair)
            live_in.has_a(people, location)

            orgbase_on = Concept(name='orgbase_on')
            orgbase_on.is_a(pair)
            orgbase_on.has_a(organization, location)

            kill = Concept(name='kill')
            kill.is_a(pair)
            kill.has_a(people, people)

    yield graph

    #------------------
    # tear down
    #------------------
    Graph.clear()
    Concept.clear()
    Relation.clear()


@pytest.fixture(params=range(1, 20))
def emr_input(emr_graph, request):
    import numpy as np
    import random

    length = request.param

    app_graph = emr_graph["application"]

    entities = [app_graph["people"],
                app_graph["organization"],
                app_graph["other"],
                app_graph["location"],
                app_graph["O"]]
    

    relations = [app_graph["work_for"],
                 app_graph["live_in"],
                 app_graph["located_in"]]

    tokens = [("John", "NNP"), ("works", "VBN"), ("for", "IN"), ("IBM", "NNP")]
    phrase = [random.choice(tokens) for _ in range(length)]
    phrase = [('{}_{}'.format(idx, token), pos_tag)
              for idx, (token, pos_tag) in enumerate(phrase)]

    entities_input = {entity : np.random.rand(length)
                       for entity in entities}
    eye_cut = 0
    relations_input = {relation : np.random.rand(length, length) * (1 - np.eye(length, length) * eye_cut)
                        for relation in relations}

    return phrase, entities_input, relations_input


def passby(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def mini_wrap(emr_graph, phrase, *inputs, benchmark=passby):
    # prepare solver
    from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
    import logging
    ilpConfig = {
        'ilpSolver' : 'mini_prob_debug',
        'log_level' : logging.DEBUG,
        'log_filename' : 'ilpOntSolver.log',
        'log_filesize' : 5*1024*1024*1024,
        'log_backupCount' : 5,
        'log_fileMode' : 'a'
    }
    solver = ilpOntSolverFactory.getOntSolverInstance(emr_graph, _ilpConfig=ilpConfig, lazy_not=True, self_relation=False)
    
    # call solver
    results = benchmark(solver.solve_legacy, phrase, *inputs)
    return results


def owl_wrap(emr_graph, phrase, *inputs, benchmark=passby):
    if len(inputs) > 3:
        raise NotImplementedError
    # prepare input
    owl_inputs = []
    key_maps = []
    for input_ in inputs:
        owl_input = {}
        key_map = {}
        for k, v in input_.items():
            owl_input[k.name] = v
            key_map[k.name] = k
        owl_inputs.append(owl_input)
        key_maps.append(key_map)

    # prepare solver
    from regr.solver.ilpOntSolverFactory import ilpOntSolverFactory
    solver = ilpOntSolverFactory.getOntSolverInstance(emr_graph)

    # call solver
    owl_results = benchmark(solver.calculateILPSelection, phrase, *owl_inputs)

    # prepare result
    results = [{key_map[k]:v for k, v in owl_result.items()}
               for owl_result, key_map in zip(owl_results, key_maps)]
    return results


solver_list = [mini_wrap, owl_wrap]


@pytest.fixture(params=solver_list)
def solver(request):
    return request.param


@pytest.fixture(params=combinations(solver_list, 2))
def solvers(request):
    return request.param


def objective(emr_input, results, lazy_not=True):
    _, *inputs = emr_input
    obj = 0
    for input_, result in zip(inputs, results):
        for key in result:
            obj += (result[key] * input_[key]).sum()
            if lazy_not:
                obj += ((1 - result[key]) * (1 - input_[key])).sum()
    return obj


@pytest.mark.slow
@pytest.mark.gurobi
def test_compare_emr(emr_graph, emr_input, solvers):
    objs = []
    for solver in solvers:
        objs.append(solver(emr_graph, *emr_input))
    # compare each two
    for a, b in combinations(objs, 2):
        # same number of arities
        assert len(a) == len(b)
        for aa, bb in zip(a, b):
            # same number of predicates
            assert len(aa) == len(bb)
            for key in aa:
                # just all the same
                assert (aa[key] == bb[key]).all()


@pytest.mark.slow
@pytest.mark.gurobi
@pytest.mark.benchmark
def test_benchmark(emr_graph, emr_input, solver, benchmark):
    solver(emr_graph, *emr_input, benchmark=benchmark)
