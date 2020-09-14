from regr.graph import Graph, Concept, Relation
import re
import pytest


class TestGraph(object):
    @pytest.fixture()
    def graph(self):
        Graph.clear()
        Concept.clear()
        Relation.clear()

        with Graph('global') as graph:
            with Graph('linguistic') as ling_graph:
                phrase = Concept(name='phrase')
                pair = Concept(name='pair')
                pair.has_a(phrase, phrase)
            with Graph('application') as app_graph:
                people = Concept(name='people')
                organization = Concept(name='organization')
                people.is_a(phrase)
                organization.is_a(phrase)
                people.not_a(organization)
                organization.not_a(people)
                work_for = Concept(name='work_for')
                work_for.is_a(pair)
                work_for.has_a(employee=people, employer=organization)
        yield graph

        Graph.clear()
        Concept.clear()
        Relation.clear()

    @pytest.fixture()
    def graph_new(self):
        Graph.clear()
        Concept.clear()
        Relation.clear()

        with Graph('global') as graph:
            with Graph('linguistic') as ling_graph:
                phrase = Concept('phrase')
                pair = Concept('pair')
                pair.has_a(phrase, phrase)
            with Graph('application') as app_graph:
                people = phrase('people')
                organization = phrase('organization')
                people.not_a(organization)
                organization.not_a(people)
                work_for = Concept(name='work_for')
                work_for.is_a(pair)
                work_for.has_a(employee=people, employer=organization)
        yield graph

        Graph.clear()
        Concept.clear()
        Relation.clear()

    def target(self, graph):
        target = {
            'concepts': {},
            'subs': {
                'linguistic': graph['linguistic'],
                'application': graph['application'],
                },
            'sup': None,
        }
        return target

    def test_what(self, graph):
        what = graph.what()
        assert what == self.target(graph)

    def test_what_new(self, graph_new):
        what = graph_new.what()
        assert what == self.target(graph_new)

    def test_repr(self, graph):
        assert repr(graph) == 'Graph(name=\'global\', fullname=\'global\')'

    def test_repr_new(self, graph_new):
        assert repr(graph_new) == 'Graph(name=\'global\', fullname=\'global\')'
