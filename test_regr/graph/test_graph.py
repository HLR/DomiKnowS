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

    @pytest.fixture()
    def target(self):
        target = \
        "Graph(name='global', what={'concepts': {},\n 'subs': {'application': Graph(name='application', what={'concepts': {'organization': Concept(name='organization', what={'relations': {'is_a': [IsA(name='organization-is_a-0-phrase', what={'dst': Concept(name='phrase'),\n 'src': Concept(name='organization')})],\n               'not_a': [NotA(name='organization-not_a-1-people', what={'dst': Concept(name='people'),\n 'src': Concept(name='organization')})]},\n 'subs': {},\n 'sup': Graph(name='application')}),\n              'people': Concept(name='people', what={'relations': {'is_a': [IsA(name='people-is_a-0-phrase', what={'dst': Concept(name='phrase'),\n 'src': Concept(name='people')})],\n               'not_a': [NotA(name='people-not_a-1-organization', what={'dst': Concept(name='organization'),\n 'src': Concept(name='people')})]},\n 'subs': {},\n 'sup': Graph(name='application')}),\n              'work_for': Concept(name='work_for', what={'relations': {'has_a': [HasA(name='work_for-has_a-employee-people', what={'dst': Concept(name='people'),\n 'src': Concept(name='work_for')}),\n                         HasA(name='work_for-has_a-employer-organization', what={'dst': Concept(name='organization'),\n 'src': Concept(name='work_for')})],\n               'is_a': [IsA(name='work_for-is_a-0-pair', what={'dst': Concept(name='pair'),\n 'src': Concept(name='work_for')})]},\n 'subs': {},\n 'sup': Graph(name='application')})},\n 'subs': {},\n 'sup': Graph(name='global')}),\n          'linguistic': Graph(name='linguistic', what={'concepts': {'pair': Concept(name='pair', what={'relations': {'has_a': [HasA(name='pair-has_a-0-phrase', what={'dst': Concept(name='phrase'),\n 'src': Concept(name='pair')}),\n                         HasA(name='pair-has_a-1-phrase', what={'dst': Concept(name='phrase'),\n 'src': Concept(name='pair')})]},\n 'subs': {},\n 'sup': Graph(name='linguistic')}),\n              'phrase': Concept(name='phrase', what={'relations': {},\n 'subs': {},\n 'sup': Graph(name='linguistic')})},\n 'subs': {},\n 'sup': Graph(name='global')})},\n 'sup': None})"
        return target

    def test_what(self, graph, target):
        reg_space = re.compile(r'\s+')
        what = re.sub(reg_space, '', repr(graph))
        target = re.sub(reg_space, '', target)
        assert what == target

    def test_what_new(self, graph_new, target):
        reg_space = re.compile(r'\s+')
        what = re.sub(reg_space, '', repr(graph_new))
        target = re.sub(reg_space, '', target)
        assert what == target
