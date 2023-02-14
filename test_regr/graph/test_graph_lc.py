from domiknows.graph import Graph, Concept, Relation, andL, orL, ifL, existsL
import re

class TestGraph(object):
    def test_what(self):
        
        Graph.clear()
        Concept.clear()
        Relation.clear()

        with Graph('global') as graph:
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
                                       
                # if token with concept people and token with concept organization exist then relation work_for also exists
                ifL(andL(existsL(people), existsL(organization)), existsL(work_for))
                
                # Example with variables
                w  = andL(people, organization)
                w2 = orL(w, location)
                w3 = ifL(w2, work_for)

        target = \
        "Graph(name='global', what={'concepts': {},\n 'subs': {'application': Graph(name='application', what={'concepts': {'organization': Concept(name='organization', what={'relations': {'is_a': [IsA(name='organization-is_a-0-phrase', what={'dst': Concept(name='phrase'),\n 'src': Concept(name='organization')})],\n               'not_a': [NotA(name='organization-not_a-1-people', what={'dst': Concept(name='people'),\n 'src': Concept(name='organization')})]},\n 'subs': {},\n 'sup': Graph(name='application')}),\n              'people': Concept(name='people', what={'relations': {'is_a': [IsA(name='people-is_a-0-phrase', what={'dst': Concept(name='phrase'),\n 'src': Concept(name='people')})],\n               'not_a': [NotA(name='people-not_a-1-organization', what={'dst': Concept(name='organization'),\n 'src': Concept(name='people')})]},\n 'subs': {},\n 'sup': Graph(name='application')}),\n              'work_for': Concept(name='work_for', what={'relations': {'has_a': [HasA(name='work_for-has_a-employee-people', what={'dst': Concept(name='people'),\n 'src': Concept(name='work_for')}),\n                         HasA(name='work_for-has_a-employer-organization', what={'dst': Concept(name='organization'),\n 'src': Concept(name='work_for')})],\n               'is_a': [IsA(name='work_for-is_a-0-pair', what={'dst': Concept(name='pair'),\n 'src': Concept(name='work_for')})]},\n 'subs': {},\n 'sup': Graph(name='application')})},\n 'subs': {},\n 'sup': Graph(name='global')}),\n          'linguistic': Graph(name='linguistic', what={'concepts': {'pair': Concept(name='pair', what={'relations': {'has_a': [HasA(name='pair-has_a-0-phrase', what={'dst': Concept(name='phrase'),\n 'src': Concept(name='pair')}),\n                         HasA(name='pair-has_a-1-phrase', what={'dst': Concept(name='phrase'),\n 'src': Concept(name='pair')})]},\n 'subs': {},\n 'sup': Graph(name='linguistic')}),\n              'phrase': Concept(name='phrase', what={'relations': {},\n 'subs': {},\n 'sup': Graph(name='linguistic')})},\n 'subs': {},\n 'sup': Graph(name='global')})},\n 'sup': None})"

        reg_space = re.compile(r'\s+')
        what = re.sub(reg_space, '', repr(graph))
        target = re.sub(reg_space, '', target)
        #assert what == target
        