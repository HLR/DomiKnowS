from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import orL, andL, existsL, notL, atLeastL, atMostL

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    thread = Concept(name='thread')
    email = Concept(name='email')
    (thread_contains_email,) = thread.contains(email)

    forwarded_by = Concept(name='forwarded_by')
    (main_email, forwarded_email) = forwarded_by.has_a(arg1=email, arg2=email)

    Spam = email(name='spam')

