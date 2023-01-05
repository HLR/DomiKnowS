from regr.graph.concept import EnumConcept
from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import ifL, nandL, orL, notL, andL, atMostL, exactL, fixedL, eqL

Graph.clear()
Concept.clear()
Relation.clear()

num_spans = 2

with Graph(name='global') as graph:
	sentence = Concept(name='sentence')
	word = Concept(name='word')

	sentence_contains, = sentence.contains(word)

	tag_names = ['t_%d' % i for i in range(3)]
	tag = word(name='tag', ConceptClass=EnumConcept, values=tag_names)

	span_names = ['s_%d' % i for i in range(num_spans)]
	span_num = sentence(name='span_num', ConceptClass=EnumConcept, values=span_names)

	spans = []
	span_constraints = []
	for i in range(num_spans):
		sp = word(name='span_%d' % i)

		FIXED = True
		fixedL(sp("x", eqL(word, "spanFixed", {True})), active = FIXED)

		ifL(
			getattr(span_num, span_names[i])('x'),
			ifL(
				sp('y'),
				getattr(tag, tag_names[1])()
				)
			)

		ifL(
			getattr(span_num, span_names[i])('x'),
			ifL(
				sp('y'),
				getattr(tag, tag_names[2])()
				)
			)

		spans.append(sp)
