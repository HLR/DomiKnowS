from domiknows.graph.concept import EnumConcept
from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, nandL, orL, notL, andL, atMostL, exactL, fixedL, eqL

Graph.clear()
Concept.clear()
Relation.clear()

num_spans = 35

with Graph(name='global') as graph:
	sentence = Concept(name='sentence')
	word = Concept(name='word')

	sentence_contains, = sentence.contains(word)

	# predicted tag for each word
	tag_names = ['t_%d' % i for i in range(3)]
	tag = word(name='tag', ConceptClass=EnumConcept, values=tag_names)

	span_names = ['s_%d' % i for i in range(num_spans)]

	# enforce that only one span is correct for each argument
	span_num_1 = word(name='span_num_1', ConceptClass=EnumConcept, values=span_names)
	span_num_2 = word(name='span_num_2', ConceptClass=EnumConcept, values=span_names)

	spans = []
	span_constraints = []
	for i in range(num_spans):
		# binary mask for span i and some arbitrary word
		sp = word(name='span_%d' % i)

		#FIXED = True
		#fixedL(sp("x", eqL(word, "spanFixed", {True})), active = FIXED)

		# if the ith span is correct, enforce that it gets predicted
		ifL(
			getattr(span_num_1, span_names[i])('x'),
			ifL(
				sp('y'),
				getattr(tag, tag_names[1])()
				)
			)

		ifL(
			getattr(span_num_2, span_names[i])('x'),
			ifL(
				sp('y'),
				getattr(tag, tag_names[2])()
			)
		)

		spans.append((sp, span_num_1, span_num_2))
