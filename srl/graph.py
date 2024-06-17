from domiknows.graph.concept import EnumConcept
from domiknows.graph import Graph, Concept, Relation
from domiknows.graph.logicalConstrain import ifL, nandL, orL, notL, andL, atMostL, exactL, fixedL, eqL

Graph.clear()
Concept.clear()
Relation.clear()

num_spans = 40
num_words = 20

with Graph(name='global') as graph:
	sentence = Concept(name='Sentence')

	# create prediction variable for each word
	predictions = []
	for j in range(num_words):
		# the jth tag is predicted as t = {0, 1, 2}
		tag_names = ['pred_%d_%d' % (j, t) for t in range(3)]
		pred = sentence(name='pred_%d' % j, ConceptClass=EnumConcept, values=tag_names)
		predictions.append((pred, tag_names))

	# create ground truth variable for each word in each valid span
	# variable is 1 if the jth word of the ith span is supposed to be predicted
	# e.g. sentence = The red ball
	# j = 0 1 2
	# if the first span consisted of the first two words, it would be represented as:
	# i = 0 and j = 0: 1
	# i = 0 and j = 1: 1
	# i = 0 and j = 2: 0
	spans = []
	for i in range(num_spans):
		single_span = []
		for j in range(num_words):
			# in the ith span, the jth tag is supposed to be predicted
			span_tkn = sentence(name='span_%d_%d' % (i, j))
			single_span.append(span_tkn)
		spans.append(single_span)

	# iterates over each valid span to see if at least one of them match
	# to check whether a single valid span matches:
	# each word in a sentence is predicted as a tag (i.e., = 1 or 2) iff it is also apart of the span (i.e., = 1)
	or_constraints_1 = []
	or_constraints_2 = []
	for i in range(num_spans):
		and_constraints_1 = []
		and_constraints_2 = []

		for j in range(num_words):
			and_constraints_1.append(andL(
				ifL(spans[i][j]('x_1'), getattr(predictions[j][0], predictions[j][1][1])('y_1')),
				ifL(getattr(predictions[j][0], predictions[j][1][1])('y_1'), spans[i][j]('x_1'))
			))

			and_constraints_2.append(andL(
				ifL(spans[i][j]('x_2'), getattr(predictions[j][0], predictions[j][1][2])('y_2')),
				ifL(getattr(predictions[j][0], predictions[j][1][2])('y_2'), spans[i][j]('x_2'))
			))

		or_constraints_1.append(andL(*and_constraints_1))
		or_constraints_2.append(andL(*and_constraints_2))
	
	orL(*or_constraints_1)
	orL(*or_constraints_2)
