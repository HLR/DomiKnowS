'''
Task: Semantic Role Labeling

e.g., John kicked the ball. Given predicate "kicked" predict ARG0="John" and ARG1="the ball".

The SRL model predicts contiguous spans of text (e.g., "the ball").

We have a set of *valid* spans that we want to constrain our model to.
e.g., The valid spans of "John kicked the ball" => {"John", "kicked", "ball", "the ball", ...}

We want to constrain the spans predicted by the model to be within that set of valid spans.
'''

# Input to predict on (e.g., "John kicked the ball")
sentence = Concept(name='sentence')

# Some contiguous span of text (e.g., "the ball")
span = Concept(name='span')

# Set of spans (e.g., {"John", "kicked", "ball", "the ball", ...})
span_set = Concept(name='span_set')
span_set.contains(span)

# Set of spans that we want to constrain our model to
valid_spans = span_set(name='valid_spans')

# Predicted arguments in the sentence (e.g., arg0 = "John" and arg1 = "the ball")
# Arguments are predicted by the model, for example, with a sequence of tag probabilities
argument = span(name='argument')
arg0 = argument(name='arg0')
arg1 = argument(name='arg1')

# Set inclusion constraint
# Each argument has to be within the set of valid spans
is_in(arg0, valid_spans)
is_in(arg1, valid_spans)
