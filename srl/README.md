# Task background
For this task the model must predict what spans of the input text corresponds to ARG-0 and ARG-1.

Here, the model does this through sequence prediction. At each word, the model outputs 0, 1, 2, corresponding to NONE, ARG-0, or ARG-1.

For example ([example source](https://web.stanford.edu/~jurafsky/slp3/slides/22_SRL.pdf)): `[ARG0 The group] agreed [ARG1 it wouldn’t make an offer]`. The first span is "The group" and the second span is "it wouldn’t make an offer".

The model's prediction would then be:
```
The		1
group		1
agreed		0
it		2
wouldn't	2
make		2
an		2
offer		2
```


# Constraints
Each sentence has a set of valid spans. If a model predicts a span (for either ARG-0 or ARG-1), then that predicted span must be in the set of valid spans.

These aren't the actual spans in practice, but, as an example, if the valid spans are `{"The group", "it wouldn’t make an offer", "The group agreed"}`, then the model can predict any of these three strings without violating constraints but it a prediction of "an offer" would violate constraints.

# Graph

## Model predictions

Model predictions are expressed in the graph as a multiclass classification for each word.

```python
# create prediction variable for each word
predictions = []
for j in range(num_words):
    # the jth tag is predicted as t = {0, 1, 2}
    tag_names = ['pred_%d_%d' % (j, t) for t in range(3)]
    pred = sentence(name='pred_%d' % j, ConceptClass=EnumConcept, values=tag_names)
    predictions.append((pred, tag_names))
```

Here, `pred_j` contains the multiclass prediction (either NONE, ARG-0, or ARG-1) for a single word. The labels for that word would be `pred_j_0, pred_j_1, or pred_j_2` corresponding to NONE, ARG-0, or ARG-1.

In other words, pred_j_0 = 1 if the model predicts that the jth word corresponds to no tag, pred_j_1 = 1 if the model predicts that the jth word corresponds to the ARG-0 tag, pred_j_2 = 1 if the model predicts that the jth word corresponds to the ARG-1 tag.

## Valid spans

The set of valid spans are expressed in the graph in a similar way. As example, with only a single valid span:
```python
single_span = []
for j in range(num_words):
    # in the ith span, the jth tag is supposed to be predicted
    span_tkn = sentence(name='span_%d' % (j,))
    single_span.append(span_tkn)
```

Here, `span_tkn = sentence(name='span_%d' % (j,))` is a binary variable indicating whether the jth word of the sentence is in the span or not. It is 0 if it is not in the span and 1 if it is in the span.

Now, if you have multiple valid spans:

```python
spans = []
for i in range(num_spans):
    single_span = []
    for j in range(num_words):
        # in the ith span, the jth tag is supposed to be predicted
        span_tkn = sentence(name='span_%d_%d' % (i, j))
        single_span.append(span_tkn)
    spans.append(single_span)
```

`sentence(name='span_%d_%d' % (i, j))` is the same variable as before but `i` corresponds to the index of the span. i.e., span_0_1 is the binary variable for the second word in the first span.

Notice that, unlike the model's prediction, the span is a binary variable (instead of a multiclass variable): it only indicates whether a word is in the span or not.

## Enforcing constraints
To check constraints, we iterate through each span and check if the model's prediction *exactly matches* that span. To check if a model's prediction matches a span, we iterate through each word and check if the model makes a prediction at a word (i.e., predicts either ARG-0 or ARG-1) **if and only if that word is in the span**.

Consider again a simple example with a single valid span, and only for ARG-0.

Here, the constraint is that if the model predictions some span for ARG-0, then that span must match the given valid span.

```python
# let `valid_span` be the single valid span for which the model's prediction must match

# check whether the model's prediction matches the span: for each word, check whether the model predicts that word as ARG-0 **if and only if** that word is in the `valid_span`
and_constraints = []
for j in range(num_words):
    # model's prediction at the jth word for ARG-0
    # 1 if the model predicts that the jth word is ARG-0 and 0 otherwise
    pred = getattr(predictions[j][0], predictions[j][1][1])

    # whether the jth word is in the valid span
    # 1 if the jth word is in the valid span and 0 otherwise
    span_val = valid_span[j]

    and_constraints.append(andL(
        # (model predicts ARG-0 at the jth word) -> (the jth word is in the valid span)
        ifL(
            pred('x_1'),
            span_val('y_1')
        ),

        # (the jth word is in the valid span) -> (model predicts ARG-0 at the jth word)
        ifL(
            span_val('y_1'),
            pred('x_1')
        )
    ))

andL(*and_constraints)
```

To handle multiple spans, we add an outer for loop:
```python
# let `spans` be the list of valid spans defined in the previous section

# make sure that the model's prediction matches *at least one* span
or_constraints = []
for i in range(num_spans):
    valid_span = spans[i]

    # check whether the model's prediction matches the span: for each word, check whether the model predicts that word as ARG-0 **if and only if** that word is in the `valid_span`

    and_constraints = []
    for j in range(num_words):
        # model's prediction at the jth word for ARG-0
        # 1 if the model predicts that the jth word is ARG-0 and 0 otherwise
        pred = getattr(predictions[j][0], predictions[j][1][1])

        # whether the jth word is in the valid span
        # 1 if the jth word is in the valid span and 0 otherwise
        span_val = valid_span[j]

        and_constraints.append(andL(
            # (model predicts ARG-0 at the jth word) -> (the jth word is in the valid span)
            ifL(
                pred('x_1'),
                span_val('y_1')
            ),

            # (the jth word is in the valid span) -> (model predicts ARG-0 at the jth word)
            ifL(
                span_val('y_1'),
                pred('x_1')
            )
        ))

    or_constraints.append(andL(*and_constraints))

orL(*or_constraints)
```

Handling both ARG-0 and ARG-1 is similar (see [https://github.com/HLR/DomiKnowS/blob/d636cdaa82b77e9b82be04ac654f46b072d3cca8/srl/graph.py#L43](graph.py) for the specific implementation).
