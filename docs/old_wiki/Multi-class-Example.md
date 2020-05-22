Declare the concepts for multi-class:
```python
phrase = Concept('phrase')
(people, organization, location, other) = phrase(multiclass=['people', 'organization', 'location', 'other'])
work_for = Relation(people, organization)
```

Declare sensors (and learners) for multi-class:
```
phrase[(people, organization, location, other)] = LabelReaderSensor(reader, 'phrase_type')
phrase[(people, organization, location, other)] = SoftmaxLearner('embedding')
# equivalent to
phrase_types = (people, organization, location, other)
phrase[phrase_types] = LabelReaderSensor(reader, 'phrase_type')
phrase[phrase_types] = SoftmaxLearner('embedding')
```

Two problems:
* Do we need to declare the `none` class? In Conll04, `other`s are "other entities" and there is `none` for "non-entities".
* What if the sensor declaration, the learner declaration, and concept declaration do not match?
