The assignment of *concept* *property* (with *sensors* and/or *learners*) is the way to connecting a the *graph* with data (through *sensors*) and/or parameterized computational units (through *learners*).

## Basic usage

The assignment of property is through the `__setitem__` interface of the `Concept` class.
For example, the following code assigns a sensor to the `"label"` property the `people`, which is a `Concept`.
```python
people['label'] = Sensor()
```

## Multiple assignments

A non-trivial semantic we introduced for our language is we allow assigning multiple sensors and learners to one property, while rather than a mean of overriding the variable name, we introduce the multiple assignments as a *consistency constraint*.

For example, the following code assigns a `Learner` and then a `Sensor` to the `"label"` property the `people`, which is a `Concept`.
```python
people['label'] = Learner()
people['label'] = Sensor()
```
The implication is **NOT** that the property has been updated or overridden with the sensor. It means we can get the actual value of `"label"` property with either the `Learner` or the `Sensor`, which should be consistent. However, initially, the output of the `Learner` may not match the output of the `Sensor`. *Learning* of the model should take place here to resolve the inconsistency, i.e. a loss function should be constructed based on the error whenever there is multiple assignments to a property.

It should be noticed that even if the two (or more) assignments are all with sensors, the loss is still considered. A sensor may rely on the output of another learner. The inconsistency error here can help to update those learners indirectly. If there is no underlying learner is still would not hurt because it is just a constant regarding any parameter in the model.
