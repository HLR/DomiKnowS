## 4. Inference

One feature of our framework is an automatic inference based on domain knowledge.
To try this out, the user must first create `Datanode`.

```python
for paragraph_ in program.populate(reader_test):
        paragraph_.inferILPResults(is_more)
        for question_ in paragraph_.getChildDataNodes():
            print(question_.getAttribute(is_more))
            print(question_.getAttribute(is_more, "ILP"))
```
Or you can use the following:

```python
program(inferTypes=["ILP"], ...) 
```

`program.populate` given the reader, will create a datagraph of `Datanode`s and returns a list of "Root" concepts. the "Root" concept here is the `paragraph` concept. each `paragraph` is an instance of `Datanode` class. `paragraph_.inferILPResults(is_more)` tells the datagraph to calculates the "ILP" inference for the property `is_more`.

we can use `getChildDataNodes` method of a `paragraph` to access its questions. each `question` we can access this way, is also a `Datanode` class. one can use the `getAttribute` method of this `Datanode` to access the calculated result for its `is_more` property or as it is shown in the next line of the code, to access this property after "ILP" inference that enforces the constraints. here, unlike in sensors, the questions and their properties are accessed individually. we can use created datagraph here to do inference and calculate the metric with or without "ILP" however we wish.

Please find in a specific topic for more information about [how to query a `Datanode`](developer/QUERY.md) and [how inference works](developer/INFERENCE.md).