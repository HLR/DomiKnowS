# Test unit for Logical constraints, IPL solver, Loss calculation and Datanode queries in the DomiKnowS environment

This example uses a [single sentence](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/test_main.py#L14)  from the [conll04](https://www.cs.upc.edu/~srlconll/st04/st04.html) dataset and [precalcualted results](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/test_main.py#L21) to test basic correctness of selected DomiKnowS functionalities.
It also can be used to learn about these functionalities.

The example [domain graph](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/graph.py#L8) is defined including sample of the connll04 [concepts](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/graph.py#L23) and [relations](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/graph.py#L34).

This follows by the definition of [logical constraints](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/graph.py#L32) encoding knowledge about the domain..
The constraint [LC2](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/graph.py#L47) shows how to deal with [nan](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/test_main.py#L77) values in the data set using the embedded *existsL* constraint. https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/graph.py#L47

After that the [model](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/test_main.py#L120) tests sensors for the graph concepts are configured to pupulate the Datanote with the test sample data.
 
This follows by the [test_graph_naming](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/test_main.py#L258) testing correctness of graph definition.

In [test_main_conll04](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/test_main.py#L298) first the Datanode is [obtained](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/test_main.py#L308) from the model build using the defined sensors.

It then runs set of calls to Datadnode [queries](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/test_main.py#L353).

The test for ILP solver and loss calculation follows repeated three times for different [specification](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/test_main.py#L336) of concepts parameters to their calls.

The [ILP solver](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/test_main.py#L346) results as well as [loss calculation](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/test_main.py#L449) results are compared with precalculated results.

The loss calculation are called with three supported [t norms](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/test_main.py#L448).

Finally the [sample logical constraints losses](https://github.com/HLR/DomiKnowS/blob/84c220bc608dcfaeeccc3deae4392ee7393a303a/test_regr/examples/conll04/test_main.py#L472) are computed.

## How To Run
```
!pytest
```
