# Test unit for Logical constraints, IPL solver, Loss calculation and Datanode queries in the DomiKnowS environment

This example uses conll04 sample and precalcualted results to test basic correctness of selected DomiKnowS functionalities.
It also can be used to learn about these functionalities.

The example **domain graph** is defined including sample of the connll04 concepts and relations.

This follows by the definition of few sample **logical constraints**.
The constraint *LC2* shows how to deal with *nan* values in the data set using the embedded *existsL* constraint.

The sample for the example are defined in the *case* structure.

After that the model sensors and **learner** for the graph concepts are configured from the sets of sensors available in the DomiKnowS environment.
 
This follows by the *test_graph_naming* testing correctness of graph definition.

In *test_main_conll04* first the Datanode is obtained from the model build using the defined sensors.

It then runs set of calls to Datadnode queries.

The test for ILP solver and loss calculation follows repeated three times for different specification of concepts parameters to their calls.

The ILP solver results as well as loss calculation results are compared with precalculated results.

The loss calculation are called with three supported t norms.

Finally the sample logical constraints losses are computed.

## How To Run
```
!pytest
```
