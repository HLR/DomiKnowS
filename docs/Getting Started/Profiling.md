## 6. Profiling the execution


#### Production Mode
When the learning model has been positively tested and is ready for the production execution there is a method which optimize the model time performance:

`setProductionLogMode(no_UseTimeLog=True)`

The method should be called in the program before the learning model is started.

The method will disable most of logging in the program (except single log file tracking time of basic steps in model). If the option *no_UseTimeLog* is set to true then all logging is disable.
Additionally, the method enables reuse of the ILP constraints model in the ILP solver. This solution attempts  previously constructed ILP model if the current example matches the examples from the previous model iteration. There is an option in the *setProductionLogMode* method to disable this mechanism by setting option *reuse_model* to False.

This example shows the usage of the method https://github.com/HLR/DomiKnowS/blob/9e228b31003dcf81dd663697aa988c504db9059b/examples/CIFAR100/main.py#L35

The below log presents difference in execution time of ILP solver for the run using the method and the one with production mode not enabled.

```
2022-10-01 10:59:29,444 - INFO - regrTimer:getDataNode - DataNode Builder the set method called - 28 times
2022-10-01 10:59:29,444 - INFO - regrTimer:getDataNode - DataNode Builder used - 15.62500000ms
2022-10-01 10:59:29,769 - INFO - regrTimer:calculateILPSelection - Calculating ILP Inferencing 
2022-10-01 10:59:31,537 - INFO - regrTimer:calculateILPSelection - ILP Variables Init - time: 1765ms
2022-10-01 10:59:31,537 - INFO - regrTimer:calculateILPSelection - Reusing ILP Model - LCs already present in the model
2022-10-01 10:59:31,538 - INFO - regrTimer:calculateILPSelection - ILP Graph and Ontology Constraints - time: 0ms
2022-10-01 10:59:31,541 - INFO - regrTimer:calculateILPSelection - Starting ILP inferencing - Found 20 logical constraints
2022-10-01 10:59:31,541 - INFO - regrTimer:calculateILPSelection - ILP Logical Constraints Preprocessing - time: 0ms
2022-10-01 10:59:31,541 - INFO - regrTimer:calculateILPSelection - Optimizing model for LCs with probabilities 100 with 8960 ILP variables and 3968 ILP constraints
2022-10-01 10:59:31,541 - INFO - regrTimer:calculateILPSelection - ILP Logical Constraints - time: 0ms
2022-10-01 10:59:31,574 - INFO - regrTimer:calculateILPSelection - Max solution was found in 31ms for p - 100 with optimal value: 23.79
2022-10-01 10:59:31,736 - INFO - regrTimer:calculateILPSelection - ILP Preparing Return Results - time: 171ms
2022-10-01 10:59:31,736 - INFO - regrTimer:calculateILPSelection - End ILP Inferencing - total time: 1.968750s
```

```
11:02:47,221 - INFO - regrTimer:getDataNode - DataNode Builder the set method called - 28 times
2022-10-01 11:02:47,221 - INFO - regrTimer:getDataNode - DataNode Builder used - 46.87500000ms
2022-10-01 11:02:47,606 - INFO - regrTimer:calculateILPSelection - Calculating ILP Inferencing 
2022-10-01 11:02:50,121 - INFO - regrTimer:calculateILPSelection - ILP Variables Init - time: 2500ms
2022-10-01 11:02:50,222 - INFO - regrTimer:calculateILPSelection - ILP Graph and Ontology Constraints - time: 109ms
2022-10-01 11:02:50,226 - INFO - regrTimer:calculateILPSelection - Starting ILP inferencing - Found 20 logical constraints
2022-10-01 11:02:50,226 - INFO - regrTimer:calculateILPSelection - ILP Logical Constraints Preprocessing - time: 0ms
2022-10-01 11:02:50,312 - INFO - regrTimer:addLogicalConstrains - Processing time for LC1(ifL) is: 93ms - created 192 new ILP constraint
2022-10-01 11:02:50,399 - INFO - regrTimer:addLogicalConstrains - Processing time for LC3(ifL) is: 78ms - created 192 new ILP constraint
2022-10-01 11:02:50,485 - INFO - regrTimer:addLogicalConstrains - Processing time for LC5(ifL) is: 93ms - created 192 new ILP constraint
2022-10-01 11:02:50,571 - INFO - regrTimer:addLogicalConstrains - Processing time for LC7(ifL) is: 78ms - created 192 new ILP constraint
2022-10-01 11:02:50,656 - INFO - regrTimer:addLogicalConstrains - Processing time for LC9(ifL) is: 93ms - created 192 new ILP constraint
2022-10-01 11:02:50,772 - INFO - regrTimer:addLogicalConstrains - Processing time for LC11(ifL) is: 109ms - created 192 new ILP constraint
2022-10-01 11:02:50,855 - INFO - regrTimer:addLogicalConstrains - Processing time for LC13(ifL) is: 78ms - created 192 new ILP constraint
2022-10-01 11:02:50,938 - INFO - regrTimer:addLogicalConstrains - Processing time for LC15(ifL) is: 93ms - created 192 new ILP constraint
2022-10-01 11:02:51,020 - INFO - regrTimer:addLogicalConstrains - Processing time for LC17(ifL) is: 78ms - created 192 new ILP constraint
2022-10-01 11:02:51,102 - INFO - regrTimer:addLogicalConstrains - Processing time for LC19(ifL) is: 78ms - created 192 new ILP constraint
2022-10-01 11:02:51,185 - INFO - regrTimer:addLogicalConstrains - Processing time for LC21(ifL) is: 78ms - created 192 new ILP constraint
2022-10-01 11:02:51,268 - INFO - regrTimer:addLogicalConstrains - Processing time for LC23(ifL) is: 93ms - created 192 new ILP constraint
2022-10-01 11:02:51,385 - INFO - regrTimer:addLogicalConstrains - Processing time for LC25(ifL) is: 109ms - created 192 new ILP constraint
2022-10-01 11:02:51,466 - INFO - regrTimer:addLogicalConstrains - Processing time for LC27(ifL) is: 78ms - created 192 new ILP constraint
2022-10-01 11:02:51,547 - INFO - regrTimer:addLogicalConstrains - Processing time for LC29(ifL) is: 93ms - created 192 new ILP constraint
2022-10-01 11:02:51,627 - INFO - regrTimer:addLogicalConstrains - Processing time for LC31(ifL) is: 78ms - created 192 new ILP constraint
2022-10-01 11:02:51,709 - INFO - regrTimer:addLogicalConstrains - Processing time for LC33(ifL) is: 78ms - created 192 new ILP constraint
2022-10-01 11:02:51,790 - INFO - regrTimer:addLogicalConstrains - Processing time for LC35(ifL) is: 78ms - created 192 new ILP constraint
2022-10-01 11:02:51,871 - INFO - regrTimer:addLogicalConstrains - Processing time for LC37(ifL) is: 78ms - created 192 new ILP constraint
2022-10-01 11:02:51,952 - INFO - regrTimer:addLogicalConstrains - Processing time for LC39(ifL) is: 93ms - created 192 new ILP constraint
2022-10-01 11:02:51,952 - INFO - regrTimer:calculateILPSelection - Optimizing model for LCs with probabilities 100 with 8960 ILP variables and 3968 ILP constraints
2022-10-01 11:02:51,952 - INFO - regrTimer:calculateILPSelection - ILP Logical Constraints - time: 1734ms
2022-10-01 11:02:51,988 - INFO - regrTimer:calculateILPSelection - Max solution was found in 31ms for p - 100 with optimal value: 21.27
2022-10-01 11:02:52,432 - INFO - regrTimer:calculateILPSelection - ILP Preparing Return Results - time: 421ms
2022-10-01 11:02:52,432 - INFO - regrTimer:calculateILPSelection - End ILP Inferencing - total time: 4.796875s
```


#### DataNode Builder Optimalization

The DataNode Builder constructs DataNode graph for the executed model. The model sensors provide information about the structure of the DataNode graph and their attributes.
The attributes are provided as tensor with value in the specific index corresponding to the particular DataNode.
In the normal mode of operation the builder with disassemble the tensor and put the appropriate value from it in the corresponding DataNode.
This process takes time, which depends on the size of the tensor. 

In order to limit the time used by the DataNode Builder it is possible to activate the DataNode skeleton mode using the method:

`setDnSkeletonMode(True)`

In the skeleton mode the builder will not distribute the attribute tensors across the DataNode but keep the in the cache.
When the root DataNode is requested from the builder it will add to the root DataNode additional two attributes to the DataNode attribute dictionary:

- `variableSet` - is associated with the dictionary of all the variables learned in the model and the corresponding value tensor
- `prpertySet` - is associated with the dictionary of all the properties in the model and the corresponding value tensor

All the child DataNodes linked with the root have their attributes extended with a single additional attribute:

- `rootDataNode` - with value of the root DataNode

This information allows the solver to transparently find the candidates for the ILP model in the same way as in the full DataNoode mode.