# Demo of complete DomiKnowS program

This example show all the basic steps in configuring learning program in the [DomiKnowS environment](https://hlr.github.io/domiknows-nlp/pipeline).

First the **domain graph** is defined including all the concepts and relations the program is going to learn about the data.

This follows by **definition of logical constraints** which use concepts from the graph and encode the knowledge about the domain.

The sample for the example and the **reader** of the data is presented next.

After that the model sensors and **learner** for the graph concepts are configured from the sets of sensors available in the DomiKnowS environment.
 
The demo uses **POIProgram** to first *train* and then *test* the model.

The demo show how to access the learned parameters and ILP solver solutions from the **Datanode**.

It concludes with **verification** of logical constraints compliance.

## How To Run
```
!python main.py
```
