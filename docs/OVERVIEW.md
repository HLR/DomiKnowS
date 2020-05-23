## Overview

Relational Graph is a learning-based program.
We propose the pipeline bridging ontologie to a relational graph as a learning-base program.
There are three major steps in the pipeline: 1) ontology declaration, 2) model declaration, and 3) inference.
To fullfill this pipeline, there are several components in this library:

1. `compiler`

   The compiler compiles an ontology which is given in a standard format (for example, [OWL](https://www.w3.org/OWL/)) to a python program that declares a graph with nodes (concepts), relations (edges), and properties.

1. `graph`

   The internal representation of ontology in our pipeline as the output of compiler.
   A language to communicate in.
   Also the body of the learning-based program, referred to as partial program.

1. `sensor`

   The interface to data and computation.
   
   1. `sensor`
   2. `learner`
   
1. `solver`

   Inference by solving constrained optimization problems.
