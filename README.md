# DomiKnowS

This library provides a language interface to create a graph by declarations of edges and nodes, as well as any constraints on the graph, against which neural network outputs bound to the graph can be evaluated.
This adds a relational overlay over elements in a network that relates physical concepts in applications.

This project is under heavy development and not yet ready to use.


## Related

The project is inspired by [DeLBP (Declarative Learning based Programming)](http://www.cs.tulane.edu/~pkordjam/delbp.htm),
closely related to [Saul programing language](https://github.com/HLR/HetSaul).
[Workshop of DeLBP](http://delbp.github.io/) is held annually to communicate the idea about combining learning models with declarative programming or reasoning.
[DeLBP 2019](http://delbp.github.io/) will be held in conjunction with IJCAI-2019, August 10-16, 2019, Macao, China.
At a wider scope, the project is related to [OntologyBasedLearning](https://github.com/HLR/OntologyBasedLearning), which can provide a source of domain knowledge as the graph (ontology) in this project.


## Contributing

Some information that might be useful when someone is going to contribute to the project.

`AllenNLP`: https://allennlp.org
Since we focus on Natural Language Processing (NLP) examples at the current stage, we involve `AllenNLP` as model backend in the examples.

`numpydoc`: https://github.com/numpy/numpydoc
We plan to use document the code use `doc string` base on `numpydoc`.

Semaphore CI: https://semaphoreci.com/
We use Semaphore CI to manage code changes.
