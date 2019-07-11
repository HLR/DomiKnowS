# Relational Graph

**Re**lational **Gr**aph operations with neural networks.

This library provides a language interface to create a graph by declarations of edges and nodes, as well as any constraints on the graph, against which neural network outputs bound to the graph can be evaluated.
This adds a relational overlay over elements in a network that relates physical concepts in applications.

This project is under heavy development and not yet ready to use.


## TODO

- [x] `emr` example: a simple NLP task that has some structural relation, which shows how the graph should weave everything together.
- [x] `make_model`: graph should be aware of all data and connecting them to the model.
- [ ] Collection: should have a clear way to define sequence/set/... or any collection of basic concepts. The has-a relation holds such semantic but should be made more clear and some detail implementation should be carried out. They should be related to the Rank of each concept.
- [ ] `ComposeConcept` : a clear definition (a class?) of compositional concepts in the graph.
- [ ] `Property`: Properties that are needed, as domain knowledge.
- [ ] Decompose graph with computation further. Take them off from `Concept`.
- [ ] Update inference interface.
- [ ] Documentation
- [x] CI / Testing ... 


## Outline

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


## Related

The project is inspired by [DeLBP (Declarative Learning based Programming)](http://www.cs.tulane.edu/~pkordjam/delbp.htm),
closely related to [Saul programing language](https://github.com/HLR/HetSaul).
[Workshop of DeLBP](http://delbp.github.io/) is held annually to communicate the idea about combining learning models with declarative programming or reasoning.
[DeLBP 2019](http://delbp.github.io/) will be held in conjunction with IJCAI-2019, August 10-16, 2019, Macao, China.
At a wider scope, the project is related to [OntologyBasedLearning](https://github.com/HLR/OntologyBasedLearning), which can provide a source of domain knowledge as the graph (ontology) in this project.


## Prerequirements and setups

Python 3 is required currently while we did considered (very limited) support for Python 2.
[Gurobi](http://www.gurobi.com) is required to solve the constrained optimization problems.
Make sure you have install the Python interface of Gurobi.
If not, the following commands may help you.
```bash
cd GUROBI_HOME
sudo python3 setup.py install
```

Install all python dependency specified in `requirements.txt` by
```bash
sudo python3 -m pip install -r requirements.txt
```
If make sure the check out if there is any additional prerequirements or setup steps in specific `README`, if you want to run an example.

Last but not least, add `regr` to you `PYTHONPATH` environment variables,
```bash
export PYTHONPATH=$PYTHONPATH:$(cd regr && pwd)
```
and you are ready to go!

## Examples

We started some [examples](examples) to see if the design really work.

* [Entity Mentioned Relation](examples/emr)


## Contributing

Some information that might be useful when someone is going to contribute to the project.

`AllenNLP`: https://allennlp.org
Since we focus on Natural Language Processing (NLP) examples at the current stage, we involve `AllenNLP` as model backend in the examples.

`numpydoc`: https://github.com/numpy/numpydoc
We plan to use document the code use `doc string` base on `numpydoc`.

Semaphore CI: https://semaphoreci.com/
We use Semaphore CI to manage code changes.
