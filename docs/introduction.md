# Intro to DomiKnows

We demonstrate a library for the integration of domain knowledge in deep learning architectures. Using this library, the structure of the data is expressed symbolically via graph declarations and the logical constraints over outputs or latent variables can be seamlessly added to the deep models. The domain knowledge can be defined explicitly, which improves the models’ explainability in addition to the performance and generalizability in the low-data regime. Several approaches for such an integration of symbolic and sub-symbolic models have been introduced; however, there is no library to facilitate the programming for such an integration in a generic way while various underlying algorithms can be used. Our library aims to simplify programming for such an integration in both training and inference phases while separating the knowledge representation from learning algorithms. 

DomiKnows framework generates a systematic process to write declarative code to define learning problems. The framework includes many different algorithms to integrate domain knowledge into the learning problems or optimize the predictions with respect to a set of rules and constraints. The flow of writing a program in DomiKnows includes `Graph Declaration`, `Model Declaration`, `Program Initialization`, `Program composition and execution`.

## Graph Declaration
The graph declaration is the entry point for every task in DomiKnows. Here you define a learning problem as a set of nodes and edges in a graph. The nodes are concept and the edges are relationships between the known concepts.

### Concepts
To define a concept we should use the following format.

```python3
from regr.graph import Concept
concept_var = Concept(name="concept_name")
```
To define a set of concepts in a graph, you can use the following syntax:

```python3
from regr.graph import Graph, Concept, Relation
with Graph('global') as graph:
	concept1 = Concept(name="concept1")
	....
```
You can introduce subgraphs like the following.

```python3
with Graph('global') as graph:
    with Graph('linguistic') as ling_subgraph:
	    concept1 = Concept(name="concept1")
```

We can extend concepts to introduce other concepts in DomiKnows. Extending a concept to generate another one generally means that the derived concept is a decision concept and is either a latent or a output concept in the learning problem.  For instance if we have a concept `phrase`, and we have a decision `people` for each phrase, then the `people` concept is a derived from `phrase`.
We define derived concepts as following:

```python3
concept1 = Concept(name="concept1")
concept2 = concept1(name="concept2")
```
This means that `concept2` is derived from `concept1`.

### Edges
DomiKnows allows a set of specific predefined edges to be added to the nodes to define their relationships. 
The set of valid edges are `is_a`, `has_a`, `contains`, `equal`.

**is_a** edge is used to define derived concept, so another way of defining a derived concept is like the following:

```python3
concept1 = Concept(name="concept1")
concept2 = Concept(name="concept2")
concept2.is_a(concept1)
```

**contains** is used to define parent and child relationship. For instance a `sentence` node contains the `word` node as a sentence is the parent of multiple `word`s. Use the following syntax to define the `contains` edge.

```python3
concept1 = Concept(name="concept1")
concept2 = Concept(name="concept2")
concept1.contains(concept2)
```
Which means that `concept1` is the parent node of `concept2`.

**has_a** edge is defined to generate relationships nodes used to introduce many-to-many relationships between the nodes in the graph. For instance to define a many-to-many relationship between set of two `phrase`s in the graph, first, we have to introduce another concept~(here `pair`) and then establish this edge between the arguments of that relationship~(here `phrase`).

```python3
concept1 = Concept(name="concept1")
concept2 = Concept(name="concept2")
concept1.has_a(arg1=concept1, arg2=concept2)

#Example for phrase and pair
phrase = Concept(name="phrase")
pair = Concept(name="pair")
pair.has_a(phrase, phrase)
```
The number of arguments of this relationship is not limited to 2 and can be more than that. To introduce more arguments simply add more concepts to the has_a relationship definition. `concept.has_a(arg1, arg2, arg3, ...)`.

### Constraints
Very often the classified instance can be assign only a single class, as is the case in the example. This knowledge can be added to the model definition as a first oder logic constrain defined using DomiKnows sub language with the methods `atMostL`.  
 
```python3
atMostL(people, organization, location, other, o)
```
The `atMostL` method in this constrain will ensure that any instance will classified with only one of the concepts listed as the method arguments or none.

The `pair` relations has 5 different possible classes in this example. The `ifL` method can be use to constrain what can be a first and second related instance of each relations type.

```python3
ifL(work_for('x'), andL(people(path=('x', rel_pair_phrase1)), organization(path=('x', rel_pair_phrase2))))
ifL(located_in('x'), andL(location(path=('x', rel_pair_phrase1)), location(path=('x', rel_pair_phrase2))))
ifL(live_in('x'), andL(people(path=('x', rel_pair_phrase1)), location(path=('x', rel_pair_phrase2))))
ifL(orgbase_on('x'), andL(organization(path=('x', rel_pair_phrase1)), location(path=('x', rel_pair_phrase2))))
ifL(kill('x'), andL(people(path=('x', rel_pair_phrase1)), people(path=('x', rel_pair_phrase2))))
```

Each of this constrain is built in the same way. The first argument of the `ifL` is a constrain concept type of `pair`. 
The concept has a name of the variable `x` which represent instances of `pair` in the example. 
The second argument of the `ifL` method is a call to another DomiKnows constrain sub language method - `andL`.
This method required that both its arguments must hold. 
It is used here to ensure that both first and second of the constrain relation instances have required concepts assigned. 
The first related instance is obtain by provided path in the graph starting from the current `x` instance. It is reach by following the edge `rel_pair_phrase1`.
Respectively the second instance is reach by following the edge `rel_pair_phrase2` from the instance `x`,

### Full Example
Following is a full example of how to define a graph for solving Conll entity-mention and relation extraction task.

```python3
from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import ifL, andL, atMostL
from regr.graph.relation import disjoint


Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    with Graph('linguistic') as ling_graph:
        word = Concept(name='word')
        phrase = Concept(name='phrase')
        sentence = Concept(name='sentence')
        (rel_sentence_contains_word,) = sentence.contains(word)
        (rel_sentence_contains_phrase,) = sentence.contains(phrase)
        (rel_phrase_contains_word,) = phrase.contains(word)

        pair = Concept(name='pair')
        (rel_pair_phrase1, rel_pair_phrase2) = pair.has_a(arg1=phrase, arg2=phrase)

    with Graph('application', auto_constraint=True) as app_graph:
        entity = phrase(name='entity')
        people = entity(name='people', auto_constraint=True)
        assert people.relate_to(entity)[0].auto_constraint == True
        organization = entity(name='organization', auto_constraint=False)
        assert organization.relate_to(entity)[0].auto_constraint == False
        location = entity(name='location', auto_constraint=None)
        other = entity(name='other')
        o = entity(name='O')

        atMostL(people, organization, location, other, o)

        work_for = pair(name='work_for')
        work_for.has_a(people, organization, auto_constraint=True)
        
        located_in = pair(name='located_in')
        located_in.has_a(location, location, auto_constraint=False)

        live_in = pair(name='live_in')
        live_in.has_a(people, location, auto_constraint=None)

        orgbase_on = pair(name='orgbase_on')
        kill = pair(name='kill')

        ifL(work_for('x'), andL(people(path=('x', rel_pair_phrase1)), organization(path=('x', rel_pair_phrase2))))

        ifL(located_in('x'), andL(location(path=('x', rel_pair_phrase1)), location(path=('x', rel_pair_phrase2))))
        
        ifL(live_in('x'), andL(people(path=('x', rel_pair_phrase1)), location(path=('x', rel_pair_phrase2))))

        ifL(orgbase_on('x'), andL(organization(path=('x', rel_pair_phrase1)), location(path=('x', rel_pair_phrase2))))
        
        ifL(kill('x'), andL(people(path=('x', rel_pair_phrase1)), people(path=('x', rel_pair_phrase2))))

```
