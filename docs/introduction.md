# Intro to DomiKnows

Domiknows framework generates a systematic process to write declarative code to define learning problems. The framework includes many different algorithms to integrate domain knowledge into the learning problems or optimize the predictions with respect to a set of rules and constraints. The flow of writing a program in Domiknows includes `Graph Declaration`, `Model Declaration`, `Program Initialization`, `Program composition and execution`.

## Graph Declaration
the graph declaration is the entry point for every task in domiknows. Here you define a learning problem as a set of nodes and edges in a graph. The nodes are concept and the edges are relationships between the known concepts.

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

We can extend concepts to introduce other concepts in Domiknows. Extending a cncept to generate another one generally means that the derived concept is a decision concept and is either a latent or a output concept in the learning problem.  For instance if we have a concept `phrase`, and we have a decision `people` for each phrase, then the `people` concept is a derived from `phrase`.
We define derived concepts as following:
```python3
concept1 = Concept(name="concept1")
concept2 = concept1(name="concept2")
```
This means that `concept2` is derived from `concept1`.

### Edges
Domiknows allows a set of specific predefined edges to be added to the nodes to define their relationships. 
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
....

### Full Example
Following is a full example of how to define a graph for sovling Conll entity-mention and relation extraction task.

```python3
from regr.graph import Graph, Concept, Relation
from regr.graph.logicalConstrain import ifL, andL, atMostL, V, exactL
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

        disjoint(people, organization, location, other, o)

        work_for = pair(name='work_for')
        work_for.has_a(people, organization, auto_constraint=True)
        
        located_in = pair(name='located_in')
        located_in.has_a(location, location, auto_constraint=False)

        live_in = pair(name='live_in')
        live_in.has_a(people, location, auto_constraint=None)

        orgbase_on = pair(name='orgbase_on')
        kill = pair(name='kill')

        ifL(work_for'x'), andL(people(path=('x', rel_pair_phrase1.name)), organization(path=('x', rel_pair_phrase2.name))))

        ifL(located_in'x'), andL(location(path=('x', rel_pair_phrase1.name)), location(path=('x', rel_pair_phrase2.name))))
        
        ifL(live_in'x'), andL(people(path=('x', rel_pair_phrase1.name)), location(path=('x', rel_pair_phrase2.name))))

        ifL(orgbase_on'x'), andL(organization(path=('x', rel_pair_phrase1.name)), location(path=('x', rel_pair_phrase2.name))))
        
        ifL(kill'x'), andL(people(path=('x', rel_pair_phrase1.name)), people(path=('x', rel_pair_phrase2.name))))

```
