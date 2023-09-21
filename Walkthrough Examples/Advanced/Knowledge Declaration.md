# Walkthrough Example

The followings are the user's steps to using our framework.

- Dataset
- **Knowledge Declaration**
- Model Declaration
- Training and Testing
- Inference

## Knowledge Declaration

In knowledge declaration, the user defines a collection of concepts and how they are related, representing the domain knowledge for a task.
We provide a graph language based on Python for knowledge declaration with the notation of `Graph`, `Concept`, `Property`, `Relation`, and `LogicalConstrain`.

The output of the Knowledge Declaration step is a `Graph`, within which there are `Concept`s, `Relation`s, and `LogicalConstrain`s. `Graph` instances are a basic container of the `Concept`s, `Relation`s, `LogicalConstrain`s and other instances in the framework. The `Graph` is a *"partial program"*, and no behavior is associated with it. It is only a data structure to express domain knowledge.

### Graph Declaration

First we define the graph code that defines the domain knowledge for this problem. This graph defines a set of input side data structure in the subgraph of linguistics and define the output decision space in the subgraph of application. We encourage you to follow the same split for more readable graph declaration but the structure is optional.

```python
with Graph('conll') as graph:
    with Graph('linguistic') as ling_graph:
        word = Concept(name='word')
        phrase = Concept(name='phrase')
        sentence = Concept(name='sentence')
        (rel_sentence_contains_word,) = sentence.contains(word)
        (rel_sentence_contains_phrase,) = sentence.contains(phrase)
        (rel_phrase_contains_word,) = phrase.contains(word)

        pair = Concept(name='pair')
        (rel_pair_phrase1, rel_pair_phrase2) = pair.has_a(arg1=phrase, arg2=phrase)
    ...

```

The graph structure can be used as a background knowledge to introduce rules into the inference algorithms automatically, you can specify the keyword auto_constraint to be True if you want to generate automatic constraints based on the graph. You can disable the constraint generation for specific relationship definition by setting the same parameter as False.

```python

with graph:
    with Graph('application', auto_constraint=True) as app_graph:
        entity = phrase(name='entity')
        people = entity(name='people', auto_constraint=True)
        organization = entity(name='organization', auto_constraint=False)
        location = entity(name='location', auto_constraint=None)
        # auto_constraint->True due to its graph
        other = entity(name='other')
        o = entity(name='O')

        work_for = pair(name='work_for')

        located_in = pair(name='located_in')

        live_in = pair(name='live_in')
        # auto_constraint->True due to its graph

        orgbase_on = pair(name='orgbase_on')
        kill = pair(name='kill')
        ...

```

The has_a relationship is equivalant to having a many to many relationships between the arguments in the relation. So the pair, we are introducing a relationship between phrases. The contains relationship also implies a parent child structure which is a one-to-many relationship between the concept that contains the other one and the concept being contained.

### Logical Constraints

To introduce the domain range constraints for relationships, we can use either the has_a graph structure or directluy introduce a constraint by using our Constraint interface. Remember that if you use the has_a to introduce the rules for the domain and range of the arguments, you have to specify the auto_constraints to be True. If you put auto_constraint=None or do not add any auto_constraint input then the auto_constraint value will be inherited from the graph definition as all the nodes and relationships are defined inside the graph.

The exactL constraint is also implying that the concept classes mentioned in the paranthesis are disjoint.

```python
with graph:
    with app_graph:
        ifL(work_for('x'), andL(people(path=('x', rel_pair_phrase1.name)), organization(path=('x', rel_pair_phrase2.name))))

        ifL(located_in('x'), andL(location(path=('x', rel_pair_phrase1.name)), location(path=('x', rel_pair_phrase2.name))))

        ifL(live_in('x'), andL(people(path=('x', rel_pair_phrase1.name)), location(path=('x', rel_pair_phrase2.name))))

        ifL(orgbase_on('x'), andL(organization(path=('x', rel_pair_phrase1.name)), location(path=('x', rel_pair_phrase2.name))))

        ifL(kill('x'), andL(people(path=('x', rel_pair_phrase1.name)), people(path=('x', rel_pair_phrase2.name))))

        exactL(people, organization, location, other, o)
```


See [here](https://github.com/HLR/DomiKnowS/blob/Doc/Main%20Components/Knowledge%20Declaration%20(Graph).md) for more details about declaring graph and constraints.

____
[Goto next section (Model Declaration)](Model%20Declaration.md)