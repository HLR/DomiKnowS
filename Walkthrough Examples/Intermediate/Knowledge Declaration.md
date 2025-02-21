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


The following is an example of how to declare a graph. 

```python

with Graph('Conll') as graph:
    word = Concept(name='word')
    phrase = Concept(name='phrase')
    sentence = Concept(name='sentence')
    (rel_sentence_contains_word,) = sentence.contains(word)
    (rel_sentence_contains_phrase,) = sentence.contains(phrase)
    (rel_phrase_contains_word,) = phrase.contains(word)

    pair = Concept(name='pair')
    (rel_pair_phrase1, rel_pair_phrase2) = pair.has_a(arg1=phrase, arg2=phrase)

    entity = phrase(name='entity')
    people = entity(name='people', auto_constraint=True)
    organization = entity(name='organization', auto_constraint=False)
    location = entity(name='location', auto_constraint=None)
    other = entity(name='other')
    o = entity(name='O')

    work_for = pair(name='work_for')
    located_in = pair(name='located_in')
    live_in = pair(name='live_in')
    orgbase_on = pair(name='orgbase_on')
    kill = pair(name='kill')
    ...
```

The above code shows the declaration of a `Graph` named `'Conll'` as a variable `graph`. First, we define the fundamental linguistic elements using the `Concept` class:

- `word`
- `phrase`
- `sentence`
- 
Next, relationships are established between these concepts using the `.contains()` method:

- `sentence` contains `word`, represented by `rel_sentence_contains_word`.
- `sentence` contains `phrase`, represented by `rel_sentence_contains_phrase`.
- `phrase` contains `word`, represented by `rel_phrase_contains_word`.

We introduce a new concept, `pair`, which represents a relationship between two phrases. The `.has_a()` method is used to define two roles within this pair:

- `rel_pair_phrase1` and `rel_pair_phrase2`, which each link to a `phrase` entity.

We define `entity` as a special type of `phrase` and further subclass it into specific categories:

- `people` (with `auto_constraint=True`)
- `organization` (with `auto_constraint=False`)
- `location` (with `auto_constraint=None`)
- `other`
- `o`

We introduce additional relationship types using `pair`, each representing a specific kind of relation between entities:

- `work_for`
- `located_in`
- `live_in`
- `orgbase_on`
- `kill`

Then, we continue the graph declaration by defining constraints on our concepts.

```python
with Graph('Conll') as graph:
    ...
    ifL(work_for('x'), andL(people(path=('x', rel_pair_phrase1.name)), organization(path=('x', rel_pair_phrase2.name))))
    ifL(located_in('x'), andL(location(path=('x', rel_pair_phrase1.name)), location(path=('x', rel_pair_phrase2.name))))
    ifL(live_in('x'), andL(people(path=('x', rel_pair_phrase1.name)), location(path=('x', rel_pair_phrase2.name))))
    ifL(orgbase_on('x'), andL(organization(path=('x', rel_pair_phrase1.name)), location(path=('x', rel_pair_phrase2.name))))
    ifL(kill('x'), andL(people(path=('x', rel_pair_phrase1.name)), people(path=('x', rel_pair_phrase2.name))))
    exactL(people, organization, location, other, o)
```


Some constraints are inherent in the graph, such as the relationships already defined between concepts. However, additional constraints must be explicitly defined to enforce logical consistency within the graph structure. The `ifL` function is used to enforce logical constraints by specifying conditions that must be met for a given relationship to exist: Ensures that `work_for` relationships can only exist between a `people` entity (as the first phrase) and an `organization` entity (as the second phrase). Similar constraints are defined for other relationships betweens phrases.

The `exactL` function enforces that an entity must belong **exclusively** to one of the specified categories (`people`, `organization`, `location`, `other`, `o`). This prevents ambiguous classifications where an entity could belong to multiple types.

See [here](https://github.com/HLR/DomiKnowS/blob/Doc/Main%20Components/Knowledge%20Declaration%20(Graph).md) for more details about declaring graph and constraints.

____
[Goto next section (Model Declaration)](Model%20Declaration.md)
