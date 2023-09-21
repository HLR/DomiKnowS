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


Follows is an example showing how to declare a graph. 

```python

with Graph('email_graph') as graph:
    email = Concept(name='email')

    Spam = email(name='spam')

    Regular = email(name='regular')
    ...

```

The above code shows the declaration of a `Graph` named `'email_graph'` as a variable `graph`. First, we define email, then we define Spam and Regular `Concept`s that are emails Further, in the graph, we define our constraints.

```python
with Graph('email_graph') as graph:
    ...
    disjoint(Spam, Regular)

```

Some constraints are inherent in the graph, such as the relations that are defined in them. But other constraints must be defined explicitly. The constraint here is the `disjoint` constraint between `Spam` and `Regular`. Disjoint means that at most one of these labels can be True simultaneously.

See [here](https://github.com/HLR/DomiKnowS/blob/Doc/Main%20Components/Knowledge%20Declaration%20(Graph).md) for more details about declaring graph and constraints.

____
[Goto next section (Model Declaration)](Model%20Declaration.md)