## 1. Knowledge Declaration

Class reference:

- `domiknows.graph.Graph`
- `domiknows.graph.Concept`
- `domiknows.graph.Property`
- `domiknows.graph.Relation`
- `domiknows.graph.LogicalConstrain`
- `domiknows.graph.Datanode`

In knowledge declaration, the user defines a collection of concepts and the way they are related to each other, representing the domain knowledge for a task.
We provide a graph language based on python for knowledge declaration with the notation of `Graph`, `Concept`, `Property`, `Relation`, and `LogicalConstrain`.

The output of the Knowledge Declaration step is a `Graph`, within which there are `Concept`s, `Relation`s, and `LogicalConstrain`s. `Graph` instances are a basic container of the `Concept`s, `Relation`s, `LogicalConstrain`s and other instances in the framework. The `Graph` is a *"partial program"*, and there is no behavior associated. It is only a data structure to express domain knowledge.


Follows is an example showing how to declare a graph. in this example, we have a paragraph, each paragraph has some questions related to it and the answer to each question can be "more", "less" and "no effect".

```python
with Graph('WIQA_graph') as graph:

    paragraph = Concept(name='paragraph')
    question = Concept(name='question')
    para_quest_contains, = paragraph.contains(question)

    is_more = question(name='is_more')
    is_less = question(name='is_less')
    no_effect = question(name='no_effect')

    symmetric = Concept(name='symmetric')
    s_arg1, s_arg2 = symmetric.has_a(arg1=question, arg2=question)

    transitive = Concept(name='transitive')
    t_arg1, t_arg2, t_arg3 = transitive.has_a(arg11=question, arg22=question, arg33=question)
    ...

```

The above code shows the declaration of a `Graph` named `'WIQA_graph'` as variable `graph`.

first, we define paragraph, then we define questions and add a contains relation from paragraph to question

In the graph, there are `Concepts`s named `'paragraph'`, `'question'`, `'symmetric'` and `'transitive'` as python variables with the same name. 
`symmetric` has two arguments named `arg1` and `arg2`, which are both `question`.
`transitive` on the other hand has three arguments named `arg11`, `arg22` and `arg33`, all of which are `question` as well.
`is_more` , `is_less` and `no_effect` are concepts that have IsA relation with `question`. we will use these three concepts as labels of questions as the answer to these questions can be one of these three.

further, in the graph, we define our constraints.

```python
with Graph('WIQA_graph') as graph:
    ...
    disjoint(is_more, is_less, no_effect)
    orL(is_more, is_less, no_effect)
    
    ifL(is_more('x'), is_less('y', path=('x', symmetric.name, s_arg2.name)))
    ifL(is_less('x'), is_more('y', path=('x', symmetric.name, s_arg2.name)))

    ifL(andL(is_more('x'), is_more('z', path=('x', transitive.name, t_arg2.name))),
        is_more('y', path=('x', transitive.name, t_arg3.name)))

    ifL(andL(is_more('x'), is_less('z', path=('x', transitive.name, t_arg2.name))),
        is_less('y', path=('x', transitive.name, t_arg3.name)))
```

some constraints are inherent in the graph such as the relations that are defined in them. but other constraints must be defined explicitly. 
the first constraint is the `disjoint` constraint between `is_more` , `is_less` and `no_effect`. disjoint means that at most one of these labels can be True at the same time. in the next line, we add `orL` among our labels to make sure at least one of them is correct as well.

further, we define the symmetric and transitive constraints. 

the symmetric relation is between questions that are opposite of each other and have opposing values. we define that if a question is `is_more` or `is_less` and it has asymmetric relation with another question, then the second question should be `is_less` and `is_more` respectively.

the transitive relation is between questions that have a transitive relation between them meaning that the effect of the first question is the cause of the second question and the third question is made of the cause of the first and the effect of the second question. the transitive relation implies that if the first and the second question are `is_more`, so should be the third question. but if the first question is `is_more` and the second question is `is_less`, then the third question should also be `is_less`

The following figure illustrates the graph for this task:
![plot](WIQA.png)

See [here](developer/KNOWLEDGE.md) for more details about declaring graph and constraints.
