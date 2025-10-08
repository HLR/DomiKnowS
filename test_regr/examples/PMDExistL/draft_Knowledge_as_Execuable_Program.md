## Executable Program

---

We utilize underlying constraints to introduce the executable program within the DomiKnowS framework.
The structure follows the same format as the constraint definition, with additional steps that enable reading and applying rules for specific instances.

The executable program encapsulates the detailed knowledge representation through the three sections below:
- Global Query Declaration – Defines the queries that will be used throughout the program.
- Individual Query Declaration  – Defines the queries that will be used within specific program.
- Support Program – Provides supporting program to enable knowledge as executable program.

### Global Query Declaration 
This section defines the global query declaration, which specifies queries that are executed and utilized to calculate the loss during training for all examples.
Although this approach is uncommon in standard problem settings, it allows the program to learn the corresponding concept across all training examples.

**Example.**
Consider a scenario where your program consistently needs to evaluate the query:

```text
“Does there exist a red and square object?”
```


You can declare this knowledge within the graph declaration to make it executable across the training process.

```python
# Within the graph declaration
program_global = existsL(red('x'), square(path=('x')))
```

Then, you are required to provide the label for constraints during **program declaration**. 
The example syntax of this process is illustrated below,
```python
graph.constraint[program_global] = ReaderSensor('logic_label',  is_constraint=True, label=True)
```

### Individual Query Declaration
This section provides an example of how to define an executable query for a specific instance.
This represents a more common problem setting, where each example requires individual execution.
An example is provided below to illustrate how such a declaration can be defined.

**Example.**
Consider the problem example asking below query:

```text
“Does there exist a blue ball object in this scene?”
```

You can declare both the executable query and the label within the *data* variable, which stores information about the dataset being used.
The executable query should be provided in string format.
These executable query and the label will be treated as variable within the dataset.
An example of the query mentioned above is shown below.

```python
# Within data reader
date[i]['logic_str'] = ''existsL(blue('x'), ball(path=('x')))''
date[i]['logic_label'] = [1]
```

Then, you need to call ``graph.compile_logic`` to add the execution of each specific example, along with its corresponding label, into the DomiKnowS knowledge graph.
This process automatically integrates the defined knowledge as an executable query within the DomiKnowS program.
An example of invoking this function is provided below.

```python
dataset = graph.compile_logic(dataset, logic_keyword='logic_str', logic_label_keyword='logic_label')
```


### Support Program


Currently, we only have one support program for enabling knowledge as executable program. This program is called 
**InferenceProgram**. We provide the example of program declaration of this below,


```python
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel

# scene, objects, is_cond1, is_cond2, relation_obj1_obj2, is_relation1, is_relation2 are variable relevant to executable queries.
program = InferenceProgram(graph, SolverModel,
                           poi=[scene, objects, is_cond1, is_cond2, relation_obj1_obj2, is_relation1, is_relation2,
                                graph.constraint],
                           tnorm="G", inferTypes=['local/argmax'])
```

