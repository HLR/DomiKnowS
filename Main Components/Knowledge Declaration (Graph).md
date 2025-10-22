# Knowledge Declaration (Graph)

The following is the overview of Knowledge Declaration through Domiknows Graph notations.

- [Introduction](#introduction)
- [Knowledge Declaration](#knowledge-declaration)
  - [Class Overview](#class-overview)
    - [Graph classes](#graph-classes)
    - [Constraints classes](#constraints-classes)
  - [Graph](#graph)
    - [Graph Relationships](#relation-types)
    - [Example](#example)
      - [Graph declaration and `with` statement](#graph-declaration-and-with-statement)
      - [Concept declaration](#concept-declaration)
        - [Direct declaration](#direct-declaration)
        - [Inherit declaration](#inherit-declaration)
    - [Access the nodes](#access-the-nodes)
  - [Constraints](#constraints)
    - [Logical Constraints](#logical-constraints-lc)
    - [Graph Constraints](#graph-constraints)
    - [Ontology Constraints](#ontology-constraint)
  - [Program Execution](#program-execution)

## Introduction

DomiKnows Library allows you to **declare knowledge** about a domain and use it during the training and inference phases. 
Knowledge about the domain could enhance the model's performance, making it more robust and generalizable. 
It also enables a faster learning process by reducing the number of training samples required.

**Knowledge about the domain in DomiKnows is represented as a graph with associated logical constraints.**  
The graph's nodes represent concepts, and the edges represent relations between these concepts. 
The logical constraints define the knowledge about the domain in the form of logical expressions on concepts and relations defined in the graph.

Declaring knowledge begins with introducing the concepts and establishing relationships between them, thereby building the **domain graph**. 
This graph represents the domain knowledge of an ML task. 
Parts of the concepts and relations exemplify classifiers that are subject to learning. 

The parent-child relationship between **concepts** in the graph represents the hierarchical structure of the domain knowledge. The expressions of parent and child are implicit, though, for example, when we write word=concept() ... entity=word(), this implies the is-a relationships between word is-a concept, entity is-a word, and the hierarchy will be concept-> word-> entity.
It implies that if a data item is classified as belonging to a concept, it also belongs to all the parent concepts of that concept. 
Conversely, if the data item is classified as belonging to a concept, it can be classified in more detail as one of the child concepts. 

In the graph, there are two distinct types of concepts.  
* The first type, referred to as **'Output concepts'**, defines the semantic abstraction in the output. 
* The second type, known as **'input concepts'**, specifies the types of data items, for example, words, sentences, pixels, etc. 
 
Output concepts are subordinate to input concepts, thereby establishing a parent-child relationship.  
Depending on the level of data classification granularity, a single data concept may encompass multiple output concepts.

The **relationships** between concepts are employed to denote not only associations within the data items but also categorized relationships between the data and elements of the output  concepts. 
They can depict part-whole relationships, like the relationship between a word and a sentence, or between a word and a phrase.
Another example is to illustrate temporal relationships between events.  
Furthermore, they can also signify relationships between the input data items and the domain knowledge.
For example, the relationship between words based on an entity-mentioned-relationship abstraction can be represented as 'works for' or 'located in', among others.

DomiKnows' **logical constraints** are defined through First Order Logic (FOL) expressions, which effectively encapsulate the domain knowledge.
In FOL, the basic building blocks of logical constraints are **predicates**. In DomiKnows, these predicates are functions that evaluate whether a given variable corresponds to a particular concept or relation.  
The variables in this context refer to an entity or entities in the domain of discourse, which, in this case, pertains to machine learning data items classified by the concepts and relationships derived from the graph during the learning phase of the ML model.

Relationships between predicates can be expressed using **logical operations**.  
We support implication statements ('if') and other logical operations, such as conjunction ('and'), disjunction ('or'), negation ('not'), and biconditional ('if and only if').

**Quantifiers** can be applied to variables within predicate expressions. By default, the universal quantifier 'for every' is presumed, suggesting that each entity from the domain of discourse is subject to the constraint unless stated otherwise.

The DomiKnows library provides constructs that enable the detailed specification of specific quantifiers. These constructs facilitate the selection of entities from the domain of discourse, which the predicate will then evaluate.

Both graphs and logical constraints are defined in **Python** code using constructs from the DomiKnows library.  
Below is an overview of the DomiKnows API and the concepts used to define domain knowledge.

## Class Overview

### Graph classes

- Package `domiknows.graph`: a set of classes for graph and constraints definitions.
- Class `Graph`: classes to construct a graph, its structure, and containers.
- Class `Concept`: classes to define concepts and their properties.
- Class `Property`: a key attached to a `Concept` that can be associated with a specific value assigned by a sensor or a learner
- Class `Relation`: a relation between two `Concepts`.

### Constraints classes

- Package `domiknows.graph.logicalConstrain`: a set of functions with logical semantics via which one can express logical constraints.
- Function `*L()`: functions based on logical notations. Linear constraints can be generated based on the logical constraints. Some of these functions are `ifL()`, `notL()`, `andL()`, `orL()`, `nandL()`, `existsL()`, `equalL()`, etc.

## Graph

`Graph` instances are basic containers of the `Concepts`, `Relations`, and constraints in the framework.
A `Graph` object is constructed either by manually coding or compiled from `OWL` (deprecated).
Each `Graph` object can contain other `Graph` objects as sub-graphs. No cyclic reference in the graph hierarchy is allowed.

You can either write an OWL ontology file, initializing your concepts and relations, or write your graph with our specific Python classes.

Each `Graph` object can contain `Concepts`.

The graph declaration is a part of the program. Sensors and learners, as described later, are data processing units that will be connected to the graph. There is no behavior associated with the graph. It is only a declarative language to express domain knowledge.

### Relation Types

* We have four defined relationship types between nodes that each program can use: `contains`, `has_a`, and `equal` are used to describe relations between concepts. The is-a relation is implicit in concept definitions and inheritance. 

`contains` is a one-to-many relationship. A.contained(B) means that concept `A` is the parent node of concept `B`, and several `B` instances can be the children of one single node `A`.
Whenever a `contains` relationship is used, it indicates a way of generating or connecting parents to children if the children are from the same type.

```Python
sentence = Concept('sentence')
word = Concept('word')
phrase = Concept('phrase')

sentence.contains(word)
phrase.contains(word)
```

The `has_a` relation is a many-to-many relationship. It defines relations between concepts that help in producing candidates of a relationship. For instance, a relationship between `word` and `word` can be defined using an intermediate concept `pair` and two `has_a` relation links as follows, 

```Python
pair = Concept("pair")
pair.has_a(arg1=word, arg2=word)
```

This means that the candidates of a `pair` concept are generated based on a `word_{i}` and a `word_{j}`.
Considering the properties of `contains` and `has_a`, in case of defining a `semantic frame` with multiple semantic elements, we can define it as follows,

```Python
semantic_frame = Concept('semantic-frame')
semantic_frame.has_a(verb=word, subject=word, object=word)
```

We only support relationships between three concepts. Therefore, in case a relation has more arguments, you have to break it into relationships between a main concept and one other concept each time.

```Python
semantic_frame = Concept('semantic-frame')
verb_semantic = Concept('verb-semantic')
subject_semantic = Concept('subject-semantic')
object_semantic = Concept('object-semantic')
verb_semantic.has_a(semantic=semantic_frame, verb=word)
subject_semantic.has_a(semantic=semantic_frame, subject=word)
object_semantic.has_a(semantic=semantic_frame, object=word)
```

The `equal` relation establishes an equality between two different concepts. For instance, if you have two different tokenizers and you want to use features from one of them in another, you have to establish an `equal` edge between the concepts holding those tokenizer instances.

```Python
word = Concept("word")
word1 = Concept("word1")
word.equal(word1)
```

This edge enables us to transfer properties of concepts between instances that are marked as equal. 

Using each of these relation edges requires assigning a sensor to them during model execution. See the descriptions of [Sensors](Model%20Declaration%20(Sensor).md#sensor). 

### Example

The following snippets show an example of a `Graph`.

```Python
with Graph() as graph:
    word = Concept('word')
    pair = Concept(word, word)
    with Graph('sub') as sub_graph:
        people = word('people')
        organization = word('organization')
        work_for = pair(people, organization)
```

#### Graph declaration and `with` statement

The first `with` statement creates a graph, assigns it to the Python variable `graph`, and declares that anything defined under it is a part of the graph.

The second `with` statement declares another graph with an explicit name 'sub'. It will also be attached to the enclosing graph and become a subgraph of it. However, everything under this point will be part of the subgraph, rather than the first graph.

#### Concept declaration

##### Direct declaration

`word = Concept('word')` creates a concept with name `'word'` (implicitly attached to the enclosing graph) and assigns it to the Python variable `word`.

`pair = Concept(word, word)` can be used to create a concept `pair` with two `word` being its arguments (two `HasA` relations are needed to establish the type of connections later).
This declaration does not include an explicit name. If an explicit name is desired, use the keyword `name=` as an argument. For example, `pair = Concept(word, word, name='pair')`.
This new node will also be attached to the enclosing graph.
A `HasA` relation will be added between the new concept and each argument concept. In other words, two `HasA` concepts will be created from `pair` to `word`.
This can be done as follows,

```Python
pair = Concept(name='pair')
pair.has_a(word)
pair.has_a(word)
```

##### Inheritance declaration

`people = word('people')` and `organization = word('organization')` create two new concepts extending `word`. The name is set as 'people' and 'organization', and assigns them to Python variables `people` and `organization`. They are attached to the enclosing subgraph `sub_graph`.
These decorations indicate inheritance and are the syntactic sugar that can be used for creating concepts with `IsA` relations.
An `IsA` relation will be created for each of these statements. It is equivalent to the following statements:

```Python
people = Concept('people')
people.is_a(word)
organization = Concept('organization')
organization.is_a(word)
```

### Access the nodes

 All the sub-`Graphs` and `Concept` instances can be retrieved from a graph (or sub-graph) with a (relative) pathname.
 For example, to retrieve `people` from the above example, one can do `graph['sub/people']` or `sub_graph['people']`.

## Constraints

The constraints are collected from three sources:

- **logical expressions** - primary source of constraint in the system,
- **graph declaration** - definitions in the graph (the relationships declared in the graph, e.g., parent-child relations, etc.)
- **ontology (OWL file)** - alternative source of knowledge, provided as a URL to the OWL file in the ontology graph.

### Logical Constraints (LC)

The basic elements of a logical constraint are its **predicates**. A predicate is constructed using the name of a concept or a relation from the declared graph, and it includes the name of a variable. This variable name is utilized to identify the set of entities pertinent to the current predicate. In DomiKnows, these entities are referred to as **candidates**. The purpose of a predicate is to evaluate whether the given concept or relation positively classifies a candidate.  

If the variable is not explicitly specified in the predicate, then a default variable name will be used. This is typically used when the variable is not referenced in the other parts of the logical constraint.

As in First Order Logic, we have to define the interpretation of a variable in the predicate. An interpretation (or model) of a first-order formula specifies what each predicate means, and the entities that can instantiate the variables. These entities form the domain of discourse or universe. In DomiKnows, the domain of discourse is the set of candidates. By default, the variable in the predicate is associated with all candidates from the data. This basic set of variable candidates is identified by searching the data of the parent 'data node' of the concept or relation used to define the predicate. This default can be modified by specifying the quantifier in the predicate (using 'path'). The quantifier defines the search criteria for selecting the candidates from the data. It employs definitions of paths through the graph to identify the candidates for the predicate. These paths can be augmented with tests that check the values of specified properties of the nodes along the path. If multiple paths are defined, then the candidates are selected from the intersection of the candidates from each path. 

The BNF definition of DomiKnows logical constraint is available at [DomiKnows Logical Constraint BNF](https://tinyurl.com/DomiKnowsConstraint-BNF). This website allows you to test the logical constraint syntax.

This is a simple example of the logical constraint:

```Python
ifL(
    work_for('x'), 
    andL(
      people(path=('x', rel_pair_phrase1)), 
      organization(path=('x', rel_pair_phrase2))
    )
  )
```
This example above states that _for every candidate in the present ML example, if a current candidate is classified as `'work_for'`concept, then the candidates found by following the paths from the current candidate to the first and second argument of the `'pair'` relation have to  be positively classified as `'people'` and `'organization'` concepts_.

The example defines variable `x` representing candidates for the 'work_for' predicate. 
This variable `x` is then used to define candidates for `'people'` and `'organization'` predicates by specifying `path` to them using names of graph edges respectively:`'rel_pair_phrase1'` and `'rel_pair_phrase2'`.

Please notice that `'people'` and `'organization'` predicates do not have their variables specified as they are not referred in other parts of this simple logical constraint.  

```Python
ifL(
    people('p'), 
    atMostL(live_in(path=('p', rel_pair_phrase1.reversed))),
    active = True
  )
```
Another example above states that _for every candidate in the present ML example, if a current candidate is classified as `'people'` concept, then not more then one candidate found by following the path from the current candidate to first argument of the `'pair'` relation will be positively classified as `'live_in'` concept_.

The logical constraint defines variable `p` representing candidates for the `people` predicate. 
This variable is then used to define candidates for the `live_in` predicate by specifying `path` to the candidates using names of the graph edge 'rel_pair_phrase1'. This edge is decorated with the `reversed` keyword to follow the edge in the reversed direction from the `people` concept to the corresponding `live_in` relation candidates.

Additionally, this constraint shows a logical constraint, an optional attribute `active`, which allows for activating or deactivating this constraint.

```Python
ifL(
    city('x'), 
    atMostL(notL(firestationCity(path=('x', eqL(cityLink, 'neighbor', {True}), city2))), 3),
    p=90
  )
```
The above example states that _for every candidate in the present ML example, if a current candidate is classified as `'city'` concept, then not more then three candidates found by following the path from the current candidate to first argument of the `'neighbor'` relation will be positively classified as `'firestationCity'` concept_.

This logical constraint shows the usage of another optional logical constraint attribute `p`, which specifies, with the value from 0 to 100, the certainty of validity of the constraint.   

The full list of DomiKnows functions implementing **logical operations**:
	- `notL()`,
	- `andL()`,
	- `orL()`,
	- `nandL()`,
	- `ifL()`,
	- `norL()`,
	- `xorL()`,
	- `epqL()` (equal p q).

Auxiliary logical constraint methods:  
    - `eqL()` -  used in the path definition to filter based on the specified attribute, e.g.: 
      _eqL(cityLink, 'neighbor', {True}) - instances of cityLink with attribute neighbor in the set containing only the single value True,  
    - `fixedL()`, used to fix selected candidates to selected classification, e.g.:  
       _fixedL(empty_entry_label("x", eqL(empty_entry, "fixed", {True}))) - candidates for empty_entry_label which have attribute fixed equal True should have their classification reset to the value of its attribute label. Candidates who do not have an attribute fixed to True should not have their classification affected.

##### Counting methods

DomiKnows also provides counting methods as an extension of logical connectives. Each counting method contains a list of predicates or nested logical constraints, and optionally, several required predicates that need to be satisfied. If the number isn't specified as the last argument in the counting method, then the default value of 1 is used. There are two  flavors of counting methods: one counts over candidates in the current context of constraint evaluation, and the other counts the domain of discourse, which is the present ML example. The latter type has an 'A' suffix in its name, indicating accumulation. Four types of counting methods are implemented: **exists, exact, atLeast,** and **atMost**.
 
Examples of counting methods usage in the logical constraint:  
* _existsAL(firestationCity)_ -   
    In the current ML example, there exists a candidate with the classification 'firestationCity'.  
* _existsAL(firestationCity, policeStationCity)_ -   
    In the present ML example, there exists a candidate with classification firestationCity or policeStationCity.  
* _exactL(firestationCity, 2)_ -  
    In the present ML example, there are exactly two candidates with classification firestationCity,   
* _atLeastL(firestationCity, 4)_ -  
    In the present ML example, there are at least four candidates with classification firestationCity,  
* _atMostL(ifL(city('x'), firestationCity(path=('x', eqL(cityLink, 'neighbor', {True}), city2))), 4)_ -  
    For every candidate in the present ML example, each city has no more than four candidates that reach through the path cityLink with attribute *neighbors* equal to True, which are classified as firestationCity.  

#### Candidate Selection

The candidates for each predicate in the logical constraint are selected independently from the current ML example. By default, all candidates are chosen. However, this default can be modified by specifying a quantifier within the predicate.

When defining the logical constraint, predicates typically do not use a quantifier, or the predicate's quantifier refers to the variable that represents the candidates for the previous predicates in the current logical constraint. In this case, there will be an equal number of candidates for each predicate in the logical constraint. This is typically the case with logical constraints.

However, if the predicates in the logical constraint employ disjoint quantifiers to select their candidates, the selected sets of candidates for each predicate can be different. This presents a problem when evaluating the logical constraint, as it is unclear how to match candidates between predicates. To solve this problem, DomiKnows allows the definition of a mechanism for selecting candidates for each predicate in the logical constraint. This mechanism is called **candidate selection**. This is achieved by defining a new class that inherits from _CandidateSelection_ and overrides the _get_candidates_ method.

An example of a new candidate selection class is combinationC, which creates a Cartesian product of candidates for each concept in the selection. Here is the code example:

  class combinationC(CandidateSelection):
      def __call__(self, candidates_list, keys=None):
        from  itertools import product
        
        # Create the Cartesian product of all candidates
        cartesian_product = list(product(*candidates_list))
        
        # Extract lists of first elements, second elements, etc.
        extracted_elements = list(zip(*cartesian_product))
              
        # Create a dictionary using the provided keys and the extracted lists
        result_dict = dict(zip(keys, extracted_elements))
        
        return result_dict
  
  This class is already available in the DomiKnows library.

  This class can be used in the logical constraint by specifying the `candidate_selection` parameter, e.g.:

        forAllL(
            combinationC(step, entity)('i', 'e'), # this is the search space
            exactL(
                final_decision('x', path=(('i', rel_step.reversed), ('e', rel_entity.reversed))), 1
            ),
        )
  
In this example, the combination of all possible candidates for `step` and `entity` concepts is created and then returned as a dictionary with keys `i and `e` respectively. This dictionary is then used to define the path for the `final_decision` concept. The forAllL constraint is then applied to all possible assignments of `i and `e` to the path of the `final_decision` concept.

The example shows the generic semantics of the `CandidateSelection` candidate selection class. It takes a list of concepts and returns a dictionary with keys corresponding to the ideas and values corresponding to the candidates for the concepts. The keys are provided as a tuple behind the `CandidateSelection` call.

### Graph Constraints

The graph can also specify constraints:

- **Parent-child relation between concepts**: e.g. `people = word(name='people')` is mapped to logical expression -

  *IfL(people, word)*

- **Disjointing between concepts**: e.g. `disjoint(people, organization, location, other, o)` is mapped to logical expression -

  *atMostL(people, organization, location, other, o, 1)*
  
- **Domain and ranges for relations**: e.g. `work_for.has_a(people, organization)` is mapped to logical expressions-

  *ifL(work_for('x'), andL(people(path=('x', rel_pair_phrase1.name)), organization(path=('x', rel_pair_phrase2.name))))*

### Ontology Constraint

The OWL ontology, on which the learning system graph was built, is loaded into the `ilpOntSolver` and parsed using the Python OWL library [owlready2](https://pypi.org/project/owlready2/).

The OWL ontology language allows specifying constraints on [classes](https://www.w3.org/TR/owl2-syntax/#Classes "OWL Class") and [properties](https://www.w3.org/TR/owl2-syntax/#Object_Properties "OWL Property"). These classes and properties relate to concepts and relations that the learning system builds a classification model for. The solver extracts these constraints.

This detail of mapping from OWL to logical representation is presented below for each OWL constraint.

**Constraints extracted from ontology [classes](https://www.w3.org/TR/owl2-syntax/#Classes "OWL Class") (*concepts*)**:

- **[disjoint](https://www.w3.org/TR/owl2-syntax/#Disjoint_Classes "OWL example of disjoint statement for classes")** statement between two classes *Concept1* and *Concept2* in ontology is mapped to equivalent logical expression -  
  
  *NAND(Concept1(token), Concept2(token))*

- **[equivalent](https://www.w3.org/TR/owl2-syntax/#Equivalent_Classes "OWL example of equivalent statement for classes")** statement between two classes *Concept1* and *Concept2* in ontology is mapped to equivalent logical expression -  
  
  *AND(Concept1(token), concept2(token))*

- **[subClassOf](https://www.w3.org/TR/owl2-syntax/#Subclass_Axioms "OWL example of subclassS statement for classes")** statement between two classes *Concept1* and *SuperConcept2* in ontology is mapped to equivalent logical expression -  
  
  *IF(concept1(token), SuperConcept2(token))*

- **[intersection](https://www.w3.org/TR/owl2-syntax/#Intersection_of_Class_Expressions "OWL example of intersection statement for classes")** statement between classes *Concept1*, *Concept2*, *Concept3*, ... in ontology is mapped to equivalent logical expression
  
  *AND(Concept1(token), Concept2(token), Concept3(token), ..)*

- **[union](https://www.w3.org/TR/owl2-syntax/#Union_of_Class_Expressions "OWL example of union statement for classes")** statement between classes *Concept1*, *Concept2*, *Concept3*, ... in ontology is mapped to equivalent logical expression -  

  *OR(concept1(token), Concept2(token), Concept3(token), ..)*

- **[objectComplementOf](https://www.w3.org/TR/owl2-syntax/#Complement_of_Class_Expressions "OWL example of complement of statement for classes")** statement between two classes *Concept1* and *Concept2* in ontology is mapped to equivalent logical expression -
  
  *XOR(Concept1(token), Concept2(token))*

**No supported yet**

- **[disjonitUnion](https://www.w3.org/TR/owl2-syntax/#Disjoint_Union_of_Class_Expressions "OWL example of disjointUnion of classes")** statement between classes *Concept1*, *Concept2*, *Concept3*, ... is Not yet supported by owlready2 Python ontology parser used inside the solver

- **[oneOf](https://www.w3.org/TR/owl2-syntax/#Enumeration_of_Individuals "OWL example of enumeration of individuals for classes")** statements for a class *Concept* in ontology

**Constraints extracted from ontology [properties](https://www.w3.org/TR/owl2-syntax/#Object_Properties "OWL Property") (*relations*)**

- **[domain](https://www.w3.org/TR/owl2-syntax/#Object_Property_Domain "OWL example of domain statement for property")** of relation *P(token1, token2)* statements in ontology are mapped to equivalent logical expression -  

  *IF(P(token1, token2), domainConcept(token1))*
  
- **[range](https://www.w3.org/TR/owl2-syntax/#Object_Property_Range, "OWL example of range statement for property")** of relation *P(token1, token2)* statements in ontology are mapped to equivalent logical expression -  

  *IF(P(token1, token2), rangeConcept(token2))*
  
- **[subproperty](https://www.w3.org/TR/owl2-syntax/#Object_Subproperties "OWL example of subproperty statement for properties")** of relations *P(token1, token2)* and *SP(token1, token2)* statements in ontology are mapped to equivalent logical expression -  

  *IF(P(token1, token2), SP(token1, token2))*

- **[equivalent](https://www.w3.org/TR/owl2-syntax/#Equivalent_Object_Properties "OWL example of equivalent statement for properties")** of relations *P1(token1, token2)* and *P2(token1, token2)* statements in ontology are mapped to equivalent logical expression -  

  *AND(P1(token1, token2), P2(token1, token2))*

- **[inverse](https://www.w3.org/TR/owl2-syntax/#Inverse_Object_Properties_2 "OWL example of inverse statement for properties")** relations *P1(token1, token2)* and *P2(token1, token2)* statements in ontology are mapped to equivalent logical expression -

  *IF(P1(token1, token2), P2(token1, token2))*

- **[reflexive](https://www.w3.org/TR/owl2-syntax/#Reflexive_Object_Properties "OWL example of reflexive statement for property")** relation *P(token1, token2)* statements in ontology are mapped to equivalent logical expression -

  *P(token, token)*

- **[irreflexive](https://www.w3.org/TR/owl2-syntax/#Irreflexive_Object_Properties "OWL example of irreflexive statement for property")** relation *P(token1, token2)* statements in ontology are mapped to equivalent logical expression -  

  *NOT(P(x,x))*

- **[symmetrical](https://www.w3.org/TR/owl2-syntax/#Symmetric_Object_Properties "OWL example of symemtrical statement for property")** relation *P(token1, token2)* statements in ontology are mapped to equivalent logical expression -  

  *IF(P(token1, token2), P(token2, token1))*

- **[asymmetric](https://www.w3.org/TR/owl2-syntax/#Asymmetric_Object_Properties "OWL example of asymmetric statement for property")** relation *P(token1, token2)* statements in ontology are mapped to equivalent logical expression -  

  *Not(IF(P(token1, token2), P(token2, token1)))*

- **[transitive](https://www.w3.org/TR/owl2-syntax/#Transitive_Object_Properties "OWL example of asymetric statement for property")** relation *P(token1, token2)* statements in ontology are mapped to equivalent logical expression -  

  *IF(AND(P(token1, token2) and P(token2, token3)), P(token1, token3))*
  
- **[allValuesFrom](https://www.w3.org/TR/owl2-syntax/#Universal_Quantification "OWL example of allValuesFrom statement for property")** statements for relation *P(token1, token2)* in ontology are mapped to equivalent logical expression -  

  *...*
  
- **[someValueFrom](https://www.w3.org/TR/owl2-syntax/#Existential_Quantification "OWL example of someValueFrom statement for property")** statements statements for relation *P(token1, token2)* in ontology are mapped to equivalent logical expression -  

  *This is an Existential constraint, not possible to check without the assumption of a closed world*
  
- **[hasValue](https://www.w3.org/TR/owl2-syntax/#Existential_Quantification "OWL example of hasValue statement for property")** statements statements for relation *P(token1, token2)* in ontology are mapped to equivalent logical expression -  

  *This is an Existential constraint, not possible to check without the assumption of a closed world*

- **[objectHasSelf](https://www.w3.org/TR/owl2-syntax/#Self-Restriction "OWL example of objectHasSelf statement for property")** statements for relation *P(token1, token2)* in ontology are mapped to equivalent logical expression -  

  *...*

- **[disjoint](https://www.w3.org/TR/owl2-syntax/#Disjoint_Object_Properties "OWL example of disjoint statement for properties")** statements for relations *P1(token1, token2)* and *P2(token1, token2)* in ontology are mapped to equivalent logical expression -  

  *NOT(IF(P1(token1, token2), P2(token1, token2)))*

- **[key](https://www.w3.org/TR/owl2-syntax/#Keys "OWL example of key statement for property")** statements for relation *P(token1, token2)*  in ontology are mapped to equivalent logical expression -  

  *...*

- **[exactCardinality](https://www.w3.org/TR/owl2-syntax/#Exact_Cardinality "OWL example of exactCardinality statement for property")** statements for relation *P(token1, token2)*  in ontology are mapped to equivalent logical expression -  

  *This is an Existential constraint, not possible to check without the assumption of a closed world*

- **[minCardinality](https://www.w3.org/TR/owl2-syntax/#Minimum_Cardinality "OWL example of minCardinality statement for property")** statements for relation *P(token1, token2)*  in ontology are mapped to equivalent logical expression -  

  *This is an Existential constraint, not possible to check without the assumption of a closed world*

- **[maxCardinality](https://www.w3.org/TR/owl2-syntax/#Maximum_Cardinality "OWL example of maxCardinality statement for property")** statements for relation *P(token1, token2)*  in ontology are mapped to equivalent logical expression -  

  *count of token2 for which P(token1, token2) <= maxCardinality*
  
- **[functional](https://www.w3.org/TR/owl2-syntax/#Functional_Object_Properties "OWL example of functional statement for properties")** relation *P(token1, token2)* statements in ontology are mapped to equivalent logical expression -  

  *Syntactic shortcut for the following maxCardinality of P(token1, token2) is 1*

- **[inverse functional](https://www.w3.org/TR/owl2-syntax/#Inverse-Functional_Object_Properties "OWL example of inverse functional statement for properties")** relation *P(token1, token2)* statements in ontology are mapped to equivalent logical expression -  

  *Syntactic shortcut for the following maxCardinality of inverse P(token2, token1) is 1*

## Program Execution

**Note**: These features are in active development.

The same logical expressions described above for the purpose of specifying constraints can also be used to describe *programs* that are executed during training and inference. We can then train the underlying model based on the ground-truth outputs of the these *programs*.

For example, in [Clever](https://github.com/HLR/DomiKnowS/tree/develop-CLEVER-relations/test_regr/Clever), we have questions (e.g., "*Does there blue big square in the image?*") represented with logical expressions (`existL(is_blue('x'), is_big(path=('x')), is_square(path=('x')))`) along with the ground-truth answers to those questions (i.e., `True`/`False`). During training, we update model parameters so that the predicted program outputs (i.e., answers to the questions) match the ground-truth values.

Specifically, during training, the program outputs are calculated by finding the (differentiable) soft-logic conversion of the underlying logical expressions. Model parameters are then updated based on the loss between these predicted program outputs and the ground-truth program outputs.

To do this in DomiKnowS, we need to add the programs (i.e., the logical expressions) that we want to execute to the graph and also provide the ground-truth outputs to these programs. There are two ways to do this, depending on whether the programs are the same across training examples or if they are different.
- Global Program Query – The queries that will be used throughout the program.
- Individual Program Query  – The queries that will be used within specific instant.


### Programs are the same across examples (Global Program Query)
If the program(s) we'd like to execute are fixed across training examples, then the API is similar to performing regular supervised learning in DomiKnowS.

We specify programs in the same way that we would specify logical constraints except, now, we also assign the expression to a variable as we need to be able to reference it later. For example, we could have the following question which can be specified as follows:

```python
with Graph('image_graph') as graph:
  # [...]

  # Does there blue big square in the image?
  question_output = existL(
    is_blue('x'),
    is_big(path=('x')),
    is_square(path=('x'))
  )
```

During training, we want to update model parameters based on the ground-truth answer of this question (which can be either `True` or `False` whereas, with logical *constraints*, we would always update model parameters to make the expression `True`). Importantly, here, the question is the same across training examples even though the answers may be different. Like for regular supervised learning, we specify labels with a sensor (e.g., a ReaderSensor) except, now, we assign it to a property of the `constraint` Concept in the graph:

```python
graph.constraint[question_output] = ReaderSensor(keyword="question_label", label=True)
```

All the other sensors & module learners can be specified in exactly the same way as before.

Finally, we use the `InferenceProgram` class:
```python
program = InferenceProgram(
  graph,
  SolverModel,
  poi=[...],
  tnorm='G' # This parameter specifies how we perform the soft-logic conversion during program execution
)
```

Training is the same as with supervised learning as well:
```python
program.train(dataset, ...)
```

The dataset here must specify the program output label (here, in the `question_label` key) in each data item.

A complete example of this API can be found [here](https://github.com/HLR/DomiKnowS/tree/develop-CLEVER-relations/test_regr/InferenceAPI).

### Programs are different across examples (Individual Program Query)
On the other hand, if the programs to be executed are different across training examples, we need to specify *both* the program (in the form of logical expression strings) and the program output labels in each data item. The syntax for specifying the programs is the same as before, except now, instead of directly adding it to the graph, we specify it through the dataset first *then* add it to the graph.

The previous example would be specified as:
```python
dataset = [
  # [...]
  {
    # Does there blue big square in the image?
    'logic_str': "existL(is_blue('x'), is_big(path=('x')), is_square(path=('x')))",

    # False
    'logic_label': [0] ,

    #[...]
  },
  # [...]
]
```

Then, we call ``graph.compile_logic`` to add the execution of each specific example, along with its corresponding label, into the DomiKnowS knowledge graph.
This process automatically integrates the defined knowledge as an executable query within the DomiKnowS program.
An example of invoking this function is provided below:

```python
# Notice that compile_logic returns a new dataset instance here.
# During training, this "transformed" dataset should be used as it
#   contains metadata needed for program execution.
transformed_dataset = graph.compile_logic(
  dataset,

  # The key used in the dataset corresponding to the executed
  #   logical expressions
  logic_keyword='logic_str',

  # The key used in the dataset corresponding to the ground-truth
  #   values of the expressions
  logic_label_keyword='logic_label'
)
```

Finally, we initialize the `InferenceProgram` and train just like before:
```python
program = InferenceProgram(
  graph,
  SolverModel,
  poi=[...],
  tnorm='G'
)

program.train(transformed_dataset, ...)
```

Two complete examples of this API can be found [CLEVR Example](https://github.com/HLR/DomiKnowS/tree/develop-CLEVER-relations/test_regr/Clever) and [Relation Learning Example](https://github.com/HLR/DomiKnowS/tree/develop-CLEVER-relations/test_regr/examples/PMDExistL/relation_learning_tests)
