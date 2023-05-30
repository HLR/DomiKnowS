# Knowledge Declaration

DomiKnows Library allows to **define knowledge** about a domain and use it during the learning and evaluation phases of ML model preparation, as well as during the model usage. 
The knowledge about the domain enhances the model's performance and makes it more robust against noise in the data. 
It also enables a faster learning process and reduces the number of training samples required for the model to learn.

**Knowledge about the domain in DomiKnows is represented as a graph with associated logical constraints.**  
The graph's nodes are concepts, and edges are relations between concepts. 
The logical constraints define the knowledge about the domain in the form of logical expressions on concepts and relations from the graph.

Capturing knowledge begins with gathering concepts and establishing their relationships to each other, thereby building the **domain graph**. 
This graph represents the domain knowledge in an ML task. 
The concepts and relations exemplify the classification of the data and the relationships among the data, which are the subjects of the ML task.

The parent-child relation between **concepts** in the graph is used to represent the hierarchical structure of the domain knowledge. 
It implies that if the data is classified as belonging to a concept, it also belongs to all the parent concepts of that concept. 
Conversely, if the data is classified as belonging to a concept, it can be classified in more detail as one of the child concepts. 

In the graph, there are two distinct types of concepts.  
* The first type, referred to as **'classification concepts'**, define the classification of the data within _semantic frame_. 
* The second type, known as **'data concepts'**, specify the types of data, for example, words, sentences, pixels, etc. 
 
Classification concepts are subordinate to data concepts, thereby establishing a parent-child relationship.  
Depending on the level of data classification granularity, a single data concept may encompass multiple classification concepts.

The **relationships** between concepts are employed to denote not only associations within the data but also categorized relationships between the data and elements of the _semantic frame_. 
They can depict part-whole relationships, like the relationship between a word and a sentence, or between a word and a phrase.
Additionally, they can illustrate temporal relationships between events.  
Furthermore, they can also signify relationships between the data and the domain knowledge.
For example, the relationship between a word and a _semantic frame_ can be represented, such as 'works for' or 'located in', among others

DomiKnows' **logical constraints** are defined through First Order Logic (FOL) expressions, which effectively encapsulate the domain knowledge.
In FOL, the basic building blocks of logical constraints are **predicates**. In DomiKnows, these predicates are functions that evaluate whether a given variable corresponds to a certain concept or relation.  
The variables in this context refer to an entity or entities in the domain of discourse, which, in this case, pertains to machine learning data classified by the concepts and relationships derived from the graph during the learning phase of the ML model.

Relationships between predicates can be expressed using **logical connectives**.  
The most common is the conditional statement ('if'), but all other logical connectives, such as conjunction ('and'), disjunction ('or'), negation ('not'), and biconditional ('if and only if'), are supported as well.

**Quantifiers** can be applied to variables within predicate expressions. By default, the universal quantifier 'for every' is presumed, suggesting that each entity from the domain of discourse is subject to the constraint unless stated otherwise.

The DomiKnows library offers constructs that allow for the detailed specification of particular quantifiers. These constructs facilitate the selection of entities from the domain of discourse, which will then be evaluated by the predicate.

Both graphs and logical constraints are defined in **Python** code using constructs from the DomiKnows library.  
Blow is the overview of the DomiKnows API and concepts used to define the domain knowledge.

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

## Class Overview

### Graph classes

- Package `domiknows.graph`: the set of class for graph and constraints definitions.
- Class `Graph`: classes to construct graph, its structure and containers.
- Class `Concept`: classes to define concepts and their properties.
- Class `Property`: a key attached to a `Concept` that can be associated with certain value assigned by a sensor or a learner
- Class `Relation`: a relation between two `Concepts`.

### Constraints classes

- Package `domiknows.graph.logicalConstrain`: a set of functions with logical semantics, that one can express logical constraints.
- Function `*L()`: functions based on logical notations. Linear constraints can be generated based on the logical constraints. Some of these functions are `ifL()`, `notL()`, `andL()`, `orL()`, `nandL()`, `existsL()`, `equalL()`, etc.

## Graph

`Graph` instances are basic container of the `Concepts`, `Relations`, constraints and other instances in the framework.
A `Graph` object is constructed either by manually coding or compiled from `OWL` (deprecated).
Each `Graph` object can contain other `Graph` objects as sub-graphs. No cyclic reference in graph hierarchy is allowed.

You can either write an OWL ontology file initializing your concepts and relations or to write your graph with our specific Python classes.

Each `Graph` object can contain `Concepts`.

The graph is a partial program, and there is no sensor or learner, which are data processing units, connected. There is no behavior associated. It is only a data structure to express domain knowledge.

### Relation Types

We have three defined relationship between nodes that each program can use. `contains`, `has_a`, and `equal` are used to define relations between concepts. 

`contains` means that concept `A` is the parent node of concept `B` and several `B` instances can be the children of one single node `A`.
When ever a `contains` relationship is used, it indicates a way of generating or connecting parent to children if the children are from the same type.

```Python
sentence = Concept('sentence')
word = Concept('word')
phrase = Concept('phrase')

sentence.contains(word)
phrase.contains(word)
```

we use the relationship `has_a` only to define relations between concepts and to produce candidates of a relationship. For instance, a relationship between `word` and `word` can be defined using an intermediate concept `pair` and two `has_a` relation links.

```Python
pair = Concept("pair")
pair.has_a(arg1=word, arg2=word)
```

This means that the candidates of a `pair` concept are generated based on a `word_{i}` and a `word_{j}`.
Considering the properties of `contains` and `has_a`, in case of defining a `semantic frame` we have to define the following code.

```Python
semantic_frame = Concept('semantic-frame')
semantic_frame.has_a(verb=word, subject=word, object=word)
```

As we only support relationships between three concepts, in case of a relation with more arguments, you have to break it to relationships between a main concept and one other concept each time.

```Python
semantic_frame = Concept('semantic-frame')
verb_semantic = Concept('verb-semantic')
subject_semantic = Concept('subject-semantic')
object_semantic = Concept('object-semantic')
verb_semantic.has_a(semantic=semantic_frame, verb=word)
subject_semantic.has_a(semantic=semantic_frame, subject=word)
object_semantic.has_a(semantic=semantic_frame, object=word)
```

the `equal` relation establishes an equality between two different concept. for instance, if you have two different tokenizers and you want to use features from one of them into another, you have to establish an `equal` edge between the concepts holding those tokenizer instances.

```Python
word = Concept("word")
word1 = Concept("word1")
word.equal(word1)
```

This edge enables us to transfer properties of concepts between instances that are marked as equal.

Using each of these relation edges requires us to assign a sensor to them in the model execution. The sensors input is selected properties from the source concept and the output will be stored as selected properties in the destination node.

### Example

The following snippets shows an example of a `Graph`.

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

The first `with` statement creates a graph, assigns it to Python variable `graph`, and declares that anything below are attached to the graph.

The second `with` statement declare another graph with an explicit name `'sub'`. It will also be attached to the enclosing graph, and become a subgraph. However, everything below this point will be attached to the subgraph, instead of the first graph.

#### Concept declaration

##### Direct declaration

`word = Concept('word')` creates a concept with name `'word'` (implicitly attached to the enclosing graph) and assign it to Python variable `word`.

`pair = Concept(word, word)` is syntactic sugar to creates a concept with two `word`s being its arguments (with two `HasA` relations).
It does not has an explicit name. If a explicit name is designable, use keyword argument `name=`. For example, `pair = Concept(word, word, name='pair')`.
It will also be attached to the enclosing graph.
A `HasA` relation will be added between the new concept and each argument concept. That implies, two `HasA` concepts will be created from `pair` to `word`.
It is equivalent to the following statements:

```Python
pair = Concept(name='pair')
pair.has_a(word)
pair.has_a(word)
```

##### Inherit declaration

`people = word('people')` and `organization = word('organization')` create another two concepts extending `word`, setting name `'people'` and `'organization'`, and assign them to Python variable `people` and `organization`. They are attached to enclosing subgraph `sub_graph`.
Inherit declaration is syntactic sugar for creating concepts and `IsA` relations.
An `IsA` relation will be create for each of these statements. It is equivalent to the following statements:

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

- **logical constraints** - main source of constraint in the system,
- **knowledge graph** - definitions in the graph (already described, e.g. parent-child relations, etc.)
- **ontology (OWL file)** - alternative source of knowledge, provided as url to OWL file in the ontology graph.

### Logical Constraints (LC)

The foundational element of a logical constraint is the definition of its **predicates**. A predicate is constructed using the name of a concept or a relation from the pre-defined graph, and it includes the name of a variable. This variable name is utilized to identify the set of entities pertinent to the current predicate. In DomiKnows, these entities are referred to as **candidates**. The purpose of a predicate is to evaluate whether a candidate is positively classified by the given concept or relation.  

If the variable is not explicitly specified in the predicate, then the default variable name will be used. This is usually used when the variable is not referred in the other parts of the logical constraint.

By default, the variable in the predicate is associated with all candidates from the data, which are identified by searching the data of the parent 'data node' of the concept or relation used to define the predicate. This default can be modified by specifying the quantifier in the predicate (using 'path'). The quantifier defines the search criteria for selecting the candidates from the data. It employs definitions of paths through the graph to identify the candidates for the predicate. These paths can be augmented with tests checking values of specified properties of the nodes in the path. If multiple paths are defined, then the candidates are selected from the intersection of the candidates from each path.

This is the simple example of the logical constraint:

```Python
ifL(
    work_for('x'), 
    andL(
      people(path=('x', rel_pair_phrase1)), 
      organization(path=('x', rel_pair_phrase2))
    )
  )
```
This example above states that _for every candidate in the present ML example if a current candidate is classified as `'work_for'`concept, then the candidates found by following the paths from the current candidate to first and second argument of the `'pair'` relation have to  be positively classified as `'people'` and `'organization'` concepts_.

More detail about the syntax of the constraint - the example defines variables `x` representing candidates for `'work_for'` predicate. 
This variable `x` is then used to define candidates for `'people'` and `'organization'` predicates by specifying `path` to them using names of graph edges respectively:`'rel_pair_phrase1'` and `'rel_pair_phrase2'`.

Please notice that `'people'` and `'organization'` predicates do not have their variables specified are they are not referred in other parts of this simple logical constraint.  

```Python
LC_SET_ADDITIONAL = True

ifL(
    people('p'), 
    atMostL(live_in(path=('p', rel_pair_phrase1.reversed))),
    active = LC_SET_ADDITIONAL
  )
```
Another example above states that _for every candidate in the present ML example if a current candidate is classified as `'people'` concept, then not more then one candidate found by following the path from the current candidate to first argument of the `'pair'` relation will be positively classified as `'live_in'` concept_.

The logical constraint defines variable `p` representing candidates for `'people'` predicate. 
This variable is then used to define candidates for `'live_in'` predicate by specifying `path` to the candidates using names of graph edge `'rel_pair_phrase1'`. This edge is decorated with `reversed` keyword to follow the edge in reversed direction from `people` concept to corresponding `live_in` relation candidates.

Additionally this constraint shows logical constrain optional attribute `active` which allow to activate or deactivate this constraint.

```Python
ifL(
    city('x'), 
    atMostL(notL(firestationCity(path=('x', eqL(cityLink, 'neighbor', {True}), city2))), 3),
    p=90
  )
```
The example states that _for every candidate in the present ML example if a current candidate is classified as `'city'` concept, then not more then three candidates found by following the path from the current candidate to first argument of the `'neighbor'` relation will be positively classified as `'firestationCity'` concept_.

Ths logical constrain show usage of another optional logical constrain attribute `p` which specify with the value from 0 to 100 the certainty of validity of the constraint.   

The full list of DomiKnows functions implementing **logical connectives**:
	- `notL()`,
	- `andL()`,
	- `orL()`,
	- `nandL()`,
	- `ifL()`,
	- `norL()`,
	- `xorL()`,
	- `epqL()` (if and only if).

Auxiliary logical constraint methods:  
    - `eqL()` -  used in the path definition to filter instances with value for specified attribute in the provided set or equal to the provided value, e.g.:   
      _eqL(cityLink, 'neighbor', {True}) - instances of cityLink with attribute neighbor in the set containing only single value True,_   
    - `fixedL()`, used to fixed selected candidates to selected classification, e.g.:  
       _fixedL(empty_entry_label("x", eqL(empty_entry, "fixed", {True}))) - candidates for empty_entry_label which have attribute fixed* should have their classification fixed to empty_entry._  

##### Counting methods

DomiKnows also provides **counting methods** as an extension of logical connectives. 
Each counting method contains a list of predicates or nested logical constraints and optionally a number of required satisfied predicates.
If the number is not specified as the last argument in the counting method, then the default value of 1 is used.
There are two flavors of counting methods: one counting over candidates in the current context of constraint's evaluation and the other counting domain of discourse (it has 'A' suffix in the name - indicating accumulation). Four types of counting methods are implemented: **exists, exact, atLeast** and **atMost**.
 
Examples of counting methods usage in the logical constraint:  
    - _existsAL(firestationCity)_ - firestationCity exists in the domain of discourse,  
    - _exactL(firestationCity, 2)_ - there are exactly 2 firestationCity in the domain of discourse,  
    - _atLeastL(firestationCity, 4)_ - there are at least 4 firestationCity in the domain of discourse,  
    - _atMostL(andL(city('x'), firestationCity(path=('x', eqL(cityLink, 'neighbor', {True}), city2))), 4)_ - each city has no more then 4 *neighbors* which are firestationCity.  

#### Candidate Selection

The candidates for each predicate in the logical constraint are selected independently. By default, all candidates from the data are selected. However, this default can be modified by specifying a quantifier within the predicate.

If the predicates do not use a quantifier or if the quantifier in the predicate refers to the variable that defines the candidates for the previous predicates, there will be an equal number of candidates for each predicate in the logical constraint. This is typically the case with logical constraints.

However, if the predicates in the logical constraint employ disjointed quantifiers to select their candidates, the selected sets of candidates for each predicate can be different.

DomiKnows defined the mechanism allowing to define to specific candidate selection for the logical constraint. This is done by defining the new class by inheriting from `CandidateSelection` and overriding the `get_candidates` method.  

This example of new candidate selection class is `combinationC` which creates cartesian product of candidates for each concept in the selection. Here is the code example:

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
            combinationC(step, entity)('i', 'e'), #this is the search space, cartesian product is expected between options
            exactL(
                final_decision('x', path=(('i', rel_step.reversed), ('e', rel_entity.reversed))), 1
            ),
        )
  
  In this example the combination of all possible candidates for `step` and `entity` concepts is created and then returned as a dictionary with keys `i` and `e` respectively. This dictionary is then used to define the path for the `final_decision` concept. The forAllL constraint is then applied to all possible assignments of `i` and `e` to the path of `final_decision` concept.

  The example show the generic semantic of the `CandidateSelection` candidate selection class. It takes a list of concepts and returns a dictionary with keys corresponding to the concepts and values corresponding to the candidates for the concepts. The keys are provided as tuple behind the `CandidateSelection` call.

### Graph Constraints

The graph can also specify constraints:

- **Subclass relation between concepts**: e.g. `people = word(name='people')` is mapped to logical expression -

  *IF(people, word)*

- **Disjointing between concepts**: e.g. `disjoint(people, organization, location, other, o)` is mapped to logical expression -

  *atMostL(people, organization, location, other, o, 1)*
  
- **Domain and ranges for relations**: e.g. `work_for.has_a(people, organization)` is mapped to logical expressions-

  *ifL(work_for('x'), andL(people(path=('x', rel_pair_phrase1.name)), organization(path=('x', rel_pair_phrase2.name))))*

### Ontology Constraint

The OWL ontology, on which the learning system graph was build is loaded into the `ilpOntSolver` and parsed using Python OWL library [owlready2](https://pythonhosted.org/Owlready2/).

The OWL ontology language allows to specify constraints on [classes](https://www.w3.org/TR/owl2-syntax/#Classes "OWL Class") and [properties](https://www.w3.org/TR/owl2-syntax/#Object_Properties "OWL Property"). These classes and properties relate to concepts and relations which the learning system builds classification model for. The solver extracts these constraints.

This detail of mapping from OWL to logical representation is presented below for each OWL constraint.

**Constraints extracted from ontology [classes](https://www.w3.org/TR/owl2-syntax/#Classes "OWL Class") (*concepts*)**:

- **[disjoint](https://www.w3.org/TR/owl2-syntax/#Disjoint_Classes "OWL example of disjoint statement for classes")** statement between two classes *Concept1* and *Concept2* in ontology is mapped to equivalent logical expression -  
  
  *NAND(Concept1(token), Concept2(token))*

- **[equivalent](https://www.w3.org/TR/owl2-syntax/#Equivalent_Classes "OWL example of equivalent statement for classes")** statement between two classes *Concept1* and *Concept2* in ontology is mapped to equivalent logical expression -  
  
  *AND(Concept1(token), concept2(token))*

- **[subClassOf](https://www.w3.org/TR/owl2-syntax/#Subclass_Axioms "OWL example of subclass statement for classes")** statement between two classes *Concept1* and *SuperConcept2* in ontology is mapped to equivalent logical expression -  
  
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

  *This is an Existential constraint not possible to check without assumption of close world*
  
- **[hasValue](https://www.w3.org/TR/owl2-syntax/#Existential_Quantification "OWL example of hasValue statement for property")** statements statements for relation *P(token1, token2)* in ontology are mapped to equivalent logical expression -  

  *This is an Existential constraint not possible to check without assumption of close world*

- **[objectHasSelf](https://www.w3.org/TR/owl2-syntax/#Self-Restriction "OWL example of objectHasSelf statement for property")** statements for relation *P(token1, token2)* in ontology are mapped to equivalent logical expression -  

  *...*

- **[disjoint](https://www.w3.org/TR/owl2-syntax/#Disjoint_Object_Properties "OWL example of disjoint statement for properties")** statements for relations *P1(token1, token2)* and *P2(token1, token2)* in ontology are mapped to equivalent logical expression -  

  *NOT(IF(P1(token1, token2), P2(token1, token2)))*

- **[key](https://www.w3.org/TR/owl2-syntax/#Keys "OWL example of key statement for property")** statements for relation *P(token1, token2)*  in ontology are mapped to equivalent logical expression -  

  *...*

- **[exactCardinality](https://www.w3.org/TR/owl2-syntax/#Exact_Cardinality "OWL example of exactCardinality statement for property")** statements for relation *P(token1, token2)*  in ontology are mapped to equivalent logical expression -  

  *This is an Existential constraint not possible to check without assumption of close world*

- **[minCardinality](https://www.w3.org/TR/owl2-syntax/#Minimum_Cardinality "OWL example of minCardinality statement for property")** statements for relation *P(token1, token2)*  in ontology are mapped to equivalent logical expression -  

  *This is an Existential constraint not possible to check without assumption of close world*

- **[maxCardinality](https://www.w3.org/TR/owl2-syntax/#Maximum_Cardinality "OWL example of maxCardinality statement for property")** statements for relation *P(token1, token2)*  in ontology are mapped to equivalent logical expression -  

  *count of token2 for which P(token1, token2) <= maxCardinality*
  
- **[functional](https://www.w3.org/TR/owl2-syntax/#Functional_Object_Properties "OWL example of functional statement for properties")** relation *P(token1, token2)* statements in ontology are mapped to equivalent logical expression -  

  *Syntactic shortcut for the following maxCardinality of P(token1, token2) is 1*

- **[inverse functional](https://www.w3.org/TR/owl2-syntax/#Inverse-Functional_Object_Properties "OWL example of inverse functional statement for properties")** relation *P(token1, token2)* statements in ontology are mapped to equivalent logical expression -  

  *Syntactic shortcut for the following maxCardinality of inverse P(token2, token1) is 1*
