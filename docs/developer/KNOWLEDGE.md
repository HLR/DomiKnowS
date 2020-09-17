# Knowledge Declaration

In knowledge declaration, the user defines a collection of concepts and the way they are related to each other, representing the domain knowledge a the task.
We provide a graph language based on python for knowledge declaration with notation of "graph", "concept", "property", "relation", and "constraints".

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
    - [Graph with logical Constrains](#graph-with-logical-constrains)
    - [Ontology file as a source of constrains](#ontology-file-as-a-source-of-constrains)
      - [No supported yet](#no-supported-yet)

## Class Overview

### Graph classes

- Package `regr.graph`: the set of class for above-mentioned notations as well as some base classes.
- Class `Graph`: a basic container for other components. It can contain sub-graphs for flexible modeling.
- Class `Concept`: a collection of `Property`s that can be related with each other by `Relation`s. It is a none leaf node in the `Graph`.
- Class `Property`: a key attached to a `Concept` that can be associated with certain value assigned by a sensor or a learner. It is a leaf node in the `Graph`.
- Class `Relation`: a relation between two `Concept`s. It is an edge in the `Graph`.

### Constraints classes

- Package `regr.graph.logicalConstrain`: a set of functions with logical symantics, that one can express logical constraints in first order logic.
- Function `*L()`: functions based on logical notations. Linear constraints can be generated based on the locigal constraints. Some of these functions are `ifL()`, `notL()`, `andL()`, `orL()`, `nandL()`, `existL()`, `equalL()`, `inSetL()`, etc.

## Graph

`Graph` instances are basic container of the `Concept`s, `Relation`s, constaints and other instances in the framework.
A `Graph` object is constructed either by manually coding or compiled from `OWL` (deprecated).
Each `Graph` object can contain other `Graph` objects as sub-graphs. No cyclic reference in graph hierarchy is allowed.

You can either write an owl file initializing your concepts and relations or to write your graph with our specific python classes.

Each `Graph` object can contain `Concept`s.

The graph is a partial program, and there is no sensor or learner, which are data processing units, connected. There is no behavior associated. It is only a data structure to express domain knowledge.

###Relation Types
We have three defined relationship between nodes that each program can use. `contains`, `has_a`, and `equal` are used to define relations between concepts. 
`contains` means that concept `A` is the parent node of concept `B` and several `B` instances can be the children of one single node `A`.
we use the relationship `has_a` only to define relations between concepts and to produce candidates of a relationship. For instance, a relationship between `word` and `word` can be defined using an intermediate concept `pair` and two `has_a` relation links.
```python
pair = Concept("pair")
pair.has_a(arg1=word, arg2=word)
```
This means that the candidates of a `pair` concept are generated based on a `word_{i}` and a `word_{j}`.
the `equal` relation establishes an equality between two different concept. for instance, if you have two different tokenizers and you want to use features from one of them into another, you have to establish an `equal` edge between the concepts holding those tokenizer instances.
```python
word = Concept("word")
word1 = Concept("word")
word.equal(word1)
```
This edge enables us to transfer properties of concepts between instances that are marked as equal.

Using each of these relation edges requires us to assign a sensor to them in the model execution. The sensors input is selected properties from the source concept and the output will be stored as selected properties in the destination node.

### Example

The following snippest shows an example of a `Graph`.

```python
with Graph() as graph:
    word = Concept('word')
    pair = Concept(word, word)
    with Graph('sub') as sub_graph:
        people = word('people')
        organization = word('organization')
        work_for = pair(people, organization)
```

#### Graph declaration and `with` statement

The first `with` statement creates a graph, assigns it to python variable `graph`, and declares that anything below are attached to the graph.

The second `with` statement declare another graph with an explicit name `'sub'`. It will also be attached to the enclosing graph, and become a subgraph. However, everything below this point will be attached to the subgraph, instead of the first graph.

#### Concept declaration

##### Direct declaration

`word = Concept('word')` creates a concept with name `'word'` (implicitly attached to the enclosing graph) and assign it to python variable `word`.

`pair = Concept(word, word)` is syntactic sugar to creates a concept with two `word`s being its arguments (with two `HasA` relations).
It does not has an explicit name. If a explicit name is designable, use keyword argument `name=`. For example, `pair = Concept(word, word, name='pair')`.
It will also be attached to the enclosing graph.
A `HasA` relation will be added between the new concept and each argument concept. That implies, two `HasA` concepts will be created from `pair` to `word`.
It is equivalent to the following statements:

```python
pair = Concept(name='pair')
pair.has_a(word)
pair.has_a(word)
```

##### Inherit declaration

`people = word('people')` and `organization = word('organization')` create another two concepts extending `word`, setting name `'people'` and `'organization'`, and assign them to python variable `people` and `organization`. They are attached to enclosing subgraph `sub_graph`.
Inherit declaration is syntactic sugar for creating concepts and `IsA` relations.
An `IsA` relation will be create for each of these statements. It is equivalent to the following statements:

```python
people = Concept('people')
people.is_a(word)
organization = Concept('organization')
organization.is_a(word)
```

### Access the nodes

 All the sub-`Graph`s and `Concept` instances can be retrieved from a graph (or sub-graph) with a (relative) pathname.
 For example, to retieve `people` from the above example, one can do `graph['sub/people']` or `sub_graph['people']`.

## Constraints

The ILP constrains could be specified in the **ontology graph itself with defined logical constrains** or in the **ontology (in OWL file)** provided as url in the ontology graph.

### Graph with logical Constrains

**If ontology url is not provided in the graph then the graph defined constrains and logical constrains will be retrieved by the ILP solver.**

The graph can specify constrains:

- **Subclass relation between concepts**: e.g. `people = word(name='people')` is mapped to logical expression -

  *IF(people(token), word(token))*

- **Disjointment between concepts**: e.g. `disjoint(people, organization, location, other, o)` is mapped to logical expression -

  *NAND(people(token), organization(token), location(token), other(token), o(token))*
  
- **Domain and ranges for relations**: e.g. `work_for.has_a(people, organization)` is mapped to logical expressions-

  *IF(work_for(token1, token2), people(token1))*

  *IF(work_for(token1, token2), organization(token1))*


Additional, logical constrains defined within the graph can use the following logical functions to build logical expression:

- `notL()`,
- `andL()`,
- `orL()`,
- `nandL()`,
- `ifL()`,
- `norL()`,
- `xorL()`,
- `epqL()`,
- `eqL()`, -  used to select, for the logical constrain, instances with value for specified attribute in the provided set or equal to the provided value, e.g.: 

*eqL(cityLink, 'neighbor', {1}* - instances of *cityLink* with attribute *neighbor* in the set containing only single value 1,

- `existsL()`, e.g.:

*existsL(('y',), andL(eqL(cityLink, 'neighbor', {1}), ('x', 'y'), firestationCity, ('y',)))* - exists a city *y* which is in *cityLink* relation with *neighbor* attribute equal 1 to city x and y is a *firestationCity*,

- `exactL()`, e.g.:

*exactL(2, ('x', ), firestationCity, ('x', ))* - exists exactly 2 *firestationCity*,

- `atLeastL()`, e.g.:

*atLeastL(1, ('x', ), firestationCity, ('x', ))* - exists at least 1 *firestationCity*,

- `atMostL()`,  e.g.:

*atMostL(4, ('y', ), andL(firestationCity, ('x',), eqL(cityLink, 'neighbor', 1), ('x', 'y')))* - Each city has no more then 4 *neighbors*.

The logical constrain can use variables to associate related objects of the logical expression.
The expressions use concepts defined in the graph and set additional constrains on them.
Example:

```python
ifL(work_for, ('x', 'y'), andL(people, ('x',), organization, ('y',)))
```

This above example logical constrain specify that: *if two object are linked by work_for relation that the first has to be of concept people and the second has to be of concept organization*.

The constrains are regular Python instructions thus they have to follow definition of tuple in Python.

Backus�Naur form of the Logical Constrain:

	<logicalConstrain> ::= <notLogicalConstrain> | <2ElementsLogicalConstrain> | <multiElementsLogicalConstrain> | <countConstrain> | <existsConstrain>

	<notLogicalConstrain> ::= <notOperator> "(" <opt_whitespace> <logicalConstrainElement> <variables> <p> ")"
	<2ElementsLogicalConstrain> ::= <2ElementsOperator> "(" <opt_whitespace> <logicalConstrainElement> <variables> "," <opt_whitespace> <logicalConstrainElement> <variables> <p> ")"
	<multiElementsLogicalConstrain> ::= <multiElementsOperator> "(" <opt_whitespace> <logicalConstrainElements> <p> ")"
	<countConstrain> ::= <countOperator> "(" <opt_whitespace> <number> "," <opt_whitespace> <variablesTuple> <opt_whitespace> "," <opt_whitespace> <logicalConstrainElement> <variables> <p> ")" 
	<existsConstrain> ::= <existsOperator> "(" <opt_whitespace> <variablesTuple> <opt_whitespace> "," <opt_whitespace> <logicalConstrainElement> <variables> <p> ")"
	
	<notOperator> ::= "notL"
	<eqOperator> ::=  "eqL"
	<2ElementsOperator> ::= "ifL" | "epqL" | "xorL"
	<multiElementsOperator> ::= "andL" | "orL" | "nandL" | "norL"
	<countOperator> ::= "exactL" | "atLeastL" | "atMostL"
	<existsOperator> ::= "existsL" 
	<eqElement> ::= <eqOperator> "(" <conceptOrRelation> "," <opt_whitespace> "'" <attributeName> "'" "," <opt_whitespace> <eqValue> ")"	
	<conceptOrRelation> ::= <name>
	<logicalConstrainElement> ::= <conceptOrRelation> | <eqElement> | <logicalConstrain>
	<logicalConstrainElements> ::= <logicalConstrainElement> <variables> | <logicalConstrainElement> <variables> "," <opt_whitespace> <logicalConstrainElements>
	
	<variables> ::= <opt_whitespace> "," <opt_whitespace> <variablesTuple> <opt_whitespace> | <opt_whitespace>
	<variablesTuple> ::= "(" <opt_whitespace> <variablesTupleElement> <opt_whitespace> ")"
    <variablesTupleElement> ::= <variablesTupleElementHead> <variablesTupleElementRest>
	<variablesTupleElementHead> ::= "'" <loLetter> "'" 
	<variablesTupleElementRest> ::= <opt_whitespace> "," <opt_whitespace> | <variablesTupleRestElements>
    <variablesTupleRestElements> ::= <opt_whitespace> "," <opt_whitespace> "'" <loLetter> "'" | "," <opt_whitespace> "'" <loLetter> "'" <opt_whitespace> <variablesTupleRestElements>
    
	<p> ::= <opt_whitespace> "," <opt_whitespace> "p" <opt_whitespace> "=" <opt_whitespace> <digit> <digit> <opt_whitespace> | <opt_whitespace>
	<attributeName> ::= <name>
	<eqValue> ::= <value> | "{" <values> "}"
	<value> ::= <number> | <name>
	<values> ::= <value> | <value> "," <values>
	<number> ::= <digit> | <digit> <number>
	<name> ::= <character> | <character> <name>
	<opt_whitespace> ::= " "*	
	<digit> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
	<character> ::= <letter> | <digit> | <symbol>
	<letter> ::= <upLetter> | <loLetter>
	<upLetter> ::= "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" | "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R" | "S" | "T" | "U" | "V" | "W" | "X" | "Y" | "Z"
	<loLetter> ::=  "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z"
	<symbol> ::= "_" | "-"
   
### Ontology file as a source of constrains

**If ontology url is provided in the graph then only this ontology will be used to retrieved constrains by the ILP solver.**

The OWL ontology, on which the learning system graph was build is loaded into the `ilpOntSolver` and parsed using python OWL library [owlready2](https://pythonhosted.org/Owlready2/).

The OWL ontology language allows to specify constrains on [classes](https://www.w3.org/TR/owl2-syntax/#Classes "OWL Class") and [properties](https://www.w3.org/TR/owl2-syntax/#Object_Properties "OWL Property"). These classes and properties relate to concepts and relations which the learning system builds classification model for. The solver extracts these constrains.

This detail of mapping from OWL to logical representation is presented below for each OWL constrain.

**Constrains extracted from ontology [classes](https://www.w3.org/TR/owl2-syntax/#Classes "OWL Class") (*concepts*)**:

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

#### No supported yet

- **[disjonitUnion](https://www.w3.org/TR/owl2-syntax/#Disjoint_Union_of_Class_Expressions "OWL example of disjointUnion of classes")** statement between classes *Concept1*, *Concept2*, *Concept3*, ... is Not yet supported by owlready2 Python ontology parser used inside the solver

- **[oneOf](https://www.w3.org/TR/owl2-syntax/#Enumeration_of_Individuals "OWL example of enumeration of individuals for classes")** statements for a class *Concept* in ontology

**Constrains extracted from ontology [properties](https://www.w3.org/TR/owl2-syntax/#Object_Properties "OWL Property") (*relations*)**

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

  *This is an Existential constrain not possible to check without assumption of close world*
  
- **[hasValue](https://www.w3.org/TR/owl2-syntax/#Existential_Quantification "OWL example of hasValue statement for property")** statements statements for relation *P(token1, token2)* in ontology are mapped to equivalent logical expression -  

  *This is an Existential constrain not possible to check without assumption of close world*

- **[objectHasSelf](https://www.w3.org/TR/owl2-syntax/#Self-Restriction "OWL example of objectHasSelf statement for property")** statements for relation *P(token1, token2)* in ontology are mapped to equivalent logical expression -  

  *...*

- **[disjoint](https://www.w3.org/TR/owl2-syntax/#Disjoint_Object_Properties "OWL example of disjoint statement for properties")** statements for relations *P1(token1, token2)* and *P2(token1, token2)* in ontology are mapped to equivalent logical expression -  

  *NOT(IF(P1(token1, token2), P2(token1, token2)))*

- **[key](https://www.w3.org/TR/owl2-syntax/#Keys "OWL example of key statement for property")** statements for relation *P(token1, token2)*  in ontology are mapped to equivalent logical expression -  

  *...*

- **[exactCardinality](https://www.w3.org/TR/owl2-syntax/#Exact_Cardinality "OWL example of exactCardinality statement for property")** statements for relation *P(token1, token2)*  in ontology are mapped to equivalent logical expression -  

  *This is an Existential constrain not possible to check without assumption of close world*

- **[minCardinality](https://www.w3.org/TR/owl2-syntax/#Minimum_Cardinality "OWL example of minCardinality statement for property")** statements for relation *P(token1, token2)*  in ontology are mapped to equivalent logical expression -  

  *This is an Existential constrain not possible to check without assumption of close world*

- **[maxCardinality](https://www.w3.org/TR/owl2-syntax/#Maximum_Cardinality "OWL example of maxCardinality statement for property")** statements for relation *P(token1, token2)*  in ontology are mapped to equivalent logical expression -  

  *count of token2 for which P(token1, token2) <= maxCardinality*
  
- **[functional](https://www.w3.org/TR/owl2-syntax/#Functional_Object_Properties "OWL example of functional statement for properties")** relation *P(token1, token2)* statements in ontology are mapped to equivalent logical expression -  

  *Syntactic shortcut for the following maxCardinality of P(token1, token2) is 1*

- **[inverse functional](https://www.w3.org/TR/owl2-syntax/#Inverse-Functional_Object_Properties "OWL example of inverse functional statement for properties")** relation *P(token1, token2)* statements in ontology are mapped to equivalent logical expression -  

  *Syntactic shortcut for the following maxCardinality of inverse P(token2, token1) is 1*
