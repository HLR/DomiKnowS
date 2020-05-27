# Knowledge Declaration

In knowledge declaration, the user defines a collection of concepts and the way they are related to each other, representing the domain knowledge a the task.
We provide a graph language based on python for knowledge declaration with notation of "graph", "concept", "property", "relation", and "constraints".

- [Knowledge Declaration](#knowledge-declaration)
  - [Class Overview](#class-overview)
    - [Graph classes](#graph-classes)
    - [Constraints classes](#constraints-classes)
  - [Graph](#graph)
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
* **Subclass relation between concepts**: e.g. people = word(name='people')
* **Disjointment between concepts**: e.g. disjoint(people, organization, location, other, o)
* **Domain and ranges for relations**: e.g. work_for.has_a(people, organization)

Additional, logical constrains defined within the graph can use the following logical functions to build logical expression: 
* **notL**, 
* **andL**, 
* **orL**, 
* **nandL**, 
* **existsL**, 
* **ifL**, 
* **equalA**, 
* **inSetA**.

The logical constrain can use variables to associate related objects of the logical expression. 
The expressions use concepts defined in the graph and set additional constrains on them. 
Example:

	ifL(work_for, ('x', 'y'), andL(people, ('x',), organization, ('y',)))
	
This above example logical constrain specify that: *if two object are linked by work_for relation that the first has to be of concept people and the second has to be of concept organization*.

The constrains are regular Python instructions thus they have to follow definition of tuple in Python.

### Ontology file as a source of constrains

**If ontology url is provided in the graph then only this ontology will be used to retrieved constrains by the ILP solver.**

The OWL ontology, on which the learning system graph was build is loaded into the [ilpOntSolver](https://github.com/kordjamshidi/RelationalGraph/blob/master/regr/solver/ilpOntSolver.py) and parsed using python OWL library [owlready2](https://pythonhosted.org/Owlready2/). 

The OWL ontology language allows to specify constrains on [classes](https://www.w3.org/TR/owl2-syntax/#Classes "OWL Class") and [properties](https://www.w3.org/TR/owl2-syntax/#Object_Properties "OWL Property"). These classes and properties relate to concepts and relations which the learning system builds classification model for. The solver extracts these constrains.   

The [ilpBooleanMethods](https://github.com/kordjamshidi/RelationalGraph/blob/master/regr/solver/ilpBooleanMethods.py) module encodes basic logical expressions (*[AND](https://github.com/kordjamshidi/RelationalGraph/blob/5abe2795ca219c81ee8fb8d39ca294e2f0d7738c/regr/solver/ilpBooleanMethods.py#L48)*, *[OR](https://github.com/kordjamshidi/RelationalGraph/blob/5abe2795ca219c81ee8fb8d39ca294e2f0d7738c/regr/solver/ilpBooleanMethods.py#L102)*, *[IF](https://github.com/kordjamshidi/RelationalGraph/blob/5abe2795ca219c81ee8fb8d39ca294e2f0d7738c/regr/solver/ilpBooleanMethods.py#L264)*, *[NAND](https://github.com/kordjamshidi/RelationalGraph/blob/5abe2795ca219c81ee8fb8d39ca294e2f0d7738c/regr/solver/ilpBooleanMethods.py#L156)*, *[XOR](https://github.com/kordjamshidi/RelationalGraph/blob/5abe2795ca219c81ee8fb8d39ca294e2f0d7738c/regr/solver/ilpBooleanMethods.py#L245)*, *[EPQ](https://github.com/kordjamshidi/RelationalGraph/blob/5abe2795ca219c81ee8fb8d39ca294e2f0d7738c/regr/solver/ilpBooleanMethods.py#L281)*, *[NOR](https://github.com/kordjamshidi/RelationalGraph/blob/5abe2795ca219c81ee8fb8d39ca294e2f0d7738c/regr/solver/ilpBooleanMethods.py#L210)*, *[NOT](https://github.com/kordjamshidi/RelationalGraph/blob/5abe2795ca219c81ee8fb8d39ca294e2f0d7738c/regr/solver/ilpBooleanMethods.py#L16)*) into the ILP equations.  

The solver [implementation using Gurobi](https://github.com/kordjamshidi/RelationalGraph/blob/master/regr/solver/gurobiILPOntSolver.py) is called with probabilities for token classification obtained from learned model. The solver encodes mapping from OWL constrains to the appropriate equivalent logical expression for the given graph and the provided probabilities. 
The solver ILP model is solved by Gurobi and the found solutions for optimal classification of tokens and relations is returned. 

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

  *This is an Existential constrain not possible to check without assumption of close world *
  
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
