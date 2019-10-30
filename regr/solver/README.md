# ILP Solver based on constrains extracted from ontology

The OWL ontology, on which the learning system graph was build is loaded into the [ilpOntSolver](https://github.com/kordjamshidi/RelationalGraph/blob/master/regr/solver/ilpOntSolver.py) and parsed using python OWL library [owlready2](https://pythonhosted.org/Owlready2/). 

The OWL ontology language allows to specify constrains on [classes](https://www.w3.org/TR/owl2-syntax/#Classes "OWL Class") and [properties](https://www.w3.org/TR/owl2-syntax/#Object_Properties "OWL Property"). These classes and properties relate to concepts and relations which the learning system builds classification model for. The solver extracts these constrains.   

The [ilpBooleanMethods](https://github.com/kordjamshidi/RelationalGraph/blob/master/regr/solver/ilpBooleanMethods.py) module encodes basic logical expressions (*[AND](https://github.com/kordjamshidi/RelationalGraph/blob/5abe2795ca219c81ee8fb8d39ca294e2f0d7738c/regr/solver/ilpBooleanMethods.py#L48)*, *[OR](https://github.com/kordjamshidi/RelationalGraph/blob/5abe2795ca219c81ee8fb8d39ca294e2f0d7738c/regr/solver/ilpBooleanMethods.py#L102)*, *[IF](https://github.com/kordjamshidi/RelationalGraph/blob/5abe2795ca219c81ee8fb8d39ca294e2f0d7738c/regr/solver/ilpBooleanMethods.py#L264)*, *[NAND](https://github.com/kordjamshidi/RelationalGraph/blob/5abe2795ca219c81ee8fb8d39ca294e2f0d7738c/regr/solver/ilpBooleanMethods.py#L156)*, *[XOR](https://github.com/kordjamshidi/RelationalGraph/blob/5abe2795ca219c81ee8fb8d39ca294e2f0d7738c/regr/solver/ilpBooleanMethods.py#L245)*, *[EPQ](https://github.com/kordjamshidi/RelationalGraph/blob/5abe2795ca219c81ee8fb8d39ca294e2f0d7738c/regr/solver/ilpBooleanMethods.py#L281)*, *[NOR](https://github.com/kordjamshidi/RelationalGraph/blob/5abe2795ca219c81ee8fb8d39ca294e2f0d7738c/regr/solver/ilpBooleanMethods.py#L210)*, *[NOT](https://github.com/kordjamshidi/RelationalGraph/blob/5abe2795ca219c81ee8fb8d39ca294e2f0d7738c/regr/solver/ilpBooleanMethods.py#L16)*) into the ILP equations.  

The solver [implementation using Gurobi](https://github.com/kordjamshidi/RelationalGraph/blob/master/regr/solver/gurobiILPOntSolver.py) is called with probabilities for token classification obtained from learned model. The solver encodes mapping from OWL constrains to the appropriate equivalent logical expression for the given graph and the provided probabilities. 
The solver ILP model is solved by Gurobi and the found solutions for optimal classifction of tokens and relations is returned. 

This detail of mapping from OWL to logical representaion is presented below for each OWL constrain.
# Constrains extracted from ontology [classes](https://www.w3.org/TR/owl2-syntax/#Classes "OWL Class") (*concepts*):

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
        
##### No supported yet:

- **[disjonitUnion](https://www.w3.org/TR/owl2-syntax/#Disjoint_Union_of_Class_Expressions "OWL example of disjointUnion of classes")** statement between classes *Concept1*, *Concept2*, *Concept3*, ... is Not yet supported by owlready2 Python ontology parser used inside the solver

- **[oneOf](https://www.w3.org/TR/owl2-syntax/#Enumeration_of_Individuals "OWL example of enumeration of individuals for classes")** statements for a class *Concept* in ontology 
   
# Constrains extracted from ontology [properties](https://www.w3.org/TR/owl2-syntax/#Object_Properties "OWL Property") (*relations*)

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