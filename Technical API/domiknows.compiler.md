# domiknows.compiler package

## Submodules

## domiknows.compiler.OntologyMLGraph module

### *class* domiknows.compiler.OntologyMLGraph.OntologyMLGraphCreator(name)

Bases: `object`

Class building graph based on ontology

#### buildGraph(ontology, fileName='graph.py', ontologyPathname='./')

#### buildSubGraph(ontology, myOnto, graphRootClass, tabSize, graphFile)

#### conceptTemplate *= <string.Template object>*

#### graphHeaderTemplate *= <string.Template object>*

#### graphImportTemlate *= <string.Template object>*

#### graphOntologyTemplate *= <string.Template object>*

#### loadOntology(ontologyURL, ontologyPathname='./')

#### parseSubGraphOntology(ontConceptClass, subGraphConcepts, tabSize, graphFile)

#### relationTemplate *= <string.Template object>*

#### subclassTemplate *= <string.Template object>*

### domiknows.compiler.OntologyMLGraph.main()

## domiknows.compiler.compiler module

### *class* domiknows.compiler.compiler.Compiler

Bases: `object`

#### *abstract* compile(src: str, dst: str)

## Module contents
