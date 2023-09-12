# domiknows.graph package

## Subpackages

* [domiknows.graph.allennlp package](domiknows.graph.allennlp.md)
  * [Submodules](domiknows.graph.allennlp.md#submodules)
  * [domiknows.graph.allennlp.base module](domiknows.graph.allennlp.md#domiknows-graph-allennlp-base-module)
  * [domiknows.graph.allennlp.metrics module](domiknows.graph.allennlp.md#domiknows-graph-allennlp-metrics-module)
  * [domiknows.graph.allennlp.model module](domiknows.graph.allennlp.md#domiknows-graph-allennlp-model-module)
  * [domiknows.graph.allennlp.utils module](domiknows.graph.allennlp.md#domiknows-graph-allennlp-utils-module)
  * [Module contents](domiknows.graph.allennlp.md#module-contents)

## Submodules

## domiknows.graph.base module

## domiknows.graph.candidates module

## domiknows.graph.concept module

### *class* domiknows.graph.concept.Concept(name=None, batch=False)

Bases: `BaseGraphTree`

#### aggregate(vals, confs)

The aggregation used in this concept to reduce the inconsistent values.

#### bvals(prop)

Properties: get all binded values

* **Parameters:**
  **prop** (*str*) – property name
* **Returns:**
  Return vals and confs where vals is a list of values binded to the property
  and confs is a list of values representing the confidence of each binded value.
  An element of vals should have the shape:
  ( batch, vdim(s…) )
  Return None is if never binded to this property.
* **Return type:**
  [barray,…], [barray,…]

#### candidates(root_data, query=None, logger=None)

#### distances(p, q)

The “distance” used in this concept to measure the consistency.
Lower value indicates better consistency.
Feature(s) of one instance is always on only the last axis.
p, q - [(batch, vdim(s…)),…] \* nval

#### getOntologyGraph()

#### get_batch()

#### get_multiassign()

#### processLCArgs(\*args, conceptT=None, \*\*kwargs)

#### relate_to(concept, \*tests)

#### *classmethod* relation_type(name=None)

#### rvals(prop, hops=1)

Properties: get all values from relations

#### *property* scope_key

#### set_apply(name, sub)

#### vals(prop, hops=1)

#### what()

### *class* domiknows.graph.concept.EnumConcept(name=None, values=[])

Bases: [`Concept`](#domiknows.graph.concept.Concept)

#### *property* attributes

#### *property* enum

#### get_concept(value)

#### get_index(value)

#### get_value(index)

## domiknows.graph.dataNode module

## domiknows.graph.dataNodeConfig module

## domiknows.graph.graph module

### *class* domiknows.graph.graph.Graph(name=None, ontology=None, iri=None, local=None, auto_constraint=None, reuse_model=False)

Bases: `BaseGraphTree`

#### *class* Ontology(iri, local)

Bases: `tuple`

#### iri

Alias for field number 0

#### local

Alias for field number 1

#### *property* auto_constraint

#### *property* batch

#### check_if_all_used_variables_are_defined(lc, found_variables, used_variables=None, headLc=None)

#### check_path(path, variableConceptParent, lc_name, foundVariables, variableName)

#### *property* concepts

#### findRootConceptOrRelation(relationConcept)

#### find_lc_variable(lc, found_variables=None, headLc=None)

#### getPathStr(path)

#### get_apply(name)

#### get_properties(\*tests)

#### get_sensors(\*tests)

#### *property* logicalConstrains

#### *property* logicalConstrainsRecursive

#### namedtuple(field_names, \*, rename=False, defaults=None, module=None)

Returns a new subclass of tuple with named fields.

```pycon
>>> Point = namedtuple('Point', ['x', 'y'])
>>> Point.__doc__                   # docstring for the new class
'Point(x, y)'
>>> p = Point(11, y=22)             # instantiate with positional args or keywords
>>> p[0] + p[1]                     # indexable like a plain tuple
33
>>> x, y = p                        # unpack like a regular tuple
>>> x, y
(11, 22)
>>> p.x + p.y                       # fields also accessible by name
33
>>> d = p._asdict()                 # convert to a dictionary
>>> d['x']
11
>>> Point(**d)                      # convert from a dictionary
Point(x=11, y=22)
>>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
Point(x=100, y=22)
```

#### *property* ontology

#### *property* relations

#### set_apply(name, sub)

#### *property* subgraphs

#### visualize(filename, open_image=False)

#### what()

## domiknows.graph.logicalConstrain module

## domiknows.graph.property module

### *class* domiknows.graph.property.Property(prop_name)

Bases: `BaseGraphShallowTree`

#### attach(sub)

#### attach_to_context(name=None)

#### find(\*sensor_tests)

#### get_fullname(delim='/')

## domiknows.graph.relation module

## domiknows.graph.trial module

## Module contents
