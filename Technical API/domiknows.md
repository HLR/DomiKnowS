# domiknows package

## Subpackages

* [domiknows.compiler package](domiknows.compiler.md)
  * [Submodules](domiknows.compiler.md#submodules)
  * [domiknows.compiler.OntologyMLGraph module](domiknows.compiler.md#module-domiknows.compiler.OntologyMLGraph)
    * [`OntologyMLGraphCreator`](domiknows.compiler.md#domiknows.compiler.OntologyMLGraph.OntologyMLGraphCreator)
      * [`OntologyMLGraphCreator.buildGraph()`](domiknows.compiler.md#domiknows.compiler.OntologyMLGraph.OntologyMLGraphCreator.buildGraph)
      * [`OntologyMLGraphCreator.buildSubGraph()`](domiknows.compiler.md#domiknows.compiler.OntologyMLGraph.OntologyMLGraphCreator.buildSubGraph)
      * [`OntologyMLGraphCreator.conceptTemplate`](domiknows.compiler.md#domiknows.compiler.OntologyMLGraph.OntologyMLGraphCreator.conceptTemplate)
      * [`OntologyMLGraphCreator.graphHeaderTemplate`](domiknows.compiler.md#domiknows.compiler.OntologyMLGraph.OntologyMLGraphCreator.graphHeaderTemplate)
      * [`OntologyMLGraphCreator.graphImportTemlate`](domiknows.compiler.md#domiknows.compiler.OntologyMLGraph.OntologyMLGraphCreator.graphImportTemlate)
      * [`OntologyMLGraphCreator.graphOntologyTemplate`](domiknows.compiler.md#domiknows.compiler.OntologyMLGraph.OntologyMLGraphCreator.graphOntologyTemplate)
      * [`OntologyMLGraphCreator.loadOntology()`](domiknows.compiler.md#domiknows.compiler.OntologyMLGraph.OntologyMLGraphCreator.loadOntology)
      * [`OntologyMLGraphCreator.parseSubGraphOntology()`](domiknows.compiler.md#domiknows.compiler.OntologyMLGraph.OntologyMLGraphCreator.parseSubGraphOntology)
      * [`OntologyMLGraphCreator.relationTemplate`](domiknows.compiler.md#domiknows.compiler.OntologyMLGraph.OntologyMLGraphCreator.relationTemplate)
      * [`OntologyMLGraphCreator.subclassTemplate`](domiknows.compiler.md#domiknows.compiler.OntologyMLGraph.OntologyMLGraphCreator.subclassTemplate)
    * [`main()`](domiknows.compiler.md#domiknows.compiler.OntologyMLGraph.main)
  * [domiknows.compiler.compiler module](domiknows.compiler.md#module-domiknows.compiler.compiler)
    * [`Compiler`](domiknows.compiler.md#domiknows.compiler.compiler.Compiler)
      * [`Compiler.compile()`](domiknows.compiler.md#domiknows.compiler.compiler.Compiler.compile)
  * [Module contents](domiknows.compiler.md#module-domiknows.compiler)
* [domiknows.data package](domiknows.data.md)
  * [Subpackages](domiknows.data.md#subpackages)
    * [domiknows.data.allennlp package](domiknows.data.allennlp.md)
      * [Submodules](domiknows.data.allennlp.md#submodules)
      * [domiknows.data.allennlp.reader module](domiknows.data.allennlp.md#domiknows-data-allennlp-reader-module)
      * [Module contents](domiknows.data.allennlp.md#module-domiknows.data.allennlp)
  * [Submodules](domiknows.data.md#submodules)
  * [domiknows.data.reader module](domiknows.data.md#module-domiknows.data.reader)
    * [`RegrReader`](domiknows.data.md#domiknows.data.reader.RegrReader)
      * [`RegrReader.make_object()`](domiknows.data.md#domiknows.data.reader.RegrReader.make_object)
      * [`RegrReader.parse_file()`](domiknows.data.md#domiknows.data.reader.RegrReader.parse_file)
      * [`RegrReader.run()`](domiknows.data.md#domiknows.data.reader.RegrReader.run)
  * [Module contents](domiknows.data.md#module-domiknows.data)
* [domiknows.graph package](domiknows.graph.md)
  * [Subpackages](domiknows.graph.md#subpackages)
    * [domiknows.graph.allennlp package](domiknows.graph.allennlp.md)
      * [Submodules](domiknows.graph.allennlp.md#submodules)
      * [domiknows.graph.allennlp.base module](domiknows.graph.allennlp.md#domiknows-graph-allennlp-base-module)
      * [domiknows.graph.allennlp.metrics module](domiknows.graph.allennlp.md#domiknows-graph-allennlp-metrics-module)
      * [domiknows.graph.allennlp.model module](domiknows.graph.allennlp.md#domiknows-graph-allennlp-model-module)
      * [domiknows.graph.allennlp.utils module](domiknows.graph.allennlp.md#domiknows-graph-allennlp-utils-module)
      * [Module contents](domiknows.graph.allennlp.md#module-contents)
  * [Submodules](domiknows.graph.md#submodules)
  * [domiknows.graph.base module](domiknows.graph.md#domiknows-graph-base-module)
  * [domiknows.graph.candidates module](domiknows.graph.md#domiknows-graph-candidates-module)
  * [domiknows.graph.concept module](domiknows.graph.md#module-domiknows.graph.concept)
    * [`Concept`](domiknows.graph.md#domiknows.graph.concept.Concept)
      * [`Concept.aggregate()`](domiknows.graph.md#domiknows.graph.concept.Concept.aggregate)
      * [`Concept.bvals()`](domiknows.graph.md#domiknows.graph.concept.Concept.bvals)
      * [`Concept.candidates()`](domiknows.graph.md#domiknows.graph.concept.Concept.candidates)
      * [`Concept.distances()`](domiknows.graph.md#domiknows.graph.concept.Concept.distances)
      * [`Concept.getOntologyGraph()`](domiknows.graph.md#domiknows.graph.concept.Concept.getOntologyGraph)
      * [`Concept.get_batch()`](domiknows.graph.md#domiknows.graph.concept.Concept.get_batch)
      * [`Concept.get_multiassign()`](domiknows.graph.md#domiknows.graph.concept.Concept.get_multiassign)
      * [`Concept.processLCArgs()`](domiknows.graph.md#domiknows.graph.concept.Concept.processLCArgs)
      * [`Concept.relate_to()`](domiknows.graph.md#domiknows.graph.concept.Concept.relate_to)
      * [`Concept.relation_type()`](domiknows.graph.md#domiknows.graph.concept.Concept.relation_type)
      * [`Concept.rvals()`](domiknows.graph.md#domiknows.graph.concept.Concept.rvals)
      * [`Concept.scope_key`](domiknows.graph.md#domiknows.graph.concept.Concept.scope_key)
      * [`Concept.set_apply()`](domiknows.graph.md#domiknows.graph.concept.Concept.set_apply)
      * [`Concept.vals()`](domiknows.graph.md#domiknows.graph.concept.Concept.vals)
      * [`Concept.what()`](domiknows.graph.md#domiknows.graph.concept.Concept.what)
    * [`EnumConcept`](domiknows.graph.md#domiknows.graph.concept.EnumConcept)
      * [`EnumConcept.attributes`](domiknows.graph.md#domiknows.graph.concept.EnumConcept.attributes)
      * [`EnumConcept.enum`](domiknows.graph.md#domiknows.graph.concept.EnumConcept.enum)
      * [`EnumConcept.get_concept()`](domiknows.graph.md#domiknows.graph.concept.EnumConcept.get_concept)
      * [`EnumConcept.get_index()`](domiknows.graph.md#domiknows.graph.concept.EnumConcept.get_index)
      * [`EnumConcept.get_value()`](domiknows.graph.md#domiknows.graph.concept.EnumConcept.get_value)
  * [domiknows.graph.dataNode module](domiknows.graph.md#domiknows-graph-datanode-module)
  * [domiknows.graph.dataNodeConfig module](domiknows.graph.md#domiknows-graph-datanodeconfig-module)
  * [domiknows.graph.graph module](domiknows.graph.md#module-domiknows.graph.graph)
    * [`Graph`](domiknows.graph.md#domiknows.graph.graph.Graph)
      * [`Graph.Ontology`](domiknows.graph.md#domiknows.graph.graph.Graph.Ontology)
      * [`Graph.auto_constraint`](domiknows.graph.md#domiknows.graph.graph.Graph.auto_constraint)
      * [`Graph.batch`](domiknows.graph.md#domiknows.graph.graph.Graph.batch)
      * [`Graph.check_if_all_used_variables_are_defined()`](domiknows.graph.md#domiknows.graph.graph.Graph.check_if_all_used_variables_are_defined)
      * [`Graph.check_path()`](domiknows.graph.md#domiknows.graph.graph.Graph.check_path)
      * [`Graph.concepts`](domiknows.graph.md#domiknows.graph.graph.Graph.concepts)
      * [`Graph.findRootConceptOrRelation()`](domiknows.graph.md#domiknows.graph.graph.Graph.findRootConceptOrRelation)
      * [`Graph.find_lc_variable()`](domiknows.graph.md#domiknows.graph.graph.Graph.find_lc_variable)
      * [`Graph.getPathStr()`](domiknows.graph.md#domiknows.graph.graph.Graph.getPathStr)
      * [`Graph.get_apply()`](domiknows.graph.md#domiknows.graph.graph.Graph.get_apply)
      * [`Graph.get_properties()`](domiknows.graph.md#domiknows.graph.graph.Graph.get_properties)
      * [`Graph.get_sensors()`](domiknows.graph.md#domiknows.graph.graph.Graph.get_sensors)
      * [`Graph.logicalConstrains`](domiknows.graph.md#domiknows.graph.graph.Graph.logicalConstrains)
      * [`Graph.logicalConstrainsRecursive`](domiknows.graph.md#domiknows.graph.graph.Graph.logicalConstrainsRecursive)
      * [`Graph.namedtuple()`](domiknows.graph.md#domiknows.graph.graph.Graph.namedtuple)
      * [`Graph.ontology`](domiknows.graph.md#domiknows.graph.graph.Graph.ontology)
      * [`Graph.relations`](domiknows.graph.md#domiknows.graph.graph.Graph.relations)
      * [`Graph.set_apply()`](domiknows.graph.md#domiknows.graph.graph.Graph.set_apply)
      * [`Graph.subgraphs`](domiknows.graph.md#domiknows.graph.graph.Graph.subgraphs)
      * [`Graph.visualize()`](domiknows.graph.md#domiknows.graph.graph.Graph.visualize)
      * [`Graph.what()`](domiknows.graph.md#domiknows.graph.graph.Graph.what)
  * [domiknows.graph.logicalConstrain module](domiknows.graph.md#domiknows-graph-logicalconstrain-module)
  * [domiknows.graph.property module](domiknows.graph.md#module-domiknows.graph.property)
    * [`Property`](domiknows.graph.md#domiknows.graph.property.Property)
      * [`Property.attach()`](domiknows.graph.md#domiknows.graph.property.Property.attach)
      * [`Property.attach_to_context()`](domiknows.graph.md#domiknows.graph.property.Property.attach_to_context)
      * [`Property.find()`](domiknows.graph.md#domiknows.graph.property.Property.find)
      * [`Property.get_fullname()`](domiknows.graph.md#domiknows.graph.property.Property.get_fullname)
  * [domiknows.graph.relation module](domiknows.graph.md#domiknows-graph-relation-module)
  * [domiknows.graph.trial module](domiknows.graph.md#domiknows-graph-trial-module)
  * [Module contents](domiknows.graph.md#module-contents)
* [domiknows.program package](domiknows.program.md)
  * [Subpackages](domiknows.program.md#subpackages)
    * [domiknows.program.model package](domiknows.program.model.md)
      * [Submodules](domiknows.program.model.md#submodules)
      * [domiknows.program.model.base module](domiknows.program.model.md#domiknows-program-model-base-module)
      * [domiknows.program.model.gbi module](domiknows.program.model.md#domiknows-program-model-gbi-module)
      * [domiknows.program.model.ilpu module](domiknows.program.model.md#domiknows-program-model-ilpu-module)
      * [domiknows.program.model.iml module](domiknows.program.model.md#domiknows-program-model-iml-module)
      * [domiknows.program.model.lossModel module](domiknows.program.model.md#domiknows-program-model-lossmodel-module)
      * [domiknows.program.model.pytorch module](domiknows.program.model.md#domiknows-program-model-pytorch-module)
      * [domiknows.program.model.torch module](domiknows.program.model.md#domiknows-program-model-torch-module)
      * [Module contents](domiknows.program.model.md#module-contents)
  * [Submodules](domiknows.program.md#submodules)
  * [domiknows.program.batchprogram module](domiknows.program.md#domiknows-program-batchprogram-module)
  * [domiknows.program.callbackprogram module](domiknows.program.md#domiknows-program-callbackprogram-module)
  * [domiknows.program.loss module](domiknows.program.md#domiknows-program-loss-module)
  * [domiknows.program.lossprogram module](domiknows.program.md#domiknows-program-lossprogram-module)
  * [domiknows.program.metric module](domiknows.program.md#domiknows-program-metric-module)
  * [domiknows.program.model_program module](domiknows.program.md#domiknows-program-model-program-module)
  * [domiknows.program.program module](domiknows.program.md#domiknows-program-program-module)
  * [domiknows.program.tracker module](domiknows.program.md#domiknows-program-tracker-module)
  * [Module contents](domiknows.program.md#module-contents)
* [domiknows.sensor package](domiknows.sensor.md)
  * [Subpackages](domiknows.sensor.md#subpackages)
    * [domiknows.sensor.allennlp package](domiknows.sensor.allennlp.md)
      * [Submodules](domiknows.sensor.allennlp.md#submodules)
      * [domiknows.sensor.allennlp.base module](domiknows.sensor.allennlp.md#domiknows-sensor-allennlp-base-module)
      * [domiknows.sensor.allennlp.learner module](domiknows.sensor.allennlp.md#domiknows-sensor-allennlp-learner-module)
      * [domiknows.sensor.allennlp.module module](domiknows.sensor.allennlp.md#domiknows-sensor-allennlp-module-module)
      * [domiknows.sensor.allennlp.sensor module](domiknows.sensor.allennlp.md#domiknows-sensor-allennlp-sensor-module)
      * [Module contents](domiknows.sensor.allennlp.md#module-contents)
    * [domiknows.sensor.pytorch package](domiknows.sensor.pytorch.md)
      * [Subpackages](domiknows.sensor.pytorch.md#subpackages)
      * [Submodules](domiknows.sensor.pytorch.md#submodules)
      * [domiknows.sensor.pytorch.learnerModels module](domiknows.sensor.pytorch.md#domiknows-sensor-pytorch-learnermodels-module)
      * [domiknows.sensor.pytorch.learners module](domiknows.sensor.pytorch.md#domiknows-sensor-pytorch-learners-module)
      * [domiknows.sensor.pytorch.query_sensor module](domiknows.sensor.pytorch.md#domiknows-sensor-pytorch-query-sensor-module)
      * [domiknows.sensor.pytorch.relation_sensors module](domiknows.sensor.pytorch.md#domiknows-sensor-pytorch-relation-sensors-module)
      * [domiknows.sensor.pytorch.sensors module](domiknows.sensor.pytorch.md#domiknows-sensor-pytorch-sensors-module)
      * [domiknows.sensor.pytorch.utils module](domiknows.sensor.pytorch.md#domiknows-sensor-pytorch-utils-module)
      * [Module contents](domiknows.sensor.pytorch.md#module-contents)
    * [domiknows.sensor.torch package](domiknows.sensor.torch.md)
      * [Submodules](domiknows.sensor.torch.md#submodules)
      * [domiknows.sensor.torch.learner module](domiknows.sensor.torch.md#domiknows-sensor-torch-learner-module)
      * [domiknows.sensor.torch.sensor module](domiknows.sensor.torch.md#domiknows-sensor-torch-sensor-module)
      * [Module contents](domiknows.sensor.torch.md#module-domiknows.sensor.torch)
  * [Submodules](domiknows.sensor.md#submodules)
  * [domiknows.sensor.learner module](domiknows.sensor.md#module-domiknows.sensor.learner)
    * [`Learner`](domiknows.sensor.md#domiknows.sensor.learner.Learner)
      * [`Learner.parameters`](domiknows.sensor.md#domiknows.sensor.learner.Learner.parameters)
  * [domiknows.sensor.sensor module](domiknows.sensor.md#module-domiknows.sensor.sensor)
    * [`Sensor`](domiknows.sensor.md#domiknows.sensor.sensor.Sensor)
      * [`Sensor.forward()`](domiknows.sensor.md#domiknows.sensor.sensor.Sensor.forward)
      * [`Sensor.propagate_context()`](domiknows.sensor.md#domiknows.sensor.sensor.Sensor.propagate_context)
      * [`Sensor.update_context()`](domiknows.sensor.md#domiknows.sensor.sensor.Sensor.update_context)
  * [Module contents](domiknows.sensor.md#module-domiknows.sensor)
* [domiknows.solver package](domiknows.solver.md)
  * [Subpackages](domiknows.solver.md#subpackages)
    * [domiknows.solver.constructor package](domiknows.solver.constructor.md)
      * [Submodules](domiknows.solver.constructor.md#submodules)
      * [domiknows.solver.constructor.constructor module](domiknows.solver.constructor.md#module-domiknows.solver.constructor.constructor)
      * [Module contents](domiknows.solver.constructor.md#module-domiknows.solver.constructor)
    * [domiknows.solver.session package](domiknows.solver.session.md)
      * [Submodules](domiknows.solver.session.md#submodules)
      * [domiknows.solver.session.gurobi_session module](domiknows.solver.session.md#module-domiknows.solver.session.gurobi_session)
      * [domiknows.solver.session.solver_session module](domiknows.solver.session.md#module-domiknows.solver.session.solver_session)
      * [Module contents](domiknows.solver.session.md#module-domiknows.solver.session)
  * [Submodules](domiknows.solver.md#submodules)
  * [domiknows.solver.allennlpInferenceSolver module](domiknows.solver.md#domiknows-solver-allennlpinferencesolver-module)
  * [domiknows.solver.allennlplogInferenceSolver module](domiknows.solver.md#domiknows-solver-allennlploginferencesolver-module)
  * [domiknows.solver.dummyILPOntSolver module](domiknows.solver.md#module-domiknows.solver.dummyILPOntSolver)
    * [`dummyILPOntSolver`](domiknows.solver.md#domiknows.solver.dummyILPOntSolver.dummyILPOntSolver)
      * [`dummyILPOntSolver.calculateILPSelection()`](domiknows.solver.md#domiknows.solver.dummyILPOntSolver.dummyILPOntSolver.calculateILPSelection)
  * [domiknows.solver.gekkoILPBooleanMethods module](domiknows.solver.md#module-domiknows.solver.gekkoILPBooleanMethods)
    * [`gekkoILPBooleanProcessor`](domiknows.solver.md#domiknows.solver.gekkoILPBooleanMethods.gekkoILPBooleanProcessor)
      * [`gekkoILPBooleanProcessor.and2Var()`](domiknows.solver.md#domiknows.solver.gekkoILPBooleanMethods.gekkoILPBooleanProcessor.and2Var)
      * [`gekkoILPBooleanProcessor.andVar()`](domiknows.solver.md#domiknows.solver.gekkoILPBooleanMethods.gekkoILPBooleanProcessor.andVar)
      * [`gekkoILPBooleanProcessor.epqVar()`](domiknows.solver.md#domiknows.solver.gekkoILPBooleanMethods.gekkoILPBooleanProcessor.epqVar)
      * [`gekkoILPBooleanProcessor.ifVar()`](domiknows.solver.md#domiknows.solver.gekkoILPBooleanMethods.gekkoILPBooleanProcessor.ifVar)
      * [`gekkoILPBooleanProcessor.ilpSolver`](domiknows.solver.md#domiknows.solver.gekkoILPBooleanMethods.gekkoILPBooleanProcessor.ilpSolver)
      * [`gekkoILPBooleanProcessor.main()`](domiknows.solver.md#domiknows.solver.gekkoILPBooleanMethods.gekkoILPBooleanProcessor.main)
      * [`gekkoILPBooleanProcessor.nand2Var()`](domiknows.solver.md#domiknows.solver.gekkoILPBooleanMethods.gekkoILPBooleanProcessor.nand2Var)
      * [`gekkoILPBooleanProcessor.nandVar()`](domiknows.solver.md#domiknows.solver.gekkoILPBooleanMethods.gekkoILPBooleanProcessor.nandVar)
      * [`gekkoILPBooleanProcessor.nor2Var()`](domiknows.solver.md#domiknows.solver.gekkoILPBooleanMethods.gekkoILPBooleanProcessor.nor2Var)
      * [`gekkoILPBooleanProcessor.norVar()`](domiknows.solver.md#domiknows.solver.gekkoILPBooleanMethods.gekkoILPBooleanProcessor.norVar)
      * [`gekkoILPBooleanProcessor.notVar()`](domiknows.solver.md#domiknows.solver.gekkoILPBooleanMethods.gekkoILPBooleanProcessor.notVar)
      * [`gekkoILPBooleanProcessor.or2Var()`](domiknows.solver.md#domiknows.solver.gekkoILPBooleanMethods.gekkoILPBooleanProcessor.or2Var)
      * [`gekkoILPBooleanProcessor.orVar()`](domiknows.solver.md#domiknows.solver.gekkoILPBooleanMethods.gekkoILPBooleanProcessor.orVar)
      * [`gekkoILPBooleanProcessor.xorVar()`](domiknows.solver.md#domiknows.solver.gekkoILPBooleanMethods.gekkoILPBooleanProcessor.xorVar)
  * [domiknows.solver.gekkoILPOntSolver module](domiknows.solver.md#domiknows-solver-gekkoilpontsolver-module)
  * [domiknows.solver.gurobiILPBooleanMethods module](domiknows.solver.md#module-domiknows.solver.gurobiILPBooleanMethods)
    * [`gurobiILPBooleanProcessor`](domiknows.solver.md#domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor)
      * [`gurobiILPBooleanProcessor.and2Var()`](domiknows.solver.md#domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor.and2Var)
      * [`gurobiILPBooleanProcessor.andVar()`](domiknows.solver.md#domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor.andVar)
      * [`gurobiILPBooleanProcessor.countVar()`](domiknows.solver.md#domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor.countVar)
      * [`gurobiILPBooleanProcessor.epqVar()`](domiknows.solver.md#domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor.epqVar)
      * [`gurobiILPBooleanProcessor.fixedVar()`](domiknows.solver.md#domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor.fixedVar)
      * [`gurobiILPBooleanProcessor.ifVar()`](domiknows.solver.md#domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor.ifVar)
      * [`gurobiILPBooleanProcessor.nand2Var()`](domiknows.solver.md#domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor.nand2Var)
      * [`gurobiILPBooleanProcessor.nandVar()`](domiknows.solver.md#domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor.nandVar)
      * [`gurobiILPBooleanProcessor.nor2Var()`](domiknows.solver.md#domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor.nor2Var)
      * [`gurobiILPBooleanProcessor.norVar()`](domiknows.solver.md#domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor.norVar)
      * [`gurobiILPBooleanProcessor.notVar()`](domiknows.solver.md#domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor.notVar)
      * [`gurobiILPBooleanProcessor.or2Var()`](domiknows.solver.md#domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor.or2Var)
      * [`gurobiILPBooleanProcessor.orVar()`](domiknows.solver.md#domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor.orVar)
      * [`gurobiILPBooleanProcessor.preprocessLogicalMethodVar()`](domiknows.solver.md#domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor.preprocessLogicalMethodVar)
      * [`gurobiILPBooleanProcessor.xorVar()`](domiknows.solver.md#domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor.xorVar)
  * [domiknows.solver.gurobiILPOntSolver module](domiknows.solver.md#domiknows-solver-gurobiilpontsolver-module)
  * [domiknows.solver.ilpBooleanMethods module](domiknows.solver.md#module-domiknows.solver.ilpBooleanMethods)
    * [`ilpBooleanProcessor`](domiknows.solver.md#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor)
      * [`ilpBooleanProcessor.and2Var()`](domiknows.solver.md#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor.and2Var)
      * [`ilpBooleanProcessor.andVar()`](domiknows.solver.md#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor.andVar)
      * [`ilpBooleanProcessor.epqVar()`](domiknows.solver.md#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor.epqVar)
      * [`ilpBooleanProcessor.fixedVar()`](domiknows.solver.md#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor.fixedVar)
      * [`ilpBooleanProcessor.ifVar()`](domiknows.solver.md#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor.ifVar)
      * [`ilpBooleanProcessor.nand2Var()`](domiknows.solver.md#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor.nand2Var)
      * [`ilpBooleanProcessor.nandVar()`](domiknows.solver.md#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor.nandVar)
      * [`ilpBooleanProcessor.nor2Var()`](domiknows.solver.md#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor.nor2Var)
      * [`ilpBooleanProcessor.norVar()`](domiknows.solver.md#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor.norVar)
      * [`ilpBooleanProcessor.notVar()`](domiknows.solver.md#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor.notVar)
      * [`ilpBooleanProcessor.or2Var()`](domiknows.solver.md#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor.or2Var)
      * [`ilpBooleanProcessor.orVar()`](domiknows.solver.md#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor.orVar)
      * [`ilpBooleanProcessor.xorVar()`](domiknows.solver.md#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor.xorVar)
  * [domiknows.solver.ilpBooleanMethodsCalculator module](domiknows.solver.md#domiknows-solver-ilpbooleanmethodscalculator-module)
  * [domiknows.solver.ilpConfig module](domiknows.solver.md#module-domiknows.solver.ilpConfig)
  * [domiknows.solver.ilpOntSolver module](domiknows.solver.md#module-domiknows.solver.ilpOntSolver)
    * [`ilpOntSolver`](domiknows.solver.md#domiknows.solver.ilpOntSolver.ilpOntSolver)
      * [`ilpOntSolver.calculateILPSelection()`](domiknows.solver.md#domiknows.solver.ilpOntSolver.ilpOntSolver.calculateILPSelection)
      * [`ilpOntSolver.loadOntology()`](domiknows.solver.md#domiknows.solver.ilpOntSolver.ilpOntSolver.loadOntology)
      * [`ilpOntSolver.setup_solver_logger()`](domiknows.solver.md#domiknows.solver.ilpOntSolver.ilpOntSolver.setup_solver_logger)
      * [`ilpOntSolver.update_config()`](domiknows.solver.md#domiknows.solver.ilpOntSolver.ilpOntSolver.update_config)
  * [domiknows.solver.ilpOntSolverFactory module](domiknows.solver.md#module-domiknows.solver.ilpOntSolverFactory)
    * [`ilpOntSolverFactory`](domiknows.solver.md#domiknows.solver.ilpOntSolverFactory.ilpOntSolverFactory)
      * [`ilpOntSolverFactory.getClass()`](domiknows.solver.md#domiknows.solver.ilpOntSolverFactory.ilpOntSolverFactory.getClass)
      * [`ilpOntSolverFactory.getOntSolverInstance()`](domiknows.solver.md#domiknows.solver.ilpOntSolverFactory.ilpOntSolverFactory.getOntSolverInstance)
  * [domiknows.solver.lcLossBooleanMethods module](domiknows.solver.md#domiknows-solver-lclossbooleanmethods-module)
  * [domiknows.solver.lcLossSampleBooleanMethods module](domiknows.solver.md#domiknows-solver-lclosssamplebooleanmethods-module)
  * [domiknows.solver.mini_solver_debug module](domiknows.solver.md#domiknows-solver-mini-solver-debug-module)
  * [domiknows.solver.solver module](domiknows.solver.md#module-domiknows.solver.solver)
    * [`Solver`](domiknows.solver.md#domiknows.solver.solver.Solver)
      * [`Solver.argmax()`](domiknows.solver.md#domiknows.solver.solver.Solver.argmax)
      * [`Solver.argmin()`](domiknows.solver.md#domiknows.solver.solver.Solver.argmin)
      * [`Solver.max()`](domiknows.solver.md#domiknows.solver.solver.Solver.max)
      * [`Solver.min()`](domiknows.solver.md#domiknows.solver.solver.Solver.min)
      * [`Solver.optimize()`](domiknows.solver.md#domiknows.solver.solver.Solver.optimize)
  * [Module contents](domiknows.solver.md#module-domiknows.solver)

## Submodules

## domiknows.base module

### *class* domiknows.base.AutoNamed(name=None)

Bases: [`Named`](#domiknows.base.Named)

#### assign_suggest_name(name=None)

#### *classmethod* clear()

#### *classmethod* get(name, value=None)

#### *static* localize_namespace(cls_)

#### *static* named_singleton(cls_)

#### *classmethod* suggest_name()

### *class* domiknows.base.Named(name)

Bases: [`Scoped`](#domiknows.base.Scoped)

### *class* domiknows.base.NamedTree(name=None)

Bases: [`NamedTreeNode`](#domiknows.base.NamedTreeNode), `OrderedDict`

#### attach(sub, name=None)

#### cutGraphName(names)

#### del_apply(name)

#### del_sub(\*names, delim='/', trim=True)

#### detach(sub=None, all=False)

#### extract_name(\*names, delim='/', trim=True)

#### get_apply(name)

#### get_sub(\*names, delim='/', trim=True)

#### parse_query_apply(func, \*names, delim='/', trim=True)

#### *property* scope_key

#### set_apply(name, sub)

#### set_sub(\*names, sub, delim='/', trim=True)

#### traversal_apply(func, filter_fn=<function NamedTree.<lambda>>, order='pre', first='depth')

#### what()

### *class* domiknows.base.NamedTreeNode(name=None)

Bases: [`Named`](#domiknows.base.Named)

#### attach_to_context(name=None)

#### attached(sup)

#### *classmethod* clear()

#### *classmethod* default()

#### *property* fullname

#### get_fullname(delim='/')

#### *static* localize_context(cls_)

#### *classmethod* share_context(cls_)

#### *property* sup

#### *property* sups

#### what()

### *class* domiknows.base.Scoped(blocking=False)

Bases: `object`

A lock that count all trials of acquiring and require same number of releasing to really unlock.
Who else need this…? Well, let me leave it nested inside then…

#### *class* LevelLock

Bases: `object`

#### acquire(blocking=False)

#### *property* level

#### release()

#### *static* class_scope(cls_)

#### *classmethod* clear()

#### *static* instance_scope(cls_)

#### scope(blocking=None)

#### *property* scope_key

## domiknows.conf module

## domiknows.config module

## domiknows.utils module

### *class* domiknows.utils.Namespace(\_Namespace_\_dict=None, \*\*kwargs)

Bases: `dict`

#### clone()

#### deepclone()

### *class* domiknows.utils.WrapperMetaClass

Bases: `type`

### domiknows.utils.caller_source()

### domiknows.utils.consume(it)

### domiknows.utils.detuple(\*args)

### domiknows.utils.dict_zip(\*dicts, fillvalue=None)

### domiknows.utils.entuple(args)

### domiknows.utils.enum(inst, cls=None, offset=0)

### domiknows.utils.extract_args(\*args, \*\*kwargs)

### domiknows.utils.find_base(s, n)

### domiknows.utils.getDnSkeletonMode()

### domiknows.utils.getDnSkeletonModeFull()

### domiknows.utils.getProductionModeStatus()

### domiknows.utils.getRegrTimer_logger(\_config={'ifLog': True, 'log_backupCount': 5, 'log_fileMode': 'a', 'log_filename': 'logs/regrTimer', 'log_filesize': 5368709120, 'log_level': 20, 'log_name': 'regrTimer'})

### domiknows.utils.getReuseModel()

### domiknows.utils.get_prop_result(prop, data)

### domiknows.utils.guess_device(data_item)

### domiknows.utils.hide_class(inst, clsinfo, sub=True)

### domiknows.utils.hide_inheritance(cls, clsinfo, sub=True, hidesub=True)

### domiknows.utils.isbad(x)

### domiknows.utils.log(\*args, \*\*kwargs)

### domiknows.utils.optional_arg_decorator(fn, test=None)

### domiknows.utils.optional_arg_decorator_for(test)

### domiknows.utils.printablesize(ni)

### domiknows.utils.prod(iterable)

### domiknows.utils.setDnSkeletonMode(dnSkeleton, full=False)

### domiknows.utils.setProductionLogMode(no_UseTimeLog=False, reuse_model=True)

### domiknows.utils.singleton(cls, getter=None, setter=None)

### domiknows.utils.wrap_batch(values, fillvalue=0)

## Module contents
