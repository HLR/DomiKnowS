# domiknows.solver package

## Subpackages

* [domiknows.solver.constructor package](domiknows.solver.constructor.md)
  * [Submodules](domiknows.solver.constructor.md#submodules)
  * [domiknows.solver.constructor.constructor module](domiknows.solver.constructor.md#module-domiknows.solver.constructor.constructor)
    * [`BatchMaskProbConstructor`](domiknows.solver.constructor.md#domiknows.solver.constructor.constructor.BatchMaskProbConstructor)
      * [`BatchMaskProbConstructor.get_predication()`](domiknows.solver.constructor.md#domiknows.solver.constructor.constructor.BatchMaskProbConstructor.get_predication)
      * [`BatchMaskProbConstructor.isskip()`](domiknows.solver.constructor.md#domiknows.solver.constructor.constructor.BatchMaskProbConstructor.isskip)
    * [`Constructor`](domiknows.solver.constructor.md#domiknows.solver.constructor.constructor.Constructor)
      * [`Constructor.candidates()`](domiknows.solver.constructor.md#domiknows.solver.constructor.constructor.Constructor.candidates)
      * [`Constructor.constraints()`](domiknows.solver.constructor.md#domiknows.solver.constructor.constructor.Constructor.constraints)
      * [`Constructor.get_predication()`](domiknows.solver.constructor.md#domiknows.solver.constructor.constructor.Constructor.get_predication)
      * [`Constructor.isskip()`](domiknows.solver.constructor.md#domiknows.solver.constructor.constructor.Constructor.isskip)
      * [`Constructor.logger`](domiknows.solver.constructor.md#domiknows.solver.constructor.constructor.Constructor.logger)
      * [`Constructor.objective()`](domiknows.solver.constructor.md#domiknows.solver.constructor.constructor.Constructor.objective)
      * [`Constructor.variables()`](domiknows.solver.constructor.md#domiknows.solver.constructor.constructor.Constructor.variables)
    * [`ProbConstructor`](domiknows.solver.constructor.md#domiknows.solver.constructor.constructor.ProbConstructor)
      * [`ProbConstructor.get_predication()`](domiknows.solver.constructor.md#domiknows.solver.constructor.constructor.ProbConstructor.get_predication)
    * [`ScoreConstructor`](domiknows.solver.constructor.md#domiknows.solver.constructor.constructor.ScoreConstructor)
      * [`ScoreConstructor.get_predication()`](domiknows.solver.constructor.md#domiknows.solver.constructor.constructor.ScoreConstructor.get_predication)
  * [Module contents](domiknows.solver.constructor.md#module-domiknows.solver.constructor)
* [domiknows.solver.session package](domiknows.solver.session.md)
  * [Submodules](domiknows.solver.session.md#submodules)
  * [domiknows.solver.session.gurobi_session module](domiknows.solver.session.md#module-domiknows.solver.session.gurobi_session)
    * [`GurobiSession`](domiknows.solver.session.md#domiknows.solver.session.gurobi_session.GurobiSession)
      * [`GurobiSession.CMAP`](domiknows.solver.session.md#domiknows.solver.session.gurobi_session.GurobiSession.CMAP)
      * [`GurobiSession.OMAP`](domiknows.solver.session.md#domiknows.solver.session.gurobi_session.GurobiSession.OMAP)
      * [`GurobiSession.VMAP`](domiknows.solver.session.md#domiknows.solver.session.gurobi_session.GurobiSession.VMAP)
      * [`GurobiSession.constr()`](domiknows.solver.session.md#domiknows.solver.session.gurobi_session.GurobiSession.constr)
      * [`GurobiSession.get_value()`](domiknows.solver.session.md#domiknows.solver.session.gurobi_session.GurobiSession.get_value)
      * [`GurobiSession.obj()`](domiknows.solver.session.md#domiknows.solver.session.gurobi_session.GurobiSession.obj)
      * [`GurobiSession.optimize()`](domiknows.solver.session.md#domiknows.solver.session.gurobi_session.GurobiSession.optimize)
      * [`GurobiSession.var()`](domiknows.solver.session.md#domiknows.solver.session.gurobi_session.GurobiSession.var)
  * [domiknows.solver.session.solver_session module](domiknows.solver.session.md#module-domiknows.solver.session.solver_session)
    * [`SolverSession`](domiknows.solver.session.md#domiknows.solver.session.solver_session.SolverSession)
      * [`SolverSession.CTYPE`](domiknows.solver.session.md#domiknows.solver.session.solver_session.SolverSession.CTYPE)
      * [`SolverSession.OTYPE`](domiknows.solver.session.md#domiknows.solver.session.solver_session.SolverSession.OTYPE)
      * [`SolverSession.VTYPE`](domiknows.solver.session.md#domiknows.solver.session.solver_session.SolverSession.VTYPE)
      * [`SolverSession.constr()`](domiknows.solver.session.md#domiknows.solver.session.solver_session.SolverSession.constr)
      * [`SolverSession.obj()`](domiknows.solver.session.md#domiknows.solver.session.solver_session.SolverSession.obj)
      * [`SolverSession.optimize()`](domiknows.solver.session.md#domiknows.solver.session.solver_session.SolverSession.optimize)
      * [`SolverSession.var()`](domiknows.solver.session.md#domiknows.solver.session.solver_session.SolverSession.var)
  * [Module contents](domiknows.solver.session.md#module-domiknows.solver.session)

## Submodules

## domiknows.solver.allennlpInferenceSolver module

## domiknows.solver.allennlplogInferenceSolver module

## domiknows.solver.dummyILPOntSolver module

### *class* domiknows.solver.dummyILPOntSolver.dummyILPOntSolver(graph, ontologiesTuple, \_ilpConfig={'ifLog': True, 'ilpSolver': 'Gurobi', 'log_backupCount': 5, 'log_fileMode': 'a', 'log_filename': 'logs/ilpOntSolver', 'log_filesize': 5368709120, 'log_level': 20, 'log_name': 'ilpOntSolver'})

Bases: [`ilpOntSolver`](#domiknows.solver.ilpOntSolver.ilpOntSolver)

#### calculateILPSelection(phrase, fun=None, epsilon=1e-05, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None)

## domiknows.solver.gekkoILPBooleanMethods module

### *class* domiknows.solver.gekkoILPBooleanMethods.gekkoILPBooleanProcessor(\_ildConfig={'ifLog': True, 'ilpSolver': 'Gurobi', 'log_backupCount': 5, 'log_fileMode': 'a', 'log_filename': 'logs/ilpOntSolver', 'log_filesize': 5368709120, 'log_level': 20, 'log_name': 'ilpOntSolver'})

Bases: `object`

#### and2Var(\_var1, \_var2)

#### andVar(\*\_var)

#### epqVar(\_var1, \_var2)

#### ifVar(\_var1, \_var2)

#### ilpSolver *= 'Gurobi'*

#### main()

#### nand2Var(\_var1, \_var2)

#### nandVar(\*\_var)

#### nor2Var(\_var1, \_var2)

#### norVar(\*\_var)

#### notVar(\_var)

#### or2Var(\_var1, \_var2)

#### orVar(\*\_var)

#### xorVar(\_var1, \_var2)

## domiknows.solver.gekkoILPOntSolver module

## domiknows.solver.gurobiILPBooleanMethods module

### *class* domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor(\_ildConfig={'ifLog': True, 'ilpSolver': 'Gurobi', 'log_backupCount': 5, 'log_fileMode': 'a', 'log_filename': 'logs/ilpOntSolver', 'log_filesize': 5368709120, 'log_level': 20, 'log_name': 'ilpOntSolver'})

Bases: [`ilpBooleanProcessor`](#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor)

#### and2Var(m, var1, var2, onlyConstrains=False)

#### andVar(m, \*var, onlyConstrains=False)

#### countVar(m, \*var, onlyConstrains=False, limitOp='None', limit=1, logicMethodName='COUNT')

#### epqVar(m, var1, var2, onlyConstrains=False)

#### fixedVar(m, var, onlyConstrains=False)

#### ifVar(m, var1, var2, onlyConstrains=False)

#### nand2Var(m, var1, var2, onlyConstrains=False)

#### nandVar(m, \*var, onlyConstrains=False)

#### nor2Var(m, var1, var2, onlyConstrains=False)

#### norVar(m, \*var, onlyConstrains=False)

#### notVar(m, var, onlyConstrains=False)

#### or2Var(m, var1, var2, onlyConstrains=False)

#### orVar(m, \*var, onlyConstrains=False)

#### preprocessLogicalMethodVar(var, logicMethodName, varNameConnector, minN=2)

#### xorVar(m, var1, var2, onlyConstrains=False)

## domiknows.solver.gurobiILPOntSolver module

## domiknows.solver.ilpBooleanMethods module

### *class* domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor(\_ildConfig={'ifLog': True, 'ilpSolver': 'Gurobi', 'log_backupCount': 5, 'log_fileMode': 'a', 'log_filename': 'logs/ilpOntSolver', 'log_filesize': 5368709120, 'log_level': 20, 'log_name': 'ilpOntSolver'})

Bases: `object`

#### *abstract* and2Var(m, \_var1, \_var2, onlyConstrains=False)

#### *abstract* andVar(m, \*\_var, onlyConstrains=False)

#### *abstract* epqVar(m, \_var1, \_var2, onlyConstrains=False)

#### *abstract* fixedVar(m, \_var, onlyConstrains=False)

#### *abstract* ifVar(m, \_var1, \_var2, onlyConstrains=False)

#### *abstract* nand2Var(m, \_var1, \_var2)

#### *abstract* nandVar(m, \*\_var, onlyConstrains=False)

#### *abstract* nor2Var(m, \_var1, \_var2)

#### *abstract* norVar(m, \*\_var, onlyConstrains=False)

#### *abstract* notVar(m, \_var, onlyConstrains=False)

#### *abstract* or2Var(m, \_var1, \_var2, onlyConstrains=False)

#### *abstract* orVar(m, \*\_var, onlyConstrains=False)

#### *abstract* xorVar(m, \_var1, \_var2, onlyConstrains=False)

## domiknows.solver.ilpBooleanMethodsCalculator module

## domiknows.solver.ilpConfig module

## domiknows.solver.ilpOntSolver module

### *class* domiknows.solver.ilpOntSolver.ilpOntSolver(graph, ontologiesTuple, \_ilpConfig)

Bases: `object`

#### *abstract* calculateILPSelection(phrase, fun=None, epsilon=1e-05, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None, minimizeObjective=False, hardConstrains=[])

#### loadOntology(ontologies)

#### setup_solver_logger(\_ilpConfig={'ifLog': True, 'ilpSolver': 'Gurobi', 'log_backupCount': 5, 'log_fileMode': 'a', 'log_filename': 'logs/ilpOntSolver', 'log_filesize': 5368709120, 'log_level': 20, 'log_name': 'ilpOntSolver'})

#### update_config(graph=None, ontologiesTuple=None, \_ilpConfig=None)

## domiknows.solver.ilpOntSolverFactory module

### *class* domiknows.solver.ilpOntSolverFactory.ilpOntSolverFactory

Bases: `object`

#### *classmethod* getClass(\*SolverClasses)

#### *classmethod* getOntSolverInstance(graph, \*SupplementalClasses, \_ilpConfig={'ifLog': True, 'ilpSolver': 'Gurobi', 'log_backupCount': 5, 'log_fileMode': 'a', 'log_filename': 'logs/ilpOntSolver', 'log_filesize': 5368709120, 'log_level': 20, 'log_name': 'ilpOntSolver'}, \*\*kwargs)

## domiknows.solver.lcLossBooleanMethods module

## domiknows.solver.lcLossSampleBooleanMethods module

## domiknows.solver.mini_solver_debug module

## domiknows.solver.solver module

### *class* domiknows.solver.solver.Solver

Bases: `object`

#### argmax()

#### argmin()

#### max()

#### min()

#### *abstract* optimize(min=True)

## Module contents
