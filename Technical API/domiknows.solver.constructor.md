# domiknows.solver.constructor package

## Submodules

## domiknows.solver.constructor.constructor module

### *class* domiknows.solver.constructor.constructor.BatchMaskProbConstructor(lazy_not=True, self_relation=True)

Bases: [`Constructor`](#domiknows.solver.constructor.constructor.Constructor)

#### get_predication(predicate, idx, negative=False)

#### isskip(value)

### *class* domiknows.solver.constructor.constructor.Constructor(lazy_not=True, self_relation=True)

Bases: `object`

#### candidates(data, \*predicates_list)

#### constraints(session, candidates, variables, variables_not, \*predicates_list)

#### get_predication(predicate, idx, negative=False)

#### isskip(value)

#### logger *= <Logger domiknows.solver.constructor.constructor (WARNING)>*

#### objective(candidates, variables, predictions, variables_not, predictions_not, \*predicates_list)

#### variables(session, candidates, \*predicates_list)

### *class* domiknows.solver.constructor.constructor.ProbConstructor(lazy_not=False, self_relation=True)

Bases: [`Constructor`](#domiknows.solver.constructor.constructor.Constructor)

#### get_predication(predicate, idx, negative=False)

### *class* domiknows.solver.constructor.constructor.ScoreConstructor(lazy_not=True, self_relation=True)

Bases: [`Constructor`](#domiknows.solver.constructor.constructor.Constructor)

#### get_predication(predicate, idx, negative=False)

## Module contents
