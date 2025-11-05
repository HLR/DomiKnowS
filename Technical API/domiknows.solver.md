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

## domiknows.solver.dummyILPOntSolver module

### *class* domiknows.solver.dummyILPOntSolver.dummyILPOntSolver(graph, ontologiesTuple, \_ilpConfig={'ifLog': True, 'ilpSolver': 'Gurobi', 'log_backupCount': 5, 'log_fileMode': 'a', 'log_filename': 'logs/ilpOntSolver', 'log_filesize': 5368709120, 'log_level': 20, 'log_name': 'ilpOntSolver'})

Bases: [`ilpOntSolver`](#domiknows.solver.ilpOntSolver.ilpOntSolver)

#### calculateILPSelection(phrase, fun=None, epsilon=1e-05, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None)

## domiknows.solver.gurobiILPBooleanMethods module

### *class* domiknows.solver.gurobiILPBooleanMethods.gurobiILPBooleanProcessor(\_ildConfig={'ifLog': True, 'ilpSolver': 'Gurobi', 'log_backupCount': 5, 'log_fileMode': 'a', 'log_filename': 'logs/ilpOntSolver', 'log_filesize': 5368709120, 'log_level': 20, 'log_name': 'ilpOntSolver'})

Bases: [`ilpBooleanProcessor`](#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor)

#### andVar(m, \*var, onlyConstrains=False)

General **N‑ary conjunction**.

Reified form:
: varAND ≤ v_i               for every i
  Σ v_i ≤ varAND + N − 1

Constraint‑only: enforce `Σ v_i ≥ N` (all inputs are 1).

#### compareCountsVar(m, varsA, varsB, , compareOp='>', diff=0, onlyConstrains=False, logicMethodName='COUNT_CMP')

Compare the counts of **two sets** of literals.

Encodes the relation:
: Σ(varsA)   compareOp   Σ(varsB) + diff

where `compareOp ∈ {'>', '>=', '<', '<=', '==', '!='}`.
With *onlyConstrains=False* the method returns a fresh binary that is
1 when the relation holds. Otherwise it just adds the constraints.

#### countVar(m, \*var, onlyConstrains=False, limitOp='None', limit=1, logicMethodName='COUNT')

Compare the **number of True literals** in *var* against a constant.

Supports three relations via *limitOp*:
: • ‘>=’  (at least *limit* Trues)
  • ‘<=’  (at most  *limit* Trues)
  • ‘==’  (exactly *limit* Trues)

Reified form returns a binary *varCOUNT* that is 1 when the chosen
relation is satisfied. Constraint‑only mode merely imposes the count
without introducing *varCOUNT*.

#### equivalenceVar(m, \*var, onlyConstrains=False)

Logical **equivalence** (biconditional/if-and-only-if).

Returns true when all input variables have the same truth value 
(all true or all false).

For binary case: equiv(a, b) = (a ↔ b) = (a → b) ∧ (b → a)
For n-ary case: equiv(a, b, c, …) = (all true) ∨ (all false)

Reified form (returns *varEQ*): constraints ensure *varEQ* = 1 
exactly when all inputs are equivalent.

Constraint‑only: enforce that all variables have the same truth value.

Args:
: m: Model context
  <br/>
  ```
  *
  ```
  <br/>
  var: Variable number of boolean variables to compare
  onlyConstrains: If True, return loss (constraint violation);
  <br/>
  > if False, return success (truth degree)

Returns:
: Truth degree of equivalence or constraint violation measure

#### fixedVar(m, var, onlyConstrains=False)

Fix an ILP literal to its ground‑truth label.

• If the data node says the variable is *true*, constrain `_var == 1`.
• If labelled *false*, constrain `_var == 0`.
• If the label is missing (e.g. VTag = “-100”), simply return 1 so
  the downstream logic treats it as satisfied.

#### ifVar(m, var1, var2, onlyConstrains=False)

Logical implication: (var1 => var2).
- If either side is None (missing), we do NOT force anything.

> * onlyConstrains=True: no constraint added (skip).
> * onlyConstrains=False: vacuously return 1.
- If both are numeric/bool, evaluate and return {0,1} (no constraints).
- If one side is numeric and the other is an ILP var:
  : * antecedent == 1  and consequent is ILP  -> add: consequent >= 1
    * antecedent == 0  and consequent is ILP  -> no constraint (vacuous truth)
    * antecedent is ILP and consequent == 1   -> no constraint (vacuous truth)
    * antecedent is ILP and consequent == 0   -> add: antecedent <= 0
- If both are ILP vars:
  : * onlyConstrains=True: add A - B <= 0   (A <= B)
    * onlyConstrains=False: create z = (¬A ∨ B) with standard linearization.

#### nandVar(m, \*var, onlyConstrains=False)

General **N‑ary NAND**.

Reified form:
: NOT(varNAND) ≤ v_i          for every i
  Σ v_i ≤ NOT(varNAND) + N − 1

Constraint‑only: enforce `Σ v_i ≤ N − 1` (not all can be True).

#### norVar(m, \*var, onlyConstrains=False)

General **N‑ary NOR**.

Reified form:
: v_i ≤ NOT(varNOR)           for every i
  Σ v_i ≥ NOT(varNOR)

Constraint‑only: enforce `Σ v_i ≤ 0` (all inputs 0).

#### notVar(m, var, onlyConstrains=False)

Logical **negation**.

Reified form:   create binary *varNOT* and add
: 1 − \_var  ==  varNOT             (two‑way equivalence)

so *varNOT* equals the logical *NOT(_var)*.

Constraint‑only form: simply force `_var == 0` so that NOT(_var)
would be *True* without introducing *varNOT*.

#### orVar(m, \*var, onlyConstrains=False)

General **N‑ary disjunction**.

Reified form:
: v_i ≤ varOR                for every i
  Σ v_i ≥ varOR

Constraint‑only: enforce `Σ v_i ≥ 1`.

#### preprocessLogicalMethodVar(var, logicMethodName, varNameConnector, minN=2)

#### summationVar(m, \*var, onlyConstrains=False, logicMethodName='SUMMATION')

Returns a linear expression that sums all provided binary variables.

Parameters:
- m: Gurobi model
- 

```
*
```

var: Variable number of binary variables or constants
- onlyConstrains: Not used for summation (kept for signature consistency)
- logicMethodName: Name for logging purposes

Returns:
- Linear expression representing the sum

#### xorVar(m, \*var, onlyConstrains=False)

Two‑input **exclusive‑or**.

Reified form (returns *varXOR*): standard 4‑constraint encoding
ensuring *varXOR* = 1 exactly when the inputs differ.

Constraint‑only: enforce `Σ v_i == 1` (one True, others False).

## domiknows.solver.gurobiILPOntSolver module

### *class* domiknows.solver.gurobiILPOntSolver.gurobiILPOntSolver(graph, ontologiesTuple, \_ilpConfig, reuse_model=False)

Bases: [`ilpOntSolver`](#domiknows.solver.ilpOntSolver.ilpOntSolver)

#### *class* OrderedDict

Bases: `dict`

Dictionary that remembers insertion order

#### clear() → None.  Remove all items from od.

#### copy() → a shallow copy of od

#### *classmethod* fromkeys(iterable, value=None)

Create a new ordered dictionary with keys from iterable and values set to value.

#### items()

Return a set-like object providing a view on the dict’s items.

#### keys()

Return a set-like object providing a view on the dict’s keys.

#### move_to_end(key, last=True)

Move an existing element to the end (or beginning if last is false).

Raise KeyError if the element does not exist.

#### pop(key) → v, remove specified key and return the corresponding value.

If the key is not found, return the default if given; otherwise,
raise a KeyError.

#### popitem(last=True)

Remove and return a (key, value) pair from the dictionary.

Pairs are returned in LIFO order if last is true or FIFO order if false.

#### setdefault(key, default=None)

Insert key with a value of default if key is not in the dictionary.

Return the value for key if key is in the dictionary, else default.

#### update(\*\*F) → None.  Update D from mapping/iterable E and F.

If E is present and has a .keys() method, then does:  for k in E.keys(): D[k] = E[k]
If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
In either case, this is followed by: for k in F:  D[k] = F[k]

#### values()

Return an object providing a view on the dict’s values.

#### addGraphConstrains(m, rootDn, \*conceptsRelations)

#### addLogicalConstrains(m, dn, lcs, p, key=None)

#### addMulticlassExclusivity(conceptsRelations, rootDn, m)

#### addOntologyConstrains(m, rootDn, \*\_conceptsRelations)

#### calculateILPSelection(dn, \*conceptsRelations, key=('local', 'softmax'), fun=None, epsilon=1e-05, minimizeObjective=False, ignorePinLCs=False)

#### calculateLcLoss(dn, tnorm='L', counting_tnorm=None, sample=False, sampleSize=0, sampleGlobalLoss=False, conceptsRelations=None)

#### calulateSampleLossForVariable(currentLcName, lcVariables, lcSuccesses, sampleSize, eliminateDuplicateSamples, replace_mul=False)

#### conceptIsBinary(concept)

#### conceptIsMultiClass(concept)

#### constructLogicalConstrains(lc, booleanProcessor, m, dn, p, key=None, lcVariablesDns=None, lcVariables=None, headLC=False, loss=False, sample=False, vNo=None, verify=False)

#### countLCVariables(rootDn, \*conceptsRelations)

#### createILPVariable(m, dn, currentConceptRelation, notV=False)

#### createILPVariables(m, x, rootDn, \*conceptsRelations, key=('local', 'softmax'), fun=None, epsilon=1e-05)

#### eliminateDuplicateSamples(lcVariables, sampleSize)

#### eliminate_duplicate_columns(data_dict, rows_to_consider, data_dict_target)

Eliminates columns that have identical elements across specified rows.

Args:
: data_dict: OrderedDict with row names as keys and lists as values
  rows_to_consider: List of row names to check for duplicates
  data_dict_target: OrderedDict with same structure to apply same column elimination

Returns:
: OrderedDict (data_dict_target) with same columns removed

#### fixedLSupport(\_dn, conceptName, vDn, i, m)

#### generateSemanticSample(rootDn, conceptsRelations)

#### getConcept(concept)

#### getConceptName(concept)

#### getDatanodesForConcept(rootDn, currentName, conceptToDNSCash=None)

#### getMLResult(dn, xPkey, e, p, loss=False, sample=False)

#### getProbability(dn, conceptRelation, key=('local', 'softmax'), fun=None, epsilon=1e-05)

#### get_logical_constraints()

#### ilpSolver *= 'Gurobi'*

#### isConceptFixed(conceptName)

#### isVariableFixed(dn, conceptName, e)

#### log_sorted_solutions(solutions_to_log)

Remove duplicates, sort solutions by datanode name and concept, then log them
in alphabetical order with red colour (console only) for items with multiple solutions.

#### processILPModelForP(p, lcP, m, x, dn, pUsed, reusingModel, ilpVarCount, minimizeObjective, lcRun)

#### reset_logical_constraints()

#### set_logical_constraints(new_logical_constraints)

#### valueToBeSkipped(x)

#### verifyResultsLC(dn, key='/argmax')

## domiknows.solver.ilpBooleanMethods module

### *class* domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor

Bases: `object`

Abstract base class that specifies *logical* building blocks for
Integer‑Linear‑Programming encodings.

Every concrete subclass must implement each Boolean operator as *either*
a *reified* form (returning a fresh binary that represents the truth
value of the expression) *or* a *hard* form (“onlyConstrains=True”) that
merely adds the appropriate constraints and returns nothing.

All operators assume their inputs are **binary literals**:
: • a Gurobi Var of type *BINARY*  (0/1)
  • the Python integers 0 or 1
  • or `None` (treated as an unknown literal: 1 for positive context,
    0 for negative, exactly as in the original implementation).

#### *abstractmethod* andVar(m, \*\_var, onlyConstrains: bool = False)

General **N‑ary conjunction**.

Reified form:
: varAND ≤ v_i               for every i
  Σ v_i ≤ varAND + N − 1

Constraint‑only: enforce `Σ v_i ≥ N` (all inputs are 1).

#### *abstractmethod* compareCountsVar(m, varsA, varsB, , compareOp: str = '>', diff: int = 0, onlyConstrains: bool = False, logicMethodName: str = 'COUNT_CMP')

Compare the counts of **two sets** of literals.

Encodes the relation:
: Σ(varsA)   compareOp   Σ(varsB) + diff

where `compareOp ∈ {'>', '>=', '<', '<=', '==', '!='}`.
With *onlyConstrains=False* the method returns a fresh binary that is
1 when the relation holds. Otherwise it just adds the constraints.

#### *abstractmethod* countVar(m, \*\_var, onlyConstrains: bool = False, limitOp: str = 'None', limit: int = 1, logicMethodName: str = 'COUNT')

Compare the **number of True literals** in *var* against a constant.

Supports three relations via *limitOp*:
: • ‘>=’  (at least *limit* Trues)
  • ‘<=’  (at most  *limit* Trues)
  • ‘==’  (exactly *limit* Trues)

Reified form returns a binary *varCOUNT* that is 1 when the chosen
relation is satisfied. Constraint‑only mode merely imposes the count
without introducing *varCOUNT*.

#### *abstractmethod* equivalenceVar(m, \*var, onlyConstrains: bool = False)

Logical **equivalence** (biconditional/if-and-only-if).

Returns true when all input variables have the same truth value 
(all true or all false).

For binary case: equiv(a, b) = (a ↔ b) = (a → b) ∧ (b → a)
For n-ary case: equiv(a, b, c, …) = (all true) ∨ (all false)

Reified form (returns *varEQ*): constraints ensure *varEQ* = 1 
exactly when all inputs are equivalent.

Constraint‑only: enforce that all variables have the same truth value.

Args:
: m: Model context
  <br/>
  ```
  *
  ```
  <br/>
  var: Variable number of boolean variables to compare
  onlyConstrains: If True, return loss (constraint violation);
  <br/>
  > if False, return success (truth degree)

Returns:
: Truth degree of equivalence or constraint violation measure

#### *abstractmethod* fixedVar(m, \_var, , onlyConstrains: bool = False)

Fix an ILP literal to its ground‑truth label.

• If the data node says the variable is *true*, constrain `_var == 1`.
• If labelled *false*, constrain `_var == 0`.
• If the label is missing (e.g. VTag = “-100”), simply return 1 so
  the downstream logic treats it as satisfied.

#### *abstractmethod* ifVar(m, \_var1, \_var2, , onlyConstrains: bool = False)

Logical **implication**  (var1 ⇒ var2).

Reified form (returns *varIF*):
: 1 − var1 ≤ varIF
  var2     ≤ varIF
  1 − var1 + var2 ≥ varIF

so *varIF* = 1 unless var1 = 1 and var2 = 0.

Constraint‑only: enforce `var1 ≤ var2`.

#### *abstractmethod* nandVar(m, \*\_var, onlyConstrains: bool = False)

General **N‑ary NAND**.

Reified form:
: NOT(varNAND) ≤ v_i          for every i
  Σ v_i ≤ NOT(varNAND) + N − 1

Constraint‑only: enforce `Σ v_i ≤ N − 1` (not all can be True).

#### *abstractmethod* norVar(m, \*\_var, onlyConstrains: bool = False)

General **N‑ary NOR**.

Reified form:
: v_i ≤ NOT(varNOR)           for every i
  Σ v_i ≥ NOT(varNOR)

Constraint‑only: enforce `Σ v_i ≤ 0` (all inputs 0).

#### *abstractmethod* notVar(m, \_var, , onlyConstrains: bool = False)

Logical **negation**.

Reified form:   create binary *varNOT* and add
: 1 − \_var  ==  varNOT             (two‑way equivalence)

so *varNOT* equals the logical *NOT(_var)*.

Constraint‑only form: simply force `_var == 0` so that NOT(_var)
would be *True* without introducing *varNOT*.

#### *abstractmethod* orVar(m, \*\_var, onlyConstrains: bool = False)

General **N‑ary disjunction**.

Reified form:
: v_i ≤ varOR                for every i
  Σ v_i ≥ varOR

Constraint‑only: enforce `Σ v_i ≥ 1`.

#### summationVar(m, \*\_var)

Sums up a list of binary literals to an integer literal.

#### *abstractmethod* xorVar(m, \*var, onlyConstrains: bool = False)

Two‑input **exclusive‑or**.

Reified form (returns *varXOR*): standard 4‑constraint encoding
ensuring *varXOR* = 1 exactly when the inputs differ.

Constraint‑only: enforce `Σ v_i == 1` (one True, others False).

## domiknows.solver.ilpBooleanMethodsCalculator module

### *class* domiknows.solver.ilpBooleanMethodsCalculator.booleanMethodsCalculator(\_ildConfig={'ifLog': True, 'ilpSolver': 'Gurobi', 'log_backupCount': 5, 'log_fileMode': 'a', 'log_filename': 'logs/ilpOntSolver', 'log_filesize': 5368709120, 'log_level': 20, 'log_name': 'ilpOntSolver'})

Bases: [`ilpBooleanProcessor`](#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor)

#### andVar(\_, \*var, onlyConstrains=False)

General **N‑ary conjunction**.

Reified form:
: varAND ≤ v_i               for every i
  Σ v_i ≤ varAND + N − 1

Constraint‑only: enforce `Σ v_i ≥ N` (all inputs are 1).

#### compareCountsVar(\_, varsA, varsB, , compareOp='>', diff=0, onlyConstrains=False, logicMethodName='COUNT_CMP')

Compare sizes of two sets of binary literals.

> result = 1  iff   count(varsA)  compareOp  ( count(varsB) + diff )

Supported operators: ‘>’, ‘>=’, ‘<’, ‘<=’, ‘==’, ‘!=’
Each literal may be 1/0, torch scalar tensor, or None (ignored).

#### countVar(\_, \*var, onlyConstrains: bool = False, limitOp: str = '==', limit: int = 1, logicMethodName: str = 'COUNT') → int

Return 1 when the number of truthy literals in 

```
*
```

var satisfies
: sum(var)  limitOp  limit

else return 0.

### Parameters

\_
: Ignored (kept for signature compatibility with ILP-based subclasses).

```
*
```

var
: Binary literals to count. None values are skipped.

onlyConstrains
: Unused here; present only for API compatibility.

limitOp
: Comparison operator: one of ‘>=’, ‘<=’, ‘==’.

limit
: Threshold on the count.

logicMethodName
: Label used in error messages (not used here).

### Returns

int
: 1 if the comparison is satisfied, otherwise 0.

#### equivalenceVar(\_, \*var, onlyConstrains=False)

Logical **equivalence** (biconditional/if-and-only-if).

Returns true when all input variables have the same truth value 
(all true or all false).

For binary case: equiv(a, b) = (a ↔ b) = (a → b) ∧ (b → a)
For n-ary case: equiv(a, b, c, …) = (all true) ∨ (all false)

Reified form (returns *varEQ*): constraints ensure *varEQ* = 1 
exactly when all inputs are equivalent.

Constraint‑only: enforce that all variables have the same truth value.

Args:
: m: Model context
  <br/>
  ```
  *
  ```
  <br/>
  var: Variable number of boolean variables to compare
  onlyConstrains: If True, return loss (constraint violation);
  <br/>
  > if False, return success (truth degree)

Returns:
: Truth degree of equivalence or constraint violation measure

#### fixedVar(\_, \_var, onlyConstrains=False)

Fix an ILP literal to its ground‑truth label.

• If the data node says the variable is *true*, constrain `_var == 1`.
• If labelled *false*, constrain `_var == 0`.
• If the label is missing (e.g. VTag = “-100”), simply return 1 so
  the downstream logic treats it as satisfied.

#### ifVar(\_, var1, var2, onlyConstrains=False)

Logical **implication**  (var1 ⇒ var2).

Reified form (returns *varIF*):
: 1 − var1 ≤ varIF
  var2     ≤ varIF
  1 − var1 + var2 ≥ varIF

so *varIF* = 1 unless var1 = 1 and var2 = 0.

Constraint‑only: enforce `var1 ≤ var2`.

#### nandVar(\_, \*var, onlyConstrains=False)

General **N‑ary NAND**.

Reified form:
: NOT(varNAND) ≤ v_i          for every i
  Σ v_i ≤ NOT(varNAND) + N − 1

Constraint‑only: enforce `Σ v_i ≤ N − 1` (not all can be True).

#### norVar(\_, \*var, onlyConstrains=False)

General **N‑ary NOR**.

Reified form:
: v_i ≤ NOT(varNOR)           for every i
  Σ v_i ≥ NOT(varNOR)

Constraint‑only: enforce `Σ v_i ≤ 0` (all inputs 0).

#### notVar(\_, var, onlyConstrains=False)

Logical **negation**.

Reified form:   create binary *varNOT* and add
: 1 − \_var  ==  varNOT             (two‑way equivalence)

so *varNOT* equals the logical *NOT(_var)*.

Constraint‑only form: simply force `_var == 0` so that NOT(_var)
would be *True* without introducing *varNOT*.

#### orVar(\_, \*var, onlyConstrains=False)

General **N‑ary disjunction**.

Reified form:
: v_i ≤ varOR                for every i
  Σ v_i ≥ varOR

Constraint‑only: enforce `Σ v_i ≥ 1`.

#### summationVar(\_, \*var, onlyConstrains=False, logicMethodName='SUMMATION')

Sums up a list of binary literals to an integer literal.

Parameters:
- \_: Model (ignored, kept for signature compatibility)
- 

```
*
```

var: Variable number of binary literals (int, bool, torch.Tensor, or None)
- onlyConstrains: Not used for summation (kept for signature consistency)
- logicMethodName: Name for logging purposes

Returns:
- Integer sum of all truthy values

#### xorVar(\_, \*var, onlyConstrains=False)

Two‑input **exclusive‑or**.

Reified form (returns *varXOR*): standard 4‑constraint encoding
ensuring *varXOR* = 1 exactly when the inputs differ.

Constraint‑only: enforce `Σ v_i == 1` (one True, others False).

## domiknows.solver.ilpConfig module

## domiknows.solver.ilpOntSolver module

### *class* domiknows.solver.ilpOntSolver.ilpOntSolver(graph, ontologiesTuple, \_ilpConfig)

Bases: `object`

#### *abstractmethod* calculateILPSelection(phrase, fun=None, epsilon=1e-05, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None, minimizeObjective=False, hardConstrains=[])

#### loadOntology(ontologies)

#### setup_solver_logger(\_ilpConfig=None)

#### update_config(graph=None, ontologiesTuple=None, \_ilpConfig=None)

### domiknows.solver.ilpOntSolver.resource_filename(package, resource)

## domiknows.solver.ilpOntSolverFactory module

### *class* domiknows.solver.ilpOntSolverFactory.ilpOntSolverFactory

Bases: `object`

#### *classmethod* clear()

Clear ilpOntSolverFactory class state.

This method clears the cached instances and classes to ensure clean state
for testing and other scenarios where solver instances need to be reset.

#### *classmethod* getClass(\*SolverClasses)

#### *classmethod* getOntSolverInstance(graph, \*SupplementalClasses, \_ilpConfig={'ifLog': True, 'ilpSolver': 'Gurobi', 'log_backupCount': 5, 'log_fileMode': 'a', 'log_filename': 'logs/ilpOntSolver', 'log_filesize': 5368709120, 'log_level': 20, 'log_name': 'ilpOntSolver'}, \*\*kwargs) → [ilpOntSolver](#domiknows.solver.ilpOntSolver.ilpOntSolver)

## domiknows.solver.lcLossBooleanMethods module

### *class* domiknows.solver.lcLossBooleanMethods.lcLossBooleanMethods(\_ildConfig={'ifLog': True, 'ilpSolver': 'Gurobi', 'log_backupCount': 5, 'log_fileMode': 'a', 'log_filename': 'logs/ilpOntSolver', 'log_filesize': 5368709120, 'log_level': 20, 'log_name': 'ilpOntSolver'})

Bases: [`ilpBooleanProcessor`](#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor)

#### andVar(\_, \*var, onlyConstrains=False)

General **N‑ary conjunction**.

Reified form:
: varAND ≤ v_i               for every i
  Σ v_i ≤ varAND + N − 1

Constraint‑only: enforce `Σ v_i ≥ N` (all inputs are 1).

#### calc_probabilities(t: Tensor, n: int | None = None) → Tensor

Poisson–binomial PMF over counts 0..n for independent Bernoulli probs in t.
Returns a length-(n+1) vector where pmf[k] = P(K==k).
Differentiable w.r.t. t. t is 1-D with entries in [0,1].

#### compareCountsVar(\_, varsA, varsB, , compareOp: str = '>', diff: int | float = 0, onlyConstrains: bool = False, logicMethodName: str = 'COUNT_CMP')

Truth / loss for  count(varsA)  compareOp  count(varsB) + diff

#### countVar(\_, \*var, onlyConstrains: bool = False, limitOp: str = '==', limit: int = 1, logicMethodName: str = 'COUNT')

Compare the **number of True literals** in *var* against a constant.

Supports three relations via *limitOp*:
: • ‘>=’  (at least *limit* Trues)
  • ‘<=’  (at most  *limit* Trues)
  • ‘==’  (exactly *limit* Trues)

Reified form returns a binary *varCOUNT* that is 1 when the chosen
relation is satisfied. Constraint‑only mode merely imposes the count
without introducing *varCOUNT*.

#### equivalenceVar(\_, \*var, onlyConstrains=False)

Logical **equivalence** (biconditional/if-and-only-if).

Returns true when all input variables have the same truth value 
(all true or all false).

For binary case: equiv(a, b) = (a ↔ b) = (a → b) ∧ (b → a)
For n-ary case: equiv(a, b, c, …) = (all true) ∨ (all false)

Reified form (returns *varEQ*): constraints ensure *varEQ* = 1 
exactly when all inputs are equivalent.

Constraint‑only: enforce that all variables have the same truth value.

Args:
: m: Model context
  <br/>
  ```
  *
  ```
  <br/>
  var: Variable number of boolean variables to compare
  onlyConstrains: If True, return loss (constraint violation);
  <br/>
  > if False, return success (truth degree)

Returns:
: Truth degree of equivalence or constraint violation measure

#### fixedVar(\_, \_var, onlyConstrains=False)

Fix an ILP literal to its ground‑truth label.

• If the data node says the variable is *true*, constrain `_var == 1`.
• If labelled *false*, constrain `_var == 0`.
• If the label is missing (e.g. VTag = “-100”), simply return 1 so
  the downstream logic treats it as satisfied.

#### ifVar(\_, var1, var2, onlyConstrains=False)

Logical **implication**  (var1 ⇒ var2).

Reified form (returns *varIF*):
: 1 − var1 ≤ varIF
  var2     ≤ varIF
  1 − var1 + var2 ≥ varIF

so *varIF* = 1 unless var1 = 1 and var2 = 0.

Constraint‑only: enforce `var1 ≤ var2`.

#### ifVarS(\_, var1, var2, , onlyConstrains=False)

#### nandVar(\_, \*var, onlyConstrains=False)

General **N‑ary NAND**.

Reified form:
: NOT(varNAND) ≤ v_i          for every i
  Σ v_i ≤ NOT(varNAND) + N − 1

Constraint‑only: enforce `Σ v_i ≤ N − 1` (not all can be True).

#### norVar(\_, \*var, onlyConstrains=False)

General **N‑ary NOR**.

Reified form:
: v_i ≤ NOT(varNOR)           for every i
  Σ v_i ≥ NOT(varNOR)

Constraint‑only: enforce `Σ v_i ≤ 0` (all inputs 0).

#### notVar(\_, var, onlyConstrains=False)

Logical **negation**.

Reified form:   create binary *varNOT* and add
: 1 − \_var  ==  varNOT             (two‑way equivalence)

so *varNOT* equals the logical *NOT(_var)*.

Constraint‑only form: simply force `_var == 0` so that NOT(_var)
would be *True* without introducing *varNOT*.

#### orVar(\_, \*var, onlyConstrains=False)

General **N‑ary disjunction**.

Reified form:
: v_i ≤ varOR                for every i
  Σ v_i ≥ varOR

Constraint‑only: enforce `Σ v_i ≥ 1`.

#### setCountingTNorm(tnorm='L')

#### setTNorm(tnorm='L')

#### summationVar(m, \*\_var, onlyConstrains=False, logicMethodName='SUMMATION')

Parameters:
- m: Model (ignored)
- 

```
*
```

\_var: Variable number of binary variables (tensors, scalars, or None)
- onlyConstrains: Not used for summation (kept for signature consistency)
- logicMethodName: Name for logging purposes

Returns:
- Differentiable tensor representing the sum

#### xorVar(\_, \*var, onlyConstrains=False)

Two‑input **exclusive‑or**.

Reified form (returns *varXOR*): standard 4‑constraint encoding
ensuring *varXOR* = 1 exactly when the inputs differ.

Constraint‑only: enforce `Σ v_i == 1` (one True, others False).

## domiknows.solver.lcLossSampleBooleanMethods module

### *class* domiknows.solver.lcLossSampleBooleanMethods.lcLossSampleBooleanMethods(\_ildConfig={'ifLog': True, 'ilpSolver': 'Gurobi', 'log_backupCount': 5, 'log_fileMode': 'a', 'log_filename': 'logs/ilpOntSolver', 'log_filesize': 5368709120, 'log_level': 20, 'log_name': 'ilpOntSolver'})

Bases: [`ilpBooleanProcessor`](#domiknows.solver.ilpBooleanMethods.ilpBooleanProcessor)

#### andVar(\_, \*var, onlyConstrains=False)

General **N‑ary conjunction**.

Reified form:
: varAND ≤ v_i               for every i
  Σ v_i ≤ varAND + N − 1

Constraint‑only: enforce `Σ v_i ≥ N` (all inputs are 1).

#### compareCountsVar(\_, varsA, varsB, , compareOp='>', diff: int = 0, onlyConstrains=False, logicMethodName='COUNT_CMP')

Compare two literal-sets by their per-sample counts.

### Returns

torch.BoolTensor  (shape = [sampleSize])
: • if onlyConstrains is False “success” mask
  • if onlyConstrains is True  “loss”  mask  (¬success)

#### countVar(\_, \*var, onlyConstrains=False, limitOp='==', limit=1, logicMethodName='COUNT')

Compare the **number of True literals** in *var* against a constant.

Supports three relations via *limitOp*:
: • ‘>=’  (at least *limit* Trues)
  • ‘<=’  (at most  *limit* Trues)
  • ‘==’  (exactly *limit* Trues)

Reified form returns a binary *varCOUNT* that is 1 when the chosen
relation is satisfied. Constraint‑only mode merely imposes the count
without introducing *varCOUNT*.

#### equivalenceVar(\_, \*var, onlyConstrains=False)

Logical **equivalence** (biconditional/if-and-only-if).

Returns true when all input variables have the same truth value 
(all true or all false).

For binary case: equiv(a, b) = (a ↔ b) = (a → b) ∧ (b → a)
For n-ary case: equiv(a, b, c, …) = (all true) ∨ (all false)

Reified form (returns *varEQ*): constraints ensure *varEQ* = 1 
exactly when all inputs are equivalent.

Constraint‑only: enforce that all variables have the same truth value.

Args:
: m: Model context
  <br/>
  ```
  *
  ```
  <br/>
  var: Variable number of boolean variables to compare
  onlyConstrains: If True, return loss (constraint violation);
  <br/>
  > if False, return success (truth degree)

Returns:
: Truth degree of equivalence or constraint violation measure

#### fixedVar(\_, var, onlyConstrains=False)

Fix an ILP literal to its ground‑truth label.

• If the data node says the variable is *true*, constrain `_var == 1`.
• If labelled *false*, constrain `_var == 0`.
• If the label is missing (e.g. VTag = “-100”), simply return 1 so
  the downstream logic treats it as satisfied.

#### ifNone(var)

#### ifVar(\_, var1, var2, onlyConstrains=False)

Logical **implication**  (var1 ⇒ var2).

Reified form (returns *varIF*):
: 1 − var1 ≤ varIF
  var2     ≤ varIF
  1 − var1 + var2 ≥ varIF

so *varIF* = 1 unless var1 = 1 and var2 = 0.

Constraint‑only: enforce `var1 ≤ var2`.

#### nandVar(\_, \*var, onlyConstrains=False)

General **N‑ary NAND**.

Reified form:
: NOT(varNAND) ≤ v_i          for every i
  Σ v_i ≤ NOT(varNAND) + N − 1

Constraint‑only: enforce `Σ v_i ≤ N − 1` (not all can be True).

#### norVar(\_, \*var, onlyConstrains=False)

General **N‑ary NOR**.

Reified form:
: v_i ≤ NOT(varNOR)           for every i
  Σ v_i ≥ NOT(varNOR)

Constraint‑only: enforce `Σ v_i ≤ 0` (all inputs 0).

#### notVar(\_, var, onlyConstrains=False)

Logical **negation**.

Reified form:   create binary *varNOT* and add
: 1 − \_var  ==  varNOT             (two‑way equivalence)

so *varNOT* equals the logical *NOT(_var)*.

Constraint‑only form: simply force `_var == 0` so that NOT(_var)
would be *True* without introducing *varNOT*.

#### orVar(\_, \*var, onlyConstrains=False)

General **N‑ary disjunction**.

Reified form:
: v_i ≤ varOR                for every i
  Σ v_i ≥ varOR

Constraint‑only: enforce `Σ v_i ≥ 1`.

#### summationVar(\_, \*var)

Sums up a list of binary literals to an integer literal.

#### xorVar(\_, \*var, onlyConstrains=False)

Two‑input **exclusive‑or**.

Reified form (returns *varXOR*): standard 4‑constraint encoding
ensuring *varXOR* = 1 exactly when the inputs differ.

Constraint‑only: enforce `Σ v_i == 1` (one True, others False).

## domiknows.solver.lossCalculator module

### *class* domiknows.solver.lossCalculator.LossCalculator

Bases: `object`

#### calculateLcLoss(dn, tnorm: str = 'L', counting_tnorm: str | None = None, sample: bool = False, sampleSize: int = 0, sampleGlobalLoss: bool = False, conceptsRelations=None) → Dict[str, Dict]

## domiknows.solver.mini_solver_debug module

### *class* domiknows.solver.mini_solver_debug.MiniProbSolverDebug(graph, ontologiesTuple, \_ilpConfig, SessionType=None, \*\*kwargs)

Bases: [`MiniSolverDebug`](#domiknows.solver.mini_solver_debug.MiniSolverDebug)

#### ilpSolver *= 'mini_prob_debug'*

### *class* domiknows.solver.mini_solver_debug.MiniSolverDebug(graph, ontologiesTuple, \_ilpConfig, constructor=None, SessionType=None, \*\*kwargs)

Bases: [`ilpOntSolver`](#domiknows.solver.ilpOntSolver.ilpOntSolver)

#### calculateILPSelection(data, \*predicates_list)

#### ilpSolver *= 'mini_debug'*

#### set_predication(predicate, idx, value)

#### solve_legacy(data, \*predicates_list)

## domiknows.solver.solver module

### *class* domiknows.solver.solver.Solver

Bases: `object`

#### argmax() → Any

#### argmin() → Any

#### max() → Any

#### min() → Any

#### *abstractmethod* optimize(min=True) → Tuple[Any, Any]

## Module contents
