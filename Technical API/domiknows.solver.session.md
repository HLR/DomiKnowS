# domiknows.solver.session package

## Submodules

## domiknows.solver.session.gurobi_session module

### *class* domiknows.solver.session.gurobi_session.GurobiSession(silence=True)

Bases: [`SolverSession`](#domiknows.solver.session.solver_session.SolverSession)

#### CMAP *= {<CTYPE.EQ: '=='>: (1, '='), <CTYPE.LT: '<'>: (-1, '>'), <CTYPE.LE: '<='>: (1, '<'), <CTYPE.GT: '>'>: (-1, '<'), <CTYPE.GE: '>='>: (1, '>')}*

#### OMAP *= {<OTYPE.MAX: 'max'>: -1, <OTYPE.MIN: 'min'>: 1}*

#### VMAP *= {<VTYPE.BIN: <class 'bool'>>: 'B', <VTYPE.INT: <class 'int'>>: 'I', <VTYPE.DEC: <class 'float'>>: 'C'}*

#### constr(lhs, ctype, rhs, name=None)

#### get_value(var)

#### obj(otype, expr)

#### optimize()

#### var(vtype, lb, ub, name=None)

## domiknows.solver.session.solver_session module

### *class* domiknows.solver.session.solver_session.SolverSession

Bases: `object`

#### *class* CTYPE(value)

Bases: `Enum`

An enumeration.

#### EQ *= '=='*

#### GE *= '>='*

#### GT *= '>'*

#### LE *= '<='*

#### LT *= '<'*

#### *class* OTYPE(value)

Bases: `Enum`

An enumeration.

#### MAX *= 'max'*

#### MIN *= 'min'*

#### *class* VTYPE(value)

Bases: `Enum`

An enumeration.

#### BIN *= <class 'bool'>*

#### DEC *= <class 'float'>*

#### INT *= <class 'int'>*

#### constr(lhs, ctype, rhs, name=None)

#### obj(otype, expr)

#### optimize()

#### var(vtype, lb, ub, name=None)

## Module contents
