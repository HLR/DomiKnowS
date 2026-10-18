"""Owned ILP models with transactional whole-rule block replacement."""
from dataclasses import dataclass, field
from time import perf_counter
import uuid

from gurobipy import GRB

from domiknows.solver.gurobiILPOntSolver import gurobiILPOntSolver
from .state import signature


@dataclass
class Block:
    fingerprint: tuple
    rows: list
    general: list
    auxiliary: list


@dataclass
class Run:
    model: object
    variables: dict
    priority: int
    blocks: dict = field(default_factory=dict)
    implicit: Block | None = None
    status: int | None = None


class IncrementalILPSolver(gurobiILPOntSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reuse_model = False
        self._session_solver = True
        self._hypothesisBatchSupported = lambda *a: False
        self.committed = {}
        self.staged = {}
        self.new_models = []
        self._building_blocks = {}
        self.change = {}
        self.stats = {}

    def begin(self, change, stats):
        self.change, self.stats = change, stats
        self.staged, self.new_models = {}, []
        self.statuses = []
        self._solve_started = {}
        self._explanation_before_optimize = lambda model, variables: self._solve_started.__setitem__(id(model), perf_counter())

    def commit(self):
        live = {id(run.model) for runs in self.staged.values() for run in runs}
        for runs in self.committed.values():
            for run in runs:
                if id(run.model) not in live:
                    run.model.dispose()
        self.committed = self.staged
        self.staged, self.new_models = {}, []

    def rollback(self):
        for model in self.new_models:
            model.dispose()
        self.staged, self.new_models = {}, []

    def close(self):
        self.rollback()
        for runs in self.committed.values():
            for run in runs:
                run.model.dispose()
        self.committed.clear()

    @staticmethod
    def _rule_key(lc, extra):
        for index, candidate in enumerate(extra):
            if lc is candidate:
                return ('hypothesis', index)
        return (lc.graph.fullname, lc.lcName)

    @staticmethod
    def _capture(model, variables_before, rows_before, general_before, fingerprint):
        model.update()
        tag = '_incremental_' + uuid.uuid4().hex
        rows, general, auxiliary = [], [], []
        for index, row in enumerate(model.getConstrs()[rows_before:]):
            name = tag + '_r' + str(index)
            row.ConstrName = name
            rows.append(name)
        for index, row in enumerate(model.getGenConstrs()[general_before:]):
            name = tag + '_g' + str(index)
            row.GenConstrName = name
            general.append(name)
        for index, var in enumerate(model.getVars()[variables_before:]):
            name = tag + '_v' + str(index)
            var.VarName = name
            auxiliary.append(name)
        model.update()
        return Block(fingerprint, rows, general, auxiliary)

    def addLogicalConstrains(self, model, dn, lcs, p, key=None, compiled=True):
        blocks = self._building_blocks.setdefault(id(model), {})
        for lc in lcs:
            if not lc.active:
                continue
            model.update()
            start = (model.NumVars, model.NumConstrs, model.NumGenConstrs)
            super().addLogicalConstrains(model, dn, [lc], p, key=key, compiled=compiled)
            blocks[self._rule_key(lc, self._extra)] = self._capture(
                model, *start, fingerprint=signature(lc))

    def _capture_run(self, model, variables, lcs, priority):
        self.stats['evaluation_ms'] += (perf_counter() - self._solve_started.pop(id(model))) * 1000
        self.new_models.append(model)
        blocks = self._building_blocks.pop(id(model), {})
        owned_rows = {name for block in blocks.values() for name in block.rows}
        owned_general = {name for block in blocks.values() for name in block.general}
        owned_aux = {name for block in blocks.values() for name in block.auxiliary}
        base_names = {var.VarName for var in variables.values() if var is not None}
        implicit = Block((), [r.ConstrName for r in model.getConstrs() if r.ConstrName not in owned_rows],
            [r.GenConstrName for r in model.getGenConstrs() if r.GenConstrName not in owned_general],
            [v.VarName for v in model.getVars() if v.VarName not in base_names | owned_aux])
        # Implicit row names can repeat. Give each one an owned, unique name.
        tag = '_incremental_base_' + uuid.uuid4().hex
        implicit.rows = []
        for index, row in enumerate(model.getConstrs()):
            if row.ConstrName not in owned_rows:
                name = tag + '_r' + str(index)
                row.ConstrName = name
                implicit.rows.append(name)
        implicit.general = []
        for index, row in enumerate(model.getGenConstrs()):
            if row.GenConstrName not in owned_general:
                name = tag + '_g' + str(index)
                row.GenConstrName = name
                implicit.general.append(name)
        model.update()
        self._captured.append(Run(model, dict(variables), priority, blocks, implicit, model.Status))
        self.statuses.append(model.Status)
        self.stats['ilp_models_created'] += 1

    @staticmethod
    def _remove_blocks(model, blocks):
        row_names = {name for block in blocks for name in block.rows}
        general_names = {name for block in blocks for name in block.general}
        aux_names = {name for block in blocks for name in block.auxiliary}
        model.remove([r for r in model.getConstrs() if r.ConstrName in row_names])
        model.remove([r for r in model.getGenConstrs() if r.GenConstrName in general_names])
        model.update()
        # Blocks are complete DSL constructions; base variables are never in
        # their auxiliary lists. Each block owns its generated auxiliaries.
        model.remove([v for v in model.getVars() if v.VarName in aux_names])
        model.update()

    @staticmethod
    def _bind_variables(dn, variables):
        for (concept, label, instance, index), var in variables.items():
            if var is None:
                continue
            root = dn.findRootConceptOrRelation(concept)
            for node in dn.findDatanodes(select=root):
                if node.instanceID == instance:
                    prefix = 'notx' if str(label).startswith('Not_') else 'x'
                    count = len(concept.enum) if hasattr(concept, 'enum') else 1
                    node.attributes.setdefault(f'<{concept.name}>/ILP/{prefix}', [None] * count)[index] = var

    def _patch(self, previous, dn, concepts, lcs, priority, key, fun, epsilon, minimize, compiled):
        patch_started = perf_counter()
        model = previous.model.copy()
        self.new_models.append(model)
        variables = {k: model.getVarByName(v.VarName) if v is not None else None
                     for k, v in previous.variables.items()}
        blocks = dict(previous.blocks)
        requested = {self._rule_key(lc, self._extra): lc for lc in lcs if lc.active}
        topology = bool(self.change.get('entities') or self.change.get('relations'))
        all_dirty = topology or self.change.get('attributes') or self.change.get('opaque')
        hinted = set(self.change.get('hinted_rules', ()))
        dirty = {name for name, block in blocks.items() if name not in requested or all_dirty or
                 block.fingerprint != signature(requested[name]) or requested[name].lcName in hinted}
        self._remove_blocks(model, [blocks.pop(name) for name in dirty])
        if topology:
            self._remove_blocks(model, [previous.implicit])
            allowed = set()
            for concept in concepts:
                for node in dn.findDatanodes(select=dn.findRootConceptOrRelation(concept[0])):
                    index = 0 if concept[2] is None else concept[2]
                    allowed.add((concept[0], concept[1], node.instanceID, index))
                    if concept[2] is None:
                        allowed.add((concept[0], 'Not_' + concept[1], node.instanceID, index))
            removed = [v for k, v in variables.items() if k not in allowed and v is not None]
            model.remove(removed)
            self.stats['variables_removed'] += len(removed)
            variables = {k: v for k, v in variables.items() if k in allowed}
            model.update()
        self._bind_variables(dn, variables)
        before = model.NumVars
        objective = self.createILPVariables(model, variables, dn, *concepts, key=key, fun=fun, epsilon=epsilon)
        model.update()
        self.stats['variables_added'] += model.NumVars - before
        implicit = previous.implicit
        if topology:
            start = (model.NumVars, model.NumConstrs, model.NumGenConstrs)
            self.addOntologyConstrains(model, dn, *concepts)
            self.addGraphConstrains(model, dn, *concepts)
            self.addMulticlassExclusivity(concepts, dn, model)
            implicit = self._capture(model, *start, fingerprint=())
        self._bind_variables(dn, variables)
        self._building_blocks[id(model)] = {}
        for name, lc in requested.items():
            if name not in blocks:
                self.addLogicalConstrains(model, dn, [lc], priority, key='/ILP/x', compiled=compiled)
                blocks[name] = self._building_blocks[id(model)].pop(name)
                self.stats['constraints_rebuilt'] += 1
            else:
                self.stats['constraints_reused'] += 1
        self._building_blocks.pop(id(model), None)
        model.setObjective(objective if objective is not None else 0,
                           GRB.MINIMIZE if minimize else GRB.MAXIMIZE)
        # Preserve a valid warm-start hint; Gurobi discards it if new rules
        # make it infeasible. It never replaces optimality checks.
        if previous.status == GRB.OPTIMAL:
            for var in previous.model.getVars():
                current = model.getVarByName(var.VarName)
                if current is not None:
                    current.Start = var.X
        started = perf_counter()
        self.stats['grounding_ms'] += (started - patch_started) * 1000
        model.optimize()
        self.stats['evaluation_ms'] += (perf_counter() - started) * 1000
        self.statuses.append(model.Status)
        self.stats['ilp_models_patched' if dirty or topology or len(requested) != len(previous.blocks)
                   else 'ilp_models_reused'] += 1
        return Run(model, variables, priority, blocks, implicit, model.Status)

    def _calculateILPSelection(self, dn, *concepts, **options):
        self.current_device = dn.current_device
        self._extra = tuple(options.get('extraLogicalConstraints', ()))
        candidate = signature(self._extra)
        self._captured = []
        lcP, _ = self._collectHeadLogicalConstraints(dn, self._extra, options.get('ignorePinLCs', False))
        previous = self.committed.get(candidate)
        if previous and ({r.priority for r in previous} != set(lcP) or self.change.get('schema')
                         or self.change.get('opaque') or self.change.get('input_layout')):
            self.stats['full_rebuild_reasons'].append('schema, input layout, priority layout, or opaque grounding changed')
            previous = None
        if previous is None:
            self._explanation_after_optimize = self._capture_run
            try:
                started = perf_counter()
                solving_before = self.stats['evaluation_ms']
                result = super()._calculateILPSelection(dn, *concepts,
                    **{**options, 'populate': False, 'forceFreshModel': True,
                       'raiseOnInfeasible': False, 'computeIIS': False})
                self.stats['grounding_ms'] += max(0., (perf_counter() - started) * 1000 -
                    (self.stats['evaluation_ms'] - solving_before))
            finally:
                self._explanation_after_optimize = None
            self.staged[candidate] = self._captured
            self.stats['constraints_rebuilt'] += sum(len(run.blocks) for run in self._captured)
            return result
        runs = []
        accumulated = []
        for priority in lcP:
            accumulated.extend(lcP[priority])
            old = next(run for run in previous if run.priority == priority)
            runs.append(self._patch(old, dn, concepts, accumulated, priority,
                options.get('key', ('local', 'softmax')), options.get('fun'),
                options.get('epsilon', .00001), options.get('minimizeObjective', False),
                options.get('compiled', True)))
        self.staged[candidate] = runs
        feasible = [run for run in runs if run.status == GRB.OPTIMAL]
        if not feasible:
            return None
        # Ordinary priority inference keeps the last candidate on objective ties.
        best = feasible[0]
        for run in feasible[1:]:
            if (run.model.ObjVal <= best.model.ObjVal if options.get('minimizeObjective')
                    else run.model.ObjVal >= best.model.ObjVal):
                best = run
        return {'objective': best.model.ObjVal, 'priority': best.priority,
                'values': self._snapshotILPVariables(best.variables)}
