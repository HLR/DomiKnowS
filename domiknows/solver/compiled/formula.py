"""Formula-stage of shared compiled logical-constraint execution.

``CompiledConstraintEvaluator`` executes immutable formula plans produced once
per constraint and cached on the solver across batches. It preserves the native
leaf and Boolean processor semantics for fuzzy loss, sampling, exact circuits,
verification, ILP, and executable inference:

* identity-grounded unary implications are grouped across the graph and run as
  ``[rules, groundings]`` tensors, bypassing per-formula Python dispatch;
* common candidate/path joins (identity, forward and ``.reversed`` relation
  paths, expansion and intersection) use tensorized binding plans; complex
  forms retain the established ``getCandidates`` fallback;
* fuzzy per-datanode probability reads are replaced by batched gathers through
  ``ProbabilityStore`` (``grounding.py``); other modes gather their native
  sampled vectors, circuit literals, discrete values, or ILP variables;
* the t-norm operators themselves are the original ``lcLossBooleanMethods``
  implementations invoked through each constraint's ``__call__`` — the
  numerics are shared with the interpreter by construction, not duplicated.

``CompiledLossCalculator`` mirrors ``LossCalculator``. Every loss-producing
``LogicalConstrain`` subclass that follows the standard formula ``__call__``
contract receives a compiled plan; support is no longer limited by a hard-coded
operator allowlist. ``eqL`` remains a structural candidate filter rather than a
loss formula. Unexpected runtime incompatibilities retain the correctness
fallback. ``fixedL`` no longer disables the whole graph: the pinned concept's
hard 0/1 substitution is replicated per concept in ``ProbabilityStore`` (see
``_fixed_spec`` / ``_apply_fixed``).
"""

from collections import OrderedDict
from threading import RLock
from time import perf_counter_ns

import torch

from domiknows.graph import LcElement, LogicalConstrain, V, CandidateSelection, fixedL
from domiknows.graph.concept import Concept, EnumConcept
from domiknows.graph.logicalConstrain import (
    ifL, sumL, queryL, iotaL, miotaL, eqL,
)
from domiknows.solver.logicalConstraintConstructor import LogicalConstraintConstructor
from domiknows.solver.lossCalculator import LossCalculator, multi_query_joint_nll
from domiknows.solver.adaptiveTNormLossCalculator import TNormSelector, get_constraint_type

from .grounding import ProbabilityStore
from .plan import CompiledPlanCache, TensorizedCandidateResolver


def shared_plan_cache(solver):
    """Return the solver-wide immutable formula-plan cache used by every mode."""
    cache = getattr(solver, '_compiled_formula_plan_cache', None)
    if cache is None:
        cache = CompiledPlanCache()
        solver._compiled_formula_plan_cache = cache
    return cache


#: Formula protocol accepted by the compiled evaluator. Concrete operators no
#: longer need registration in an allowlist; ``lc_tree_supported`` applies the
#: structural-only exclusions such as ``eqL``.
SUPPORTED_LC_TYPES = (LogicalConstrain,)


def lc_tree_supported(lc):
    """Return whether ``lc`` can execute through the compiled formula protocol.

    Formula execution is polymorphic: the evaluator compiles operands and then
    invokes the constraint's normal ``__call__`` implementation. Consequently a
    custom constraint does not need registration in ``SUPPORTED_LC_TYPES``.
    ``eqL`` is excluded only because it is a candidate-path filter, has no truth
    output of its own, and deliberately does not implement the formula call
    signature.
    """
    if not isinstance(lc, LogicalConstrain) or isinstance(lc, eqL):
        return False
    for e in lc.e:
        if isinstance(e, LogicalConstrain) and not lc_tree_supported(e):
            return False
    return True


class CompiledConstraintEvaluator(LogicalConstraintConstructor):
    """Loss-path evaluator with batched probability gathers.

    Subclasses ``LogicalConstraintConstructor`` to reuse its structural
    helpers (``addLossTovDns``, ``eliminate_duplicate_columns``, dtype/device
    handling) while replacing the per-datanode probability loop with
    ``ProbabilityStore.gather_variable``.
    """

    def __init__(self, logger, probStore: ProbabilityStore, plan_cache=None):
        super().__init__(logger)
        self.probStore = probStore
        self.plan_cache = plan_cache if plan_cache is not None else CompiledPlanCache()
        self.candidateResolver = TensorizedCandidateResolver(probStore)
        # Atomic classifier-probability layouts gathered while evaluating the
        # current head constraint. They are normalized to the final grounding
        # axis only after the formula output shape is known.
        self._gathered_probs = []
        # Final, post-expansion/reduction operands passed to the head formula.
        # Prefer these for feature alignment because they already carry the
        # constructor's exact grounding order (including non-primary axes).
        self._feature_operands = None

    def rebind(self, probStore):
        """Bind the persistent evaluator to a new data item."""
        self.probStore = probStore
        self.candidateResolver.rebind(probStore)

    def _record_gathered(self, gathered):
        """Record every tensor in a gathered literal layout.

        The final constraint grounding count is not known at gather time:
        relation expansion can repeat an earlier vector, while existential
        reduction and counting can collapse several literal rows into one
        output. Preserve the nested layout here and align it later in
        :meth:`grounding_features` instead of discarding irregular structures.
        """
        def detach(value):
            if torch.is_tensor(value):
                return value.detach()
            if isinstance(value, list):
                return [detach(item) for item in value]
            if isinstance(value, tuple):
                return tuple(detach(item) for item in value)
            return value

        self._gathered_probs.append(detach(gathered))

    @staticmethod
    def _layout_tensors(value):
        tensors = []
        if torch.is_tensor(value):
            tensors.append(value.detach().reshape(-1))
        elif isinstance(value, (list, tuple)):
            for item in value:
                tensors.extend(CompiledConstraintEvaluator._layout_tensors(item))
        return tensors

    @classmethod
    def _probability_layout(cls, value):
        """Detach tensors and replace native circuit leaves by probabilities."""
        if torch.is_tensor(value):
            return value.detach()
        probability = getattr(value, 'probability', None)
        if torch.is_tensor(probability):
            return probability.detach()
        if isinstance(value, list):
            return [cls._probability_layout(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._probability_layout(item) for item in value)
        if isinstance(value, dict):
            return type(value)(
                (key, cls._probability_layout(item))
                for key, item in value.items())
        return None

    @staticmethod
    def _padded_rows(rows, *, device, dtype):
        width = max((row.numel() for row in rows), default=0)
        if width == 0:
            return None
        matrix = torch.full(
            (len(rows), width), float('nan'), device=device, dtype=dtype)
        for index, row in enumerate(rows):
            if row.numel():
                matrix[index, :row.numel()] = row.to(device=device, dtype=dtype)
        return matrix

    @classmethod
    def _layout_feature_matrix(cls, layout, num_groundings):
        """Convert one literal layout into an exact ``[G, L]`` matrix.

        The cases mirror constructor layout changes:

        * ``G`` non-empty candidate groups become ragged rows (missing groups
          are omitted in the same way ``_normalize_loss_list`` omits ``None``);
        * a vector of length ``N`` expands to ``G=kN`` by contiguous
          ``repeat_interleave``, matching path expansion order;
        * a vector of length ``kG`` reshapes to ``[G,k]``, retaining every
          literal collapsed by counting/existential reduction;
        * enum columns and other parallel tensors concatenate as features.

        Any non-integral layout is an invariant violation and is reported
        rather than silently removing the critic's literal inputs.
        """
        if num_groundings <= 0:
            raise ValueError("grounding feature alignment requires G > 0")

        tensors = cls._layout_tensors(layout)
        if not tensors:
            return None
        device = tensors[0].device
        dtype = tensors[0].dtype

        groups = list(layout) if isinstance(layout, (list, tuple)) else None
        if groups is not None:
            nonempty_rows = []
            for group in groups:
                values = cls._layout_tensors(group)
                if values:
                    nonempty_rows.append(torch.cat(values))

            # Counting/reduction produces one output per candidate group while
            # retaining all literals in that group as feature columns.
            if len(nonempty_rows) == num_groundings:
                return cls._padded_rows(
                    nonempty_rows, device=device, dtype=dtype)

        scalar_leaves = [tensor for tensor in tensors if tensor.numel() == 1]
        if len(scalar_leaves) == len(tensors) == num_groundings:
            return torch.stack(scalar_leaves).reshape(num_groundings, 1)

        matrices = []
        for tensor in tensors:
            size = tensor.numel()
            if size == num_groundings:
                matrix = tensor.reshape(num_groundings, 1)
            elif size == 1:
                matrix = tensor.reshape(1, 1).expand(num_groundings, 1)
            elif num_groundings % size == 0:
                matrix = tensor.repeat_interleave(
                    num_groundings // size).reshape(num_groundings, 1)
            elif size % num_groundings == 0:
                matrix = tensor.reshape(num_groundings, size // num_groundings)
            else:
                raise ValueError(
                    "cannot align literal feature vector of length "
                    f"{size} to {num_groundings} constraint groundings")
            matrices.append(matrix.to(device=device, dtype=dtype))
        return torch.cat(matrices, dim=1)

    def grounding_features(self, num_groundings):
        """Return all gathered literals aligned to ``[groundings, features]``."""
        if not num_groundings:
            return None
        layouts = (
            list(self._feature_operands.values())
            if self._feature_operands is not None
            else self._gathered_probs
        )
        matrices = [
            matrix
            for layout in layouts
            if (matrix := self._layout_feature_matrix(
                layout, num_groundings)) is not None
        ]
        if not matrices:
            raise ValueError(
                "compiled constraint produced a loss but no literal "
                "probabilities for amortized-dual features")
        if any(matrix.shape[0] != num_groundings for matrix in matrices):
            raise ValueError("grounding feature matrices are not row aligned")
        return torch.cat(matrices, dim=1)

    def _get_dtype(self):
        if self.current_dtype is not None:
            return self.current_dtype
        if self.probStore.dtype is not None:
            return self.probStore.dtype
        return torch.float32

    def _gather_runtime_variable(
            self, dns_list, element, lc, *, key, p, loss, sample,
            circuit, model):
        """Read mode-specific leaves over candidates from a compiled plan.

        Fuzzy loss uses ``ProbabilityStore``'s batched tensor gather. Other
        processors need their native leaf objects: sampled Boolean vectors,
        exact-circuit literals, ILP variables, or discrete verification values.
        Candidate/path resolution and formula traversal remain compiled while
        this adapter preserves ``getMLResult``'s established leaf semantics.
        """
        concept = element[0]
        concept_name = concept.name
        prediction_key = '<' + concept_name + '>' + key
        values = []
        sample_info = [] if sample else None

        for candidates in dns_list:
            candidate_values = []
            candidate_sample_info = [] if sample else None
            for candidate in candidates:
                if candidate is None:
                    candidate_values.append(None)
                    continue

                if isinstance(concept, EnumConcept) and element[2] is None:
                    indices = range(len(concept.enum))
                elif isinstance(concept, EnumConcept):
                    indices = (element[2],)
                else:
                    indices = (None,)

                for index in indices:
                    if isinstance(concept, EnumConcept):
                        leaf_element = (concept_name, index, index)
                        fixed_index = index
                    else:
                        leaf_element = (concept_name, 1, 0)
                        fixed_index = 1

                    leaf = self.getMLResult(
                        candidate, prediction_key, leaf_element, p,
                        loss=loss, sample=sample, circuit=circuit,
                        concept=concept)
                    if sample:
                        leaf, leaf_sample_info = leaf
                        candidate_sample_info.append(leaf_sample_info)
                    if str(lc) == 'fixedL':
                        leaf = self.fixedLSupport(
                            candidate, concept_name, leaf, fixed_index, model)
                    candidate_values.append(leaf)

            values.append(candidate_values)
            if sample:
                sample_info.append(candidate_sample_info)

        if circuit:
            def probabilities(value):
                probability = getattr(value, 'probability', None)
                if torch.is_tensor(probability):
                    return probability
                if isinstance(value, list):
                    return [probabilities(item) for item in value]
                if isinstance(value, tuple):
                    return tuple(probabilities(item) for item in value)
                return None

            self._record_gathered(probabilities(values))

        return values, sample_info

    def constructCompiled(self, lc, booleanProcessor, dn, key=None,
                          lcVariablesDns=None, lcVariables=None, headLC=False,
                          vNo=None, label=None, plan=None, *, model=None, p=0,
                          loss=True, sample=False, verify=False, circuit=False,
                          concept_bindings=None):
        """Replay a compiled formula plan for any constraint processor mode.

        Returns the same ``(lc_result, lcVariables)`` tuple the interpreter
        returns, the four-value sampling tuple when ``sample=True``, or the
        candidate dict for ``CandidateSelection`` elements.
        """
        if key is None:
            key = ""

        if lcVariablesDns is None:
            lcVariablesDns = OrderedDict()
        if lcVariables is None:
            lcVariables = OrderedDict()

        if plan is None:
            plan = self.plan_cache.get(lc)

        if concept_bindings is None:
            bound_concepts = {}
        elif isinstance(concept_bindings, dict):
            bound_concepts = concept_bindings
        else:
            representatives = getattr(lc, '_template_concepts', ())
            if len(representatives) != len(concept_bindings):
                raise ValueError(
                    f"Template {lc.lcName} expects {len(representatives)} "
                    f"concept bindings, got {len(concept_bindings)}")
            bound_concepts = {
                id(representative): actual
                for representative, actual in zip(
                    representatives, concept_bindings)
            }

        def bind_element(element):
            if isinstance(element, Concept):
                return bound_concepts.get(id(element), element)
            if (
                isinstance(element, tuple)
                and element
                and isinstance(element[0], Concept)
            ):
                actual = bound_concepts.get(id(element[0]))
                if actual is not None:
                    values = list(element)
                    values[0] = actual
                    if len(values) > 1:
                        values[1] = actual.name
                    return tuple(values)
            return element

        usedVariablesNames = set()
        sampleInfo = {}
        lcVariablesSet = {}
        # Mirrors the interpreter's grounding bookkeeping so operands
        # enumerated over different variable tuples are aligned identically.
        lcVariableBindings = OrderedDict()
        lcVariableVs = OrderedDict()

        if vNo is None:
            vNo = [1, 1]

        integrate = False
        newVariables = {}

        for step in plan.steps:
            eIndex = step.index
            e = bind_element(step.element)

            if isinstance(e, (Concept, LcElement, tuple)):
                variable = step.variable
                variableName = step.variable_name
                if isinstance(e, tuple) and isinstance(e[0], CandidateSelection):
                    e = e[0]

                if variableName in lcVariables:
                    # Re-binding: the variable was used before; reuse its
                    # datanodes but look up the new concept's predictions.
                    newVariableName = "_x" + str(vNo[0])
                    vNo[0] += 1

                    lcVariablesDns[newVariableName] = lcVariablesDns[variableName]

                    is_concept_tuple = isinstance(e, tuple) and len(e) >= 1 and isinstance(e[0], Concept)
                    if is_concept_tuple:
                        if loss and not sample and not circuit:
                            gathered = self.probStore.gather_variable(
                                lcVariablesDns[variableName], e)
                            self._record_gathered(gathered)
                        else:
                            gathered, gathered_sample_info = self._gather_runtime_variable(
                                lcVariablesDns[variableName], e, lc,
                                key=key, p=p, loss=loss, sample=sample,
                                circuit=circuit, model=model)
                            if sample:
                                sampleInfo[newVariableName] = gathered_sample_info
                        lcVariables[newVariableName] = gathered
                    else:
                        lcVariables[newVariableName] = lcVariables[variableName]
                    usedVariablesNames.add(newVariableName)

                elif isinstance(e, (Concept, tuple)):
                    result = self.candidateResolver.get_candidates(
                        step.candidate, dn, e, variable, lcVariablesDns, lc,
                        self.myLogger)

                    if result is None or result[0] is None:
                        continue

                    dnsList, referedVariables, expansionInfo = result

                    lcVariablesDns[variableName] = dnsList

                    lcVariableVs[variableName] = variable
                    binding = self.groundingBinding(variable, dnsList, lcVariablesDns)
                    if binding is not None:
                        lcVariableBindings[variableName] = binding

                    if expansionInfo is not None:
                        # Realign previously collected variable tensors with
                        # the expanded grounding rows (same as interpreter).
                        mapping = expansionInfo['mapping']
                        expanded_vars = expansionInfo['expanded_vars']

                        pre_expansion_len = max(idx for idx, _ in mapping) + 1 if mapping else 0
                        vars_to_expand = set(expanded_vars)
                        for var_name in list(lcVariables.keys()):
                            if var_name in vars_to_expand:
                                continue
                            old_structure = lcVariables[var_name]
                            if old_structure and len(old_structure) == pre_expansion_len:
                                vars_to_expand.add(var_name)

                        # Expansion re-grounds earlier variables onto this one.
                        if binding is not None:
                            for var_name in vars_to_expand:
                                lcVariableBindings[var_name] = binding

                        for var_name in vars_to_expand:
                            if var_name not in lcVariables:
                                continue

                            old_structure = lcVariables[var_name]
                            if not old_structure:
                                continue

                            new_structure = []
                            for orig_group_idx, item_idx in mapping:
                                if orig_group_idx < len(old_structure):
                                    old_group = old_structure[orig_group_idx]
                                    if old_group:
                                        if isinstance(old_group, list):
                                            if len(old_group) == 1:
                                                new_structure.append([old_group[0]])
                                            elif item_idx < len(old_group):
                                                new_structure.append([old_group[item_idx]])
                                            else:
                                                new_structure.append([old_group[0]])
                                        else:
                                            new_structure.append([old_group])
                                    else:
                                        new_structure.append([None])
                                else:
                                    new_structure.append([None])

                            lcVariables[var_name] = new_structure

                    if isinstance(lc, CandidateSelection):
                        continue

                    if len(referedVariables) == 1:
                        referedVariable = referedVariables.pop()

                        if referedVariable.startswith('p'):
                            if referedVariable not in newVariables:
                                newVariables[referedVariable] = set()
                            newVariables[referedVariable].add(variableName)

                    if loss and not sample and not circuit:
                        gathered = self.probStore.gather_variable(dnsList, e)
                        gathered_sample_info = None
                        self._record_gathered(gathered)
                    else:
                        gathered, gathered_sample_info = self._gather_runtime_variable(
                            dnsList, e, lc, key=key, p=p, loss=loss,
                            sample=sample, circuit=circuit, model=model)
                    lcVariables[variableName] = gathered
                    if sample:
                        sampleInfo[variableName] = gathered_sample_info
                    usedVariablesNames.add(variableName)

                if isinstance(e, LcElement):

                    if isinstance(e, CandidateSelection):
                        lcVariablesDnsNew = self.constructCompiled(
                            e, booleanProcessor, dn, key=key,
                            lcVariablesDns=lcVariablesDns, lcVariables=lcVariables,
                            headLC=False, vNo=vNo, plan=step.child,
                            model=model, p=p, loss=loss, sample=sample,
                            verify=verify, circuit=circuit,
                            concept_bindings=bound_concepts)

                        lcVariablesDns = lcVariablesDnsNew
                        vDns = None
                        if lcVariablesDns:
                            length_of_list = len(next(iter(lcVariablesDns.values())))
                            if sample:
                                sample_width = getattr(
                                    booleanProcessor, 'sampleSize', p)
                                vDns = [[torch.ones(
                                    sample_width, device=self.current_device,
                                    requires_grad=False, dtype=torch.bool)]
                                    for _ in range(length_of_list)]
                            elif loss:
                                vDns = [[torch.zeros(
                                    length_of_list, device=self.current_device,
                                    requires_grad=True, dtype=self._get_dtype())]]
                                vDns = self.addLossTovDns(True, vDns)
                            else:
                                vDns = [[1] for _ in range(length_of_list)]

                    if isinstance(e, LogicalConstrain):
                        self.myLogger.info('Processing Nested %r - %s' % (e, e.strEs()))

                        nested = self.constructCompiled(
                            e, booleanProcessor, dn, key=key,
                            lcVariablesDns=lcVariablesDns, lcVariables=lcVariables,
                            headLC=False, vNo=vNo, label=label, plan=step.child,
                            model=model, p=p, loss=loss, sample=sample,
                            verify=verify, circuit=circuit,
                            concept_bindings=bound_concepts)
                        if sample:
                            vDns, nested_sample_info, nested_sets, lcVariableUpdated = nested
                            sampleInfo.update(nested_sample_info)
                            lcVariablesSet.update(nested_sets)
                        else:
                            vDns, lcVariableUpdated = nested
                        if verify and not loss and not sample:
                            flattened = []
                            for row in vDns:
                                if isinstance(row, list):
                                    for item in row:
                                        if isinstance(item, list):
                                            flattened.extend([subitem] for subitem in item)
                                        else:
                                            flattened.append([item])
                                else:
                                    flattened.append([row])
                            if flattened:
                                vDns = flattened
                        vDns = self.addLossTovDns(loss, vDns)
                        lcVariables = lcVariableUpdated

                    if vDns is None:
                        self.myLogger.warning(
                            'Not found data for %s(%s) nested Logical Constraint required to build %s(%s) - skipping it'
                            % (e.lcName, e, lc.lcName, lc))
                        return None

                    lcVariables[variableName] = vDns
                    usedVariablesNames.add(variableName)

            elif isinstance(e, (int, str)):
                pass
            else:
                self.myLogger.error('Logical Constraint %s has incorrect element %s' % (lc, e))
                return None

        for referedVariable in newVariables:
            refVarSet = newVariables[referedVariable]
            refVarSet.add(referedVariable)
            lcVariables = self.eliminate_duplicate_columns(lcVariablesDns, refVarSet, lcVariables)

        useLcVariables = {k: v for k, v in lcVariables.items() if k in usedVariablesNames}

        isEntitySelector = isinstance(lc, (iotaL, miotaL))
        if isEntitySelector:
            self.fillPathBindings(useLcVariables, lcVariableVs,
                                  lcVariablesDns, lcVariableBindings)
            useLcVariables = self.reduceSelectorToPrimaryGrounding(
                lc, useLcVariables, lcVariableBindings, booleanProcessor, model)

        if isinstance(lc, CandidateSelection):
            return lc(lcVariablesDns, keys=lc.CandidateSelectionVariable)

        # Same per-element alignment the interpreter applies in loss mode.
        if loss and not sample and not isEntitySelector:
            self.fillPathBindings(useLcVariables, lcVariableVs,
                                  lcVariablesDns, lcVariableBindings)
            useLcVariables = self.reduceToCommonGrounding(
                useLcVariables, lcVariableBindings, booleanProcessor)

            split_tensors = any(
                useLcVariables[v] and len(useLcVariables[v]) > 1
                for v in useLcVariables)
            if split_tensors:
                for v in useLcVariables:
                    if useLcVariables[v] and len(useLcVariables[v]) > 1:
                        continue
                    # Split every column in parallel; doing only column 0 would
                    # re-collapse a bare EnumConcept.
                    useLcVariables[v] = self.splitLossColumns(useLcVariables[v])

        if headLC and (circuit or (loss and not sample)):
            self._feature_operands = OrderedDict(
                (name, self._probability_layout(value))
                for name, value in useLcVariables.items())

        # sumL is the one operator that needs the runtime label (its target
        # count); mirror the interpreter's conditional kwarg exactly.
        output = lc(
            model, booleanProcessor, useLcVariables, headConstrain=headLC,
            integrate=integrate,
            **({"label": label} if isinstance(lc, sumL) else {}))
        if sample:
            lcVariablesSet[lc] = useLcVariables
            return output, sampleInfo, lcVariablesSet, lcVariables
        return output, lcVariables


class CompiledModeExecutor:
    """Bind shared formula plans to one DataNode for non-fuzzy processors."""

    def __init__(self, solver):
        self.solver = solver
        self.plan_cache = shared_plan_cache(solver)
        self._store = None
        self._evaluator = None
        self._bound_dn = None
        self._bound_key = None
        self.bindings = 0
        self._execution_lock = RLock()

    def bind(self, dn, key='/local/softmax'):
        with self._execution_lock:
            if self._bound_dn is dn and self._bound_key == key:
                return self
            if self._store is None:
                self._store = ProbabilityStore(
                    dn, '/local/softmax', graphs=self.solver.myGraph)
                self._evaluator = CompiledConstraintEvaluator(
                    self.solver.myLogger, self._store,
                    plan_cache=self.plan_cache)
            else:
                self._store.rebind(
                    dn, '/local/softmax', graphs=self.solver.myGraph)
                self._evaluator.rebind(self._store)
            self._bound_dn = dn
            self._bound_key = key
            self.bindings += 1
        return self

    def construct(self, lc, processor, dn, *, key, headLC=True, label=None,
                  model=None, p=0, loss=False, sample=False, verify=False,
                  circuit=False, concept_bindings=None):
        with self._execution_lock:
            self.bind(dn, key)
            self._evaluator.current_device = dn.current_device
            self._evaluator.current_dtype = getattr(dn, 'current_dtype', None)
            return self._evaluator.constructCompiled(
                lc, processor, dn, key=key, headLC=headLC, label=label,
                model=model, p=p, loss=loss, sample=sample, verify=verify,
                circuit=circuit, concept_bindings=concept_bindings)

    def cache_info(self):
        info = self.plan_cache.info()
        info.update({
            'data_bindings': self.bindings,
            'tensorized_candidate_calls': (
                0 if self._evaluator is None
                else self._evaluator.candidateResolver.tensorized_calls),
            'candidate_fallback_calls': (
                0 if self._evaluator is None
                else self._evaluator.candidateResolver.fallback_calls),
        })
        return info

    def reset_grounding_features(self):
        if self._evaluator is not None:
            self._evaluator._gathered_probs = []
            self._evaluator._feature_operands = None

    def grounding_features(self, num_groundings):
        if self._evaluator is None:
            raise RuntimeError("compiled executor is not bound")
        return self._evaluator.grounding_features(num_groundings)


class CompiledLossCalculator(LossCalculator):
    """``LossCalculator`` variant using the compiled evaluator when possible.

    Built-in and custom formula subclasses share the compiled formula protocol.
    The interpreter is retained only as a correctness fallback for an unexpected
    runtime incompatibility while executing a compiled plan.
    """

    def __init__(self, solver, tnorm_selector=None):
        super().__init__(solver, tnorm_selector)
        self._prob_store = None
        self._evaluator = None
        self.plan_cache = shared_plan_cache(solver)
        self.data_bindings = 0
        self.batched_formula_groups = 0
        self.batched_formula_constraints = 0
        self.batched_formula_groundings = 0
        self.batched_formula_fallbacks = 0
        self.batched_formula_plan_hits = 0
        self.batch_index_rebuilds = 0
        self.graph_snapshot_rebuilds = 0
        self.active_rule_cache_hits = 0
        self.active_rule_cache_misses = 0
        self._active_rule_cache_limit = 4096
        self._graph_snapshot_token = None
        self._global_constraints_by_graph = OrderedDict()
        self._batch_index_token = None
        self._batch_plans_by_graph = OrderedDict()
        self._batch_plan_ids = set()
        self._batch_plan_by_id = OrderedDict()
        # The persistent evaluator is rebound per data item. Protect that small
        # mutable execution context when callers share a solver across threads.
        self._execution_lock = RLock()

    @staticmethod
    def _labels_for_template_bindings(label, binding_count, lc_name):
        if binding_count == 1:
            return (label,)
        if not torch.is_tensor(label) or label.dim() == 0:
            raise ValueError(
                f"Parameterized executable {lc_name} has {binding_count} "
                "bindings but its runtime label has no matching leading axis")
        if label.shape[0] != binding_count:
            raise ValueError(
                f"Parameterized executable {lc_name} has {binding_count} "
                f"bindings but {label.shape[0]} runtime labels")
        return tuple(label[index] for index in range(binding_count))

    @staticmethod
    def _merge_template_results(results):
        """Merge per-binding outputs into the existing executable result shape."""
        results = [result for result in results if result is not None]
        if not results:
            return None
        if len(results) == 1:
            return results[0]

        merged = dict(results[0])
        for key in (
            'conversionSigmoid',
            'selectionDistribution',
            'queryDistribution',
            'queryAnswer',
        ):
            values = [result.get(key) for result in results]
            if all(torch.is_tensor(value) for value in values):
                merged[key] = torch.stack(values, dim=0)

        loss_tensors = [
            result.get('lossTensor') for result in results
            if torch.is_tensor(result.get('lossTensor'))
        ]
        if len(loss_tensors) == len(results):
            merged['lossTensor'] = torch.cat(
                [value.reshape(-1) for value in loss_tensors])

        features = [
            result.get('groundingFeatures') for result in results
            if torch.is_tensor(result.get('groundingFeatures'))
        ]
        if len(features) == len(results):
            merged['groundingFeatures'] = torch.cat(features, dim=0)

        losses = [
            result.get('loss') for result in results
            if torch.is_tensor(result.get('loss'))
        ]
        merged['loss'] = (
            torch.stack([value.reshape(()) for value in losses]).mean()
            if len(losses) == len(results) else None
        )
        merged['elapsedInMsLC'] = sum(
            float(result.get('elapsedInMsLC', 0.0)) for result in results)
        merged['templateInstanceCount'] = len(results)
        return merged

    def calculateLoss(self, dn, tnorm='L', counting_tnorm=None,
                      include_executable=False, include_global=True):
        with self._execution_lock:
            return self._calculate_bound_loss(
                dn, tnorm, counting_tnorm, include_executable, include_global)

    def bind(self, dn):
        """Prepare one DataNode binding for repeated single-formula calls."""
        with self._execution_lock:
            self._bind_data(dn)
        return self

    def _calculate_bound_loss(self, dn, tnorm='L', counting_tnorm=None,
                              include_executable=False, include_global=True):
        self._bind_data(dn)
        if include_global:
            self._refresh_batch_index()
        return self._calculate_loss_with_formula_batches(
            dn, tnorm, counting_tnorm,
            include_executable=include_executable,
            include_global=include_global)

    def _bind_data(self, dn):
        self._refresh_graph_snapshots()

        # The graphs are handed to the store so it can honour ``fixedL`` per
        # concept; previously any active fixedL disabled compilation for the
        # whole graph, including constraints that never touch the pinned concept.
        if self._prob_store is None:
            self._prob_store = ProbabilityStore(
                dn, "/local/softmax", graphs=self.solver.myGraph)
        else:
            self._prob_store.rebind(
                dn, "/local/softmax", graphs=self.solver.myGraph)
        if self._evaluator is None:
            self._evaluator = CompiledConstraintEvaluator(
                self.solver.myLogger, self._prob_store,
                plan_cache=self.plan_cache)
        else:
            self._evaluator.rebind(self._prob_store)
        self.data_bindings += 1

    @staticmethod
    def _constraint_revision(graph):
        revision = getattr(graph, 'constraint_revision', None)
        if revision is not None:
            return revision
        return (
            tuple(
                (id(lc), getattr(lc, '_compile_revision', 0))
                for lc in graph.logicalConstrains.values()),
            tuple(
                (id(executable),
                 getattr(getattr(executable, 'innerLC', None),
                         '_compile_revision', 0))
                for executable in graph.executableLCs.values()),
        )

    def _refresh_graph_snapshots(self):
        graphs = tuple(self.solver.myGraph)
        token = tuple(
            (id(graph), self._constraint_revision(graph))
            for graph in graphs
        )
        if token == self._graph_snapshot_token:
            return

        snapshots = OrderedDict()
        live_constraints = []
        for graph in graphs:
            constraints = tuple(graph.logicalConstrains.values())
            snapshots[graph] = constraints
            live_constraints.extend(constraints)
            live_constraints.extend(
                executable.innerLC
                for executable in graph.executableLCs.values()
                if getattr(executable, 'innerLC', None) is not None
            )

        self.plan_cache.retain(live_constraints)
        self._global_constraints_by_graph = snapshots
        self._graph_snapshot_token = token
        self._batch_index_token = None
        self.graph_snapshot_rebuilds += 1

    def _refresh_batch_index(self):
        """Compile the graph's unary implication adjacency once.

        The token is cheap to check and includes each formula revision, so
        structural edits rebuild the index before the changed rule can run.
        """
        token = self._graph_snapshot_token
        if token == self._batch_index_token:
            return

        plans_by_graph = OrderedDict()
        plan_ids = set()
        plans_by_id = OrderedDict()
        for graph, constraints in self._global_constraints_by_graph.items():
            graph_index = {
                'all': [],
                'by_source': {},
                'fallback': [],
                'active_batch_cache': OrderedDict(),
                'active_fallback_cache': OrderedDict(),
            }
            plans_by_graph[graph] = graph_index
            for lc in constraints:
                formula_plan = self.plan_cache.get(lc)
                batch_plan = formula_plan.batched_unary_implication
                if batch_plan is None:
                    graph_index['fallback'].append(lc)
                    continue
                graph_index['all'].append(batch_plan)
                graph_index['by_source'].setdefault(
                    batch_plan.source_concept, []).append(batch_plan)
                plan_ids.add(id(lc))
                plans_by_id[id(lc)] = batch_plan

        self._batch_index_token = token
        self._batch_plans_by_graph = plans_by_graph
        self._batch_plan_ids = plan_ids
        self._batch_plan_by_id = plans_by_id
        self.batch_index_rebuilds += 1

    def _cache_active_rules(self, cache, key, build):
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            self.active_rule_cache_hits += 1
            return cached
        selected = tuple(build())
        cache[key] = selected
        if len(cache) > self._active_rule_cache_limit:
            cache.popitem(last=False)
        self.active_rule_cache_misses += 1
        return selected

    @staticmethod
    def _compiled_vector(gathered):
        """Extract the one-column vector produced by a compiled literal."""
        if (
            isinstance(gathered, list) and len(gathered) == 1
            and isinstance(gathered[0], list) and len(gathered[0]) == 1
            and torch.is_tensor(gathered[0][0])
            and gathered[0][0].dim() == 1
        ):
            return gathered[0][0]
        return None

    def _active_batch_plans(self):
        """Select active rules through the graph's concept adjacency index."""
        selected = []
        for graph, index in self._batch_plans_by_graph.items():
            controller = graph._activation_controller()
            if controller is None:
                key = None
            else:
                key = controller._active_concepts

            def build_candidates():
                if controller is None:
                    return index['all']
                active = controller._active_concepts
                return (
                    plan
                    for source in active
                    for plan in index['by_source'].get(source, ())
                    if plan.target_concept in active
                )

            candidates = self._cache_active_rules(
                index['active_batch_cache'], key, build_candidates)

            for plan in candidates:
                lc = plan.lc
                if not lc.headLC or not lc.declared_active:
                    continue
                selected.append(plan)
        return selected

    def _active_fallback_constraints(self, graph, index):
        controller = graph._activation_controller()
        key = None if controller is None else controller._active_concepts

        def build_candidates():
            if controller is None:
                return index['fallback']
            return (
                lc for lc in index['fallback']
                if graph.are_concepts_active(lc.getLcConcepts())
            )

        candidates = self._cache_active_rules(
            index['active_fallback_cache'], key, build_candidates)
        return (
            lc for lc in candidates
            if lc.headLC and lc.declared_active
        )

    def _evaluate_formula_batches(self, selector):
        """Evaluate active unary implication rules as ``[R, G]`` tensors."""
        myBooleanMethods = self.solver.myLcLossBooleanMethods
        literal_cache = {}
        buckets = OrderedDict()
        fallback_ids = set()

        for plan in self._active_batch_plans():
            lc = plan.lc
            selected_tnorm = selector.select(lc=lc)
            if selected_tnorm not in ('L', 'G', 'P'):
                fallback_ids.add(id(lc))
                self.batched_formula_fallbacks += 1
                continue

            dns = tuple(self._prob_store.concept_datanodes(
                plan.source_concept.name))
            if not dns:
                fallback_ids.add(id(lc))
                self.batched_formula_fallbacks += 1
                continue
            grounding_key = tuple(id(node) for node in dns)

            vectors = []
            for element in (plan.source_element, plan.target_element):
                literal_key = (id(element[0]), element[2], grounding_key)
                vector = literal_cache.get(literal_key)
                if vector is None:
                    vector = self._prob_store.gather_identity_vector(dns, element)
                    if vector is not None:
                        literal_cache[literal_key] = vector
                vectors.append(vector)

            source, target = vectors
            if (
                source is None or target is None
                or source.shape != target.shape
                or source.numel() == 0
            ):
                fallback_ids.add(id(lc))
                self.batched_formula_fallbacks += 1
                continue

            bucket_key = (
                selected_tnorm, grounding_key, source.device, source.dtype)
            buckets.setdefault(bucket_key, []).append(
                (plan, source, target))

        results = {}
        for (selected_tnorm, _grounding, _device, _dtype), entries in buckets.items():
            started = perf_counter_ns()
            source = torch.stack([entry[1] for entry in entries], dim=0)
            target = torch.stack([entry[2] for entry in entries], dim=0)
            myBooleanMethods.setTNorm(selected_tnorm)
            try:
                losses = myBooleanMethods.ifVarBatched(
                    None, source, target, onlyConstrains=True)
            except Exception as exc:
                self.solver.myLogger.warning(
                    'Batched implication evaluation failed (%s: %s) - '
                    'falling back per constraint', type(exc).__name__, exc)
                for plan, _, _ in entries:
                    fallback_ids.add(id(plan.lc))
                self.batched_formula_fallbacks += len(entries)
                continue

            elapsed = (perf_counter_ns() - started) / 1_000_000
            per_constraint_elapsed = elapsed / len(entries)
            self.batched_formula_groups += 1
            self.batched_formula_constraints += len(entries)
            self.batched_formula_groundings += losses.numel()
            # These two identity bindings are candidate resolutions too; keep
            # the existing aggregate diagnostic meaningful for batched plans.
            self._evaluator.candidateResolver.tensorized_calls += 2 * len(entries)

            for row, (plan, source_row, target_row) in enumerate(entries):
                loss_vector = losses[row]
                mean_loss = loss_vector.mean()
                results[id(plan.lc)] = {
                    'lc': plan.lc,
                    'tnorm_used': selected_tnorm,
                    'constraint_type': get_constraint_type(plan.lc),
                    'lossTensor': loss_vector,
                    'groundingFeatures': torch.stack(
                        (source_row.detach(), target_row.detach()), dim=-1),
                    'loss': mean_loss,
                    'conversionSigmoid': 1.0 - mean_loss,
                    'elapsedInMsLC': per_constraint_elapsed,
                    'batchedFormula': True,
                }

        return results, fallback_ids

    def _calculate_loss_with_formula_batches(
            self, dn, tnorm='L', counting_tnorm=None,
            include_executable=False, include_global=True):
        """LossCalculator loop with one grouped fast path for unary ``ifL``."""
        myBooleanMethods = self.solver.myLcLossBooleanMethods
        myBooleanMethods.current_device = dn.current_device
        myBooleanMethods.current_dtype = self.solver.constraintConstructor.current_dtype
        self._evaluator.current_device = self.solver.current_device
        self._evaluator.current_dtype = self.solver.constraintConstructor.current_dtype

        self.solver.myLogger.info(
            'Calculating compiled loss with batched unary implications')
        key = "/local/softmax"
        lcLosses = {}
        dn.setActiveExecutableLCs()
        selector = (
            self._external_selector
            if self._external_selector is not None
            else TNormSelector(tnorm, counting_tnorm)
        )
        if include_global:
            batched, batch_fallbacks = self._evaluate_formula_batches(selector)
        else:
            batched, batch_fallbacks = {}, set()
        if self.data_bindings > 1:
            self.batched_formula_plan_hits += len(batched)

        if include_global:
            for result in batched.values():
                lc = result['lc']
                lcLosses[lc.lcName] = result

            for graph, index in self._batch_plans_by_graph.items():
                for lc in self._active_fallback_constraints(graph, index):
                    result = self.calculate_single_lc_loss(
                        lc, dn, key, tnorm=tnorm,
                        counting_tnorm=counting_tnorm, label=None)
                    if result is not None:
                        lcLosses[lc.lcName] = result

            for lc_id, plan in self._batch_plan_by_id.items():
                if lc_id not in batch_fallbacks:
                    continue
                lc = plan.lc
                result = self.calculate_single_lc_loss(
                    lc, dn, key, tnorm=tnorm,
                    counting_tnorm=counting_tnorm, label=None)
                if result is not None:
                    lcLosses[lc.lcName] = result

        for graph in self.solver.myGraph:
            if not include_executable:
                continue
            for executable_name in dn.getActiveExecutableConstraintNames():
                executable = graph.executableLCs.get(executable_name)
                if executable is None:
                    continue
                label = dn.getExecutableConstraintLabel(executable_name)
                inner = getattr(executable, 'innerLC', None)
                if label is None or inner is None:
                    continue
                old_head = inner.headLC
                inner.headLC = True
                try:
                    bindings = dn.getExecutableConstraintBindings(
                        executable_name)
                    if bindings:
                        labels = self._labels_for_template_bindings(
                            label, len(bindings), executable_name)
                        result = self._merge_template_results([
                            self.calculate_single_lc_loss(
                                inner, dn, key, tnorm=tnorm,
                                counting_tnorm=counting_tnorm,
                                label=instance_label,
                                concept_bindings=binding)
                            for binding, instance_label in zip(bindings, labels)
                        ])
                    else:
                        result = self.calculate_single_lc_loss(
                            inner, dn, key, tnorm=tnorm,
                            counting_tnorm=counting_tnorm, label=label)
                finally:
                    inner.headLC = old_head
                if result is not None:
                    result['executableName'] = executable_name
                    lcLosses[executable_name] = result

        return lcLosses

    def cache_info(self):
        """Return persistent-plan and current binding diagnostics."""
        info = self.plan_cache.info()
        info.update({
            'data_bindings': self.data_bindings,
            'tensorized_candidate_calls': (
                0 if self._evaluator is None
                else self._evaluator.candidateResolver.tensorized_calls),
            'candidate_fallback_calls': (
                0 if self._evaluator is None
                else self._evaluator.candidateResolver.fallback_calls),
            'batched_formula_groups': self.batched_formula_groups,
            'batched_formula_constraints': self.batched_formula_constraints,
            'batched_formula_groundings': self.batched_formula_groundings,
            'batched_formula_fallbacks': self.batched_formula_fallbacks,
            'batched_formula_plan_hits': self.batched_formula_plan_hits,
            'batch_index_rebuilds': self.batch_index_rebuilds,
            'graph_snapshot_rebuilds': self.graph_snapshot_rebuilds,
            'active_rule_cache_hits': self.active_rule_cache_hits,
            'active_rule_cache_misses': self.active_rule_cache_misses,
            'fixed_index_rebuilds': (
                0 if self._prob_store is None
                else self._prob_store.fixed_index_rebuilds),
            'fixed_candidate_checks': (
                0 if self._prob_store is None
                else self._prob_store.fixed_candidate_checks),
        })
        return info

    def calculate_single_lc_loss(
            self, lc, dn, key, tnorm='L', counting_tnorm=None, label=None,
            concept_bindings=None):
        if self._prob_store is None or not lc_tree_supported(lc):
            if concept_bindings is not None:
                raise RuntimeError(
                    f"Parameterized executable {lc.lcName} requires the "
                    "compiled formula protocol")
            return super().calculate_single_lc_loss(
                lc, dn, key, tnorm=tnorm, counting_tnorm=counting_tnorm, label=label)

        if not lc.headLC or not lc.active:
            return None
        if type(lc) is fixedL:
            return None

        selector = self._external_selector if self._external_selector is not None else TNormSelector(tnorm, counting_tnorm)

        start = perf_counter_ns()
        myBooleanMethods = self.solver.myLcLossBooleanMethods
        result = {'lc': lc, 'compiled': True}

        selected_tnorm = selector.select(lc=lc)
        result['tnorm_used'] = selected_tnorm
        result['constraint_type'] = get_constraint_type(lc)

        myBooleanMethods.setTNorm(selected_tnorm)

        # sumL needs its runtime target count; the interpreter reads it from the
        # constraint datanode when the caller did not supply one.
        if isinstance(lc, sumL) and label is None:
            _rawLabel = dn.getExecutableConstraintLabel(lc.lcName)
            if _rawLabel is None:
                return None
            label = _rawLabel.float()
        elif isinstance(lc, queryL) and lc.is_multi_answer and label is None:
            label = dn.getExecutableConstraintLabel(lc.lcName)

        self.solver.myLogger.info(f'Processing {lc} (compiled) with t-norm {selected_tnorm}')

        self.solver.constraintConstructor.current_device = self.solver.current_device
        self.solver.constraintConstructor.myGraph = self.solver.myGraph

        self._evaluator.current_device = self.solver.current_device
        self._evaluator.current_dtype = self.solver.constraintConstructor.current_dtype
        self._evaluator._gathered_probs = []  # reset per head constraint
        self._evaluator._feature_operands = None

        try:
            if isinstance(lc, miotaL):
                selection_output, _ = self._evaluator.constructCompiled(
                    lc, myBooleanMethods, dn, key=key, headLC=False,
                    concept_bindings=concept_bindings)
                selection_distribution = self._normalize_selection_distribution(
                    selection_output)
                result['selectionDistribution'] = selection_distribution
                result['lossTensor'] = None
                result['conversionSigmoid'] = selection_distribution
                result['loss'] = None
                result['elapsedInMsLC'] = (perf_counter_ns() - start) / 1_000_000
                return result

            if isinstance(lc, queryL):
                # queryL is evaluated as a non-head expression and post-processed
                # into a class distribution rather than a violation vector.
                query_output, _ = self._evaluator.constructCompiled(
                    lc, myBooleanMethods, dn, key=key, headLC=False,
                    concept_bindings=concept_bindings)
                if lc.is_multi_answer:
                    query_distribution = self._normalize_multi_query_distribution(
                        query_output, lc.num_subclasses)
                else:
                    query_distribution = self._normalize_query_distribution(
                        query_output, lc.num_subclasses)
                result['queryDistribution'] = query_distribution
                if lc.is_multi_answer:
                    result['queryAnswer'] = self._decode_multi_query(
                        query_distribution, lc.threshold)
                result['lossTensor'] = None
                if query_distribution is not None:
                    result['conversionSigmoid'] = query_distribution
                    if lc.is_multi_answer and label is not None:
                        _target, probability, losses, mean_loss = multi_query_joint_nll(
                            query_distribution,
                            label,
                            lc.num_subclasses,
                            label_name=f"multi-answer queryL {lc.lcName}",
                        )
                        result['probability'] = probability
                        result['lossTensor'] = losses
                        result['loss'] = mean_loss
                    else:
                        result['loss'] = (
                            None if lc.is_multi_answer
                            else 1.0 - query_distribution.max()
                        )
                else:
                    result['conversionSigmoid'] = None
                    result['loss'] = None
                result['elapsedInMsLC'] = (perf_counter_ns() - start) / 1_000_000
                return result

            lossTensor = self._evaluator.constructCompiled(
                lc, myBooleanMethods, dn, key=key, headLC=True,
                label=int(label) if label is not None else None,
                concept_bindings=concept_bindings)
        except Exception as exc:  # fall back to the interpreter on any failure
            if concept_bindings is not None:
                raise RuntimeError(
                    f"Compiled parameterized evaluation of {lc.lcName} failed; "
                    "the interpreter cannot apply runtime concept slots") from exc
            self.solver.myLogger.warning(
                'Compiled evaluation of %s failed (%s: %s) - falling back to interpreter',
                lc.lcName, type(exc).__name__, exc)
            return super().calculate_single_lc_loss(
                lc, dn, key, tnorm=tnorm, counting_tnorm=counting_tnorm, label=label)

        # Propagate the detected dtype so later tensor creation matches.
        if self._evaluator.current_dtype is None and self._prob_store.dtype is not None:
            self._evaluator.current_dtype = self._prob_store.dtype

        normalized = self._normalize_loss_list(lossTensor)
        result['lossTensor'] = normalized

        # Per-grounding literal-probability features for the amortized DualCritic
        # (R5B). Every gathered layout must align with the final loss vector;
        # alignment failures are surfaced rather than silently zero-filled.
        if normalized is not None:
            result['groundingFeatures'] = self._evaluator.grounding_features(normalized.shape[0])

        if normalized is None:
            lossTensor = None
        else:
            lossTensor = normalized.mean()

        result['loss'] = lossTensor

        if lossTensor is not None:
            result['conversionSigmoid'] = 1.0 - lossTensor
            if isinstance(lc, sumL) or (hasattr(lc, 'innerLC') and isinstance(lc.innerLC, sumL)):
                result['expectedCount'] = lossTensor
        else:
            result['conversionSigmoid'] = None

        result['elapsedInMsLC'] = (perf_counter_ns() - start) / 1_000_000
        return result
