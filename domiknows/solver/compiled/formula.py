"""Formula-stage of the compiled logical-constraint loss path.

``CompiledConstraintEvaluator`` is a faithful port of
``LogicalConstraintConstructor.constructLogicalConstrains`` restricted to the
loss path (``loss=True, sample=False, verify=False, m=None``):

* candidate/path resolution reuses ``getCandidates`` unchanged (it is purely
  structural — forward relations, ``.reversed`` paths, eqL filters, expansion);
* per-datanode probability reads are replaced by batched gathers through
  ``ProbabilityStore`` (``grounding.py``), producing the exact tensor layouts
  the interpreter builds (stack-then-fallback, first-candidate-per-group);
* the t-norm operators themselves are the original ``lcLossBooleanMethods``
  implementations invoked through each constraint's ``__call__`` — the
  numerics are shared with the interpreter by construction, not duplicated.

``CompiledLossCalculator`` mirrors ``LossCalculator`` and falls back to the
interpreter per constraint for unsupported types (eqL-as-element, ``execute``
wrappers) or on any evaluation error. ``fixedL`` no longer disables the whole
graph: the pinned concept's hard 0/1 substitution is replicated per concept in
``ProbabilityStore`` (see ``_fixed_spec`` / ``_apply_fixed``).
"""

from collections import OrderedDict
from time import perf_counter_ns

import torch

from domiknows.graph import LcElement, LogicalConstrain, V, CandidateSelection, fixedL
from domiknows.graph.concept import Concept
from domiknows.graph.candidates import getCandidates
from domiknows.graph.logicalConstrain import (
    notL, andL, orL, nandL, ifL, norL, xorL, equivalenceL, forAllL,
    atMostL, atLeastL, exactL, existsL,
    atMostAL, atLeastAL, exactAL, existsAL,
    greaterL, greaterEqL, lessL, lessEqL, equalCountsL, notEqualCountsL,
    sumL, queryL, iotaL, sameL, differentL,
)
from domiknows.solver.logicalConstraintConstructor import LogicalConstraintConstructor
from domiknows.solver.lossCalculator import LossCalculator
from domiknows.solver.adaptiveTNormLossCalculator import TNormSelector, get_constraint_type

from .grounding import ProbabilityStore


#: Constraint types the compiled evaluator handles.  Anything else — and any
#: tree containing anything else — is delegated to the interpreter.
SUPPORTED_LC_TYPES = (
    notL, andL, orL, nandL, ifL, norL, xorL, equivalenceL, forAllL,
    atMostL, atLeastL, exactL, existsL,
    atMostAL, atLeastAL, exactAL, existsAL,
    greaterL, greaterEqL, lessL, lessEqL, equalCountsL, notEqualCountsL,
    sumL, sameL, differentL, iotaL, queryL,
)


def lc_tree_supported(lc):
    """True when the whole constraint tree consists of supported LC types."""
    if not isinstance(lc, SUPPORTED_LC_TYPES):
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

    def __init__(self, logger, probStore: ProbabilityStore):
        super().__init__(logger)
        self.probStore = probStore
        # Per-grounding classifier probability tensors gathered while evaluating
        # the current head constraint — exported as `groundingFeatures` for the
        # amortized DualCritic (R5B). Reset per head constraint.
        self._gathered_probs = []

    def _record_gathered(self, gathered):
        """Record a [G]-shaped gathered probability tensor (best-effort).

        ``gather_variable`` returns ``[[tensor]]`` on its batched fast path
        (tensor is one probability per grounding group); that is the only shape
        that aligns with the constraint's per-grounding loss vector, so other
        (fallback) layouts are simply skipped — the critic zero-fills when no
        features align.
        """
        try:
            if (isinstance(gathered, list) and len(gathered) == 1
                    and isinstance(gathered[0], list) and len(gathered[0]) == 1):
                t = gathered[0][0]
                if torch.is_tensor(t) and t.dim() == 1:
                    self._gathered_probs.append(t.detach())
        except Exception:  # noqa: BLE001 - features are optional
            pass

    def grounding_features(self, num_groundings):
        """Stack the gathered [G]-tensors matching ``num_groundings`` into [G, L]."""
        if not num_groundings:
            return None
        feats = [t for t in self._gathered_probs if t.shape[0] == num_groundings]
        if not feats:
            return None
        return torch.stack(feats, dim=-1)

    def _get_dtype(self):
        if self.current_dtype is not None:
            return self.current_dtype
        if self.probStore.dtype is not None:
            return self.probStore.dtype
        return torch.float32

    def constructCompiled(self, lc, booleanProcessor, dn, key=None,
                          lcVariablesDns=None, lcVariables=None, headLC=False,
                          vNo=None, label=None):
        """Port of ``constructLogicalConstrains`` for the loss path.

        Returns the same ``(lc_result, lcVariables)`` tuple the interpreter
        returns (or the candidate dict for ``CandidateSelection`` elements).
        """
        if key is None:
            key = ""

        if lcVariablesDns is None:
            lcVariablesDns = OrderedDict()
        if lcVariables is None:
            lcVariables = OrderedDict()

        usedVariablesNames = set()
        # Mirrors the interpreter's grounding bookkeeping so operands
        # enumerated over different variable tuples are aligned identically.
        lcVariableBindings = OrderedDict()
        lcVariableVs = OrderedDict()

        if vNo is None:
            vNo = [1, 1]

        firstV = None
        integrate = False
        newVariables = {}

        iter_es = lc.e

        for eIndex, e in enumerate(iter_es):
            if isinstance(e, V):
                continue

            if isinstance(e, (Concept, LcElement, tuple)):
                # Look ahead for the variable name (same auto-naming as the
                # interpreter so paths referring to earlier variables align).
                if eIndex + 1 < len(iter_es) and isinstance(iter_es[eIndex + 1], V):
                    variable = iter_es[eIndex + 1]
                else:
                    if isinstance(e, LogicalConstrain):
                        variable = V(name="_lc" + str(vNo[1]))
                        vNo[1] += 1
                    elif isinstance(e, tuple) and isinstance(e[0], CandidateSelection):
                        e[0].CandidateSelectionVariable = e[1]
                        e = e[0]
                        variable = V(name="_cs" + str(vNo[1]))
                        vNo[1] += 1
                    else:
                        if firstV is None:
                            variable = V(name="_x" + str(vNo[0]))
                            if not isinstance(lc, CandidateSelection):
                                firstV = variable.name
                            vNo[0] += 1
                        else:
                            variable = V(name="_x" + str(vNo[0]), v=(firstV,))
                            vNo[0] += 1

                if variable.name:
                    variableName = variable.name
                else:
                    variableName = "V" + str(vNo[0])
                    vNo[0] += 1

                if variableName in lcVariables:
                    # Re-binding: the variable was used before; reuse its
                    # datanodes but look up the new concept's predictions.
                    newVariableName = "_x" + str(vNo[0])
                    vNo[0] += 1

                    lcVariablesDns[newVariableName] = lcVariablesDns[variableName]

                    is_concept_tuple = isinstance(e, tuple) and len(e) >= 1 and isinstance(e[0], Concept)
                    if is_concept_tuple:
                        gathered = self.probStore.gather_variable(
                            lcVariablesDns[variableName], e)
                        self._record_gathered(gathered)
                        lcVariables[newVariableName] = gathered
                    else:
                        lcVariables[newVariableName] = lcVariables[variableName]
                    usedVariablesNames.add(newVariableName)

                elif isinstance(e, (Concept, tuple)):
                    result = getCandidates(dn, e, variable, lcVariablesDns, lc,
                                           self.myLogger, integrate=integrate)

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

                    gathered = self.probStore.gather_variable(dnsList, e)
                    self._record_gathered(gathered)
                    lcVariables[variableName] = gathered
                    usedVariablesNames.add(variableName)

                if isinstance(e, LcElement):

                    if isinstance(e, CandidateSelection):
                        lcVariablesDnsNew = self.constructCompiled(
                            e, booleanProcessor, dn, key=key,
                            lcVariablesDns=lcVariablesDns, lcVariables=lcVariables,
                            headLC=False, vNo=vNo)

                        lcVariablesDns = lcVariablesDnsNew
                        vDns = None
                        if lcVariablesDns:
                            length_of_list = len(next(iter(lcVariablesDns.values())))
                            vDns = [[torch.zeros(length_of_list, device=self.current_device,
                                                 requires_grad=True, dtype=self._get_dtype())]]
                            vDns = self.addLossTovDns(True, vDns)

                    if isinstance(e, LogicalConstrain):
                        self.myLogger.info('Processing Nested %r - %s' % (e, e.strEs()))

                        vDns, lcVariableUpdated = self.constructCompiled(
                            e, booleanProcessor, dn, key=key,
                            lcVariablesDns=lcVariablesDns, lcVariables=lcVariables,
                            headLC=False, vNo=vNo, label=label)

                        vDns = self.addLossTovDns(True, vDns)
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

        if isinstance(lc, CandidateSelection):
            return lc(lcVariablesDns, keys=lc.CandidateSelectionVariable)

        # Same per-element split the interpreter applies in loss mode when a
        # sibling variable kept a multi-group (fallback) structure.
        self.fillPathBindings(useLcVariables, lcVariableVs,
                              lcVariablesDns, lcVariableBindings)
        useLcVariables = self.reduceToCommonGrounding(
            useLcVariables, lcVariableBindings, booleanProcessor)

        slpitT = False
        for v in useLcVariables:
            if useLcVariables[v] and len(useLcVariables[v]) > 1:
                slpitT = True
                break

        if slpitT:
            for v in useLcVariables:
                if useLcVariables[v] and len(useLcVariables[v]) > 1:
                    continue

                # Splits every column in parallel (see splitLossColumns); doing
                # only column 0 would re-collapse a bare EnumConcept.
                useLcVariables[v] = self.splitLossColumns(useLcVariables[v])

        # sumL is the one operator that needs the runtime label (its target
        # count); mirror the interpreter's conditional kwarg exactly.
        return lc(None, booleanProcessor, useLcVariables, headConstrain=headLC,
                  integrate=integrate,
                  **({"label": label} if isinstance(lc, sumL) else {})), lcVariables


class CompiledLossCalculator(LossCalculator):
    """``LossCalculator`` variant using the compiled evaluator when possible.

    Falls back to the interpreter path (``super().calculate_single_lc_loss``)
    per constraint for unsupported types, for graphs with active ``fixedL``
    constraints (which alter probability lookups globally), and on any
    evaluation error.
    """

    def __init__(self, solver, tnorm_selector=None):
        super().__init__(solver, tnorm_selector)
        self._prob_store = None
        self._evaluator = None

    def calculateLoss(self, dn, tnorm='L', counting_tnorm=None, include_executable=False):
        # The graphs are handed to the store so it can honour ``fixedL`` per
        # concept; previously any active fixedL disabled compilation for the
        # whole graph, including constraints that never touch the pinned concept.
        self._prob_store = ProbabilityStore(
            dn, "/local/softmax", graphs=self.solver.myGraph)
        self._evaluator = CompiledConstraintEvaluator(self.solver.myLogger, self._prob_store)
        return super().calculateLoss(dn, tnorm, counting_tnorm,
                                     include_executable=include_executable)

    def calculate_single_lc_loss(self, lc, dn, key, tnorm='L', counting_tnorm=None, label=None):
        if self._prob_store is None or not lc_tree_supported(lc):
            return super().calculate_single_lc_loss(
                lc, dn, key, tnorm=tnorm, counting_tnorm=counting_tnorm, label=label)

        if not lc.headLC or not lc.active:
            return None
        if type(lc) is fixedL:
            return None

        selector = self._external_selector if self._external_selector is not None else TNormSelector(tnorm, counting_tnorm)

        start = perf_counter_ns()
        myBooleanMethods = self.solver.myLcLossBooleanMethods
        result = {'lc': lc}

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

        self.solver.myLogger.info(f'Processing {lc} (compiled) with t-norm {selected_tnorm}')

        self.solver.constraintConstructor.current_device = self.solver.current_device
        self.solver.constraintConstructor.myGraph = self.solver.myGraph

        self._evaluator.current_device = self.solver.current_device
        self._evaluator.current_dtype = self.solver.constraintConstructor.current_dtype
        self._evaluator._gathered_probs = []  # reset per head constraint

        try:
            if isinstance(lc, queryL):
                # queryL is evaluated as a non-head expression and post-processed
                # into a class distribution rather than a violation vector.
                query_output, _ = self._evaluator.constructCompiled(
                    lc, myBooleanMethods, dn, key=key, headLC=False)
                query_distribution = self._normalize_query_distribution(
                    query_output, lc.num_subclasses)
                result['queryDistribution'] = query_distribution
                result['lossTensor'] = None
                if query_distribution is not None:
                    result['conversionSigmoid'] = query_distribution
                    result['loss'] = 1.0 - query_distribution.max()
                else:
                    result['conversionSigmoid'] = None
                    result['loss'] = None
                result['elapsedInMsLC'] = (perf_counter_ns() - start) / 1_000_000
                return result

            lossTensor = self._evaluator.constructCompiled(
                lc, myBooleanMethods, dn, key=key, headLC=True,
                label=int(label) if label is not None else None)
        except Exception as exc:  # fall back to the interpreter on any failure
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
        # (R5B). Row-aligned with the loss vector; None when nothing aligns.
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
