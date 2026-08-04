"""Model-side integration of the R3/R4 structure mechanisms.

The R-line splits into two families:

* **cmodel side** — R1 ``compile_lc``, R2 :class:`SemanticLossModel`, R5 duals.
  These shape the constraint *loss* and already plug into ``LossProgram``'s
  ``CModel`` slot.
* **model side** — R3 :class:`FactorGraphHead`, R4 :class:`ConstraintRefinement`.
  These change the *forward pass*, and until now had no seam: ``PoiModel``
  runs each concept's sensors and takes an independent loss per concept, with
  nothing in between.

:class:`StructuredModel` is that seam. It follows the shape ``SolverModel``
already established — run sensors, post-process, then compute loss with
``run=False`` — inserting constraint structure into the post-process step.

Because ``Model`` is a parameter of every Program, using this class is all that
is needed to combine R3/R4 with R1/R2/R5::

    PrimalDualProgram(graph, StructuredModel_factory, compile_lc=True,
                      dual_algorithm='augmented')     # R1 + R3/R4 + R5A

How refined beliefs reach the losses
------------------------------------
Two consumers read the beliefs by different routes, so write-back has two
halves (both verified against the framework's own caching contract):

* the **supervised** loss calls ``sensor(builder)``, which returns the value
  cached under the *sensor object* (``Sensor.update_context`` stores
  ``data_item[self] = val`` and returns it unless ``force``). Writing that entry
  with ``dict.__setitem__`` updates it without re-triggering
  ``DataNodeBuilder.__setitem__``'s datanode construction.
* the **constraint** loss reads ``<concept>/local/softmax`` off the datanodes.
  ``inferLocal`` computes that key only ``if not dn.hasAttribute(...)``, so a
  value written first is kept rather than recomputed — including by
  ``SemanticLossModel``, which calls ``inferLocal`` unconditionally.

``belief_flow`` selects how far the refinement propagates; see the class
docstring.
"""

import logging

import torch

from domiknows.graph import DataNodeBuilder
from domiknows.graph.concept import Concept, EnumConcept
from domiknows.solver.bdd import CircuitSizeLimitExceeded
from domiknows.solver.compiled.grounding import ProbabilityStore
from domiknows.program.model.pytorch import SolverModel
from domiknows.program.model.refinement import (
    ConstraintRefinement, build_constraint_factors, _factor_kind, _parse_literals,
)
from domiknows.program.model.factorGraphHead import (
    FactorGraphHead, FactorGraphReport, VariableGroup,
)

BELIEF_FLOWS = ('write_back', 'constraint_only')

_SOFTMAX_KEY = '/local/softmax'


def static_concept_widths(graph):
    """``{concept_name: n_classes}`` for every concept a factor can reference.

    Derived from the constraint *declarations* alone — ``_factor_kind`` and
    ``_parse_literals`` are purely syntactic — so the refinement layer can be
    built in ``__init__`` rather than on the first forward pass.

    That timing matters: ``LearningBasedProgram.train`` builds the optimizer from
    ``model.parameters()`` *before* any forward runs, so a lazily created
    refiner's gate would never be handed to the optimizer and could never learn.
    """
    widths = {}
    rec = getattr(graph, 'logicalConstrainsRecursive', None)
    constraints = rec if rec is not None else getattr(graph, 'logicalConstrains', {}).items()
    for _key, lc in constraints:
        if not getattr(lc, 'headLC', False) or not getattr(lc, 'active', True):
            continue
        kind = _factor_kind(lc)
        if kind is None:
            continue
        specs = _parse_literals(lc, kind)
        if not specs:
            continue
        for concept_tuple, _binding in specs:
            concept = concept_tuple[0]
            widths[concept.name] = (len(concept.enum)
                                    if isinstance(concept, EnumConcept) else 2)
    return widths


class StructuredModel(SolverModel):
    """``SolverModel`` that applies constraint structure inside the forward pass.

    :param refine: apply R4B :class:`ConstraintRefinement` — messages are the
        constraint's own violation gradient, so beliefs move toward satisfaction
        by construction. Vectorised over all groundings; the cheap path.
    :param factor_graph: apply R3 :class:`FactorGraphHead` — replaces beliefs
        with **exact constrained marginals** ``P(y | x, phi)``. Applied per
        grounding (one small circuit per factor row), so it is exact but costs
        more than refinement; budget overruns fall back and are reported.
    :param belief_flow:
        ``'write_back'`` (default) — refined beliefs replace the cached sensor
        outputs *and* the datanode softmax, so the supervised loss, constraint
        loss, metrics and inference all see them. This is what lets constraints
        correct *representations* rather than only the output layer, and it is
        how a structured (CRF-like) layer normally trains.
        ``'constraint_only'`` — only the datanode softmax is updated, so the
        supervised loss keeps the raw head outputs. Fully backward compatible,
        and the ablation for "did the structure actually do the work".
    :param structure_warmup: skip structure until this many *training steps*
        have run, so the heads become meaningful before beliefs are rewritten.
        Evaluation passes do not advance the counter.

    ``report`` carries the enforced/fallback partition after each forward pass.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, graph, poi=None, loss=None, metric=None, inferTypes=None,
                 inference_with=None, probKey=('local', 'softmax'), device='auto',
                 probAcc=None, ignore_modules=False, kwargs=None,
                 *, refine=False, factor_graph=False,
                 belief_flow='write_back', refine_steps=2, refine_step_size=5.0,
                 refine_learn_gate=True, circuit_backend='bdd',
                 circuit_max_nodes=100_000, structure_warmup=0):
        super().__init__(graph, poi=poi, loss=loss, metric=metric,
                         inferTypes=inferTypes, inference_with=inference_with,
                         probKey=probKey, device=device, probAcc=probAcc,
                         ignore_modules=ignore_modules, kwargs=kwargs)
        if belief_flow not in BELIEF_FLOWS:
            raise ValueError(
                f"belief_flow must be one of {BELIEF_FLOWS}, got {belief_flow!r}")

        self.refine = refine
        self.factor_graph = factor_graph
        self.belief_flow = belief_flow
        self.circuit_backend = circuit_backend
        self.circuit_max_nodes = circuit_max_nodes
        self.structure_warmup = structure_warmup
        self._train_steps = 0
        self._concept_props = None

        # Built EAGERLY. The widths come from the constraint declarations, which
        # are available now, and building here is what puts the refiner's gate
        # into ``model.parameters()`` before ``train()`` constructs the
        # optimizer from it. Building it lazily on the first forward left the
        # gate out of the optimizer entirely, so it could never learn.
        self._refine_config = dict(steps=refine_steps, step_size=refine_step_size,
                                   learn_gate=refine_learn_gate)
        self._refiner = None
        if refine:
            widths = static_concept_widths(graph)
            if widths:
                self._refiner = ConstraintRefinement(widths, **self._refine_config)
        self.report = FactorGraphReport()
        #: constraint names whose satisfaction the structure enforces, for the
        #: Program to drop from the loss/duals (see ``StructuredProgram``).
        self.enforced_constraints = set()
        #: formula-shape -> FactorGraphHead. A plain dict (not ModuleDict): the
        #: heads hold no parameters, and keeping them out of ``state_dict``
        #: avoids checkpointing a pure cache. Persisting across forwards is what
        #: makes each constraint's circuit compile once per run, not once per row.
        self._fg_cache = {}

    # ------------------------------------------------------------------ #
    # Belief harvesting
    # ------------------------------------------------------------------ #

    def _graphs(self):
        graph = self.graph
        return [graph] + list(getattr(graph, 'subgraphs', {}).values())

    def _harvest(self, datanode, concept_names):
        """``{concept: {'matrix': [N x K], 'dns': [...]}}`` for the given concepts.

        Reuses :meth:`ProbabilityStore.concept_matrix`, the accessor added for
        exactly this purpose, rather than re-deriving the per-concept layout.
        """
        store = ProbabilityStore(datanode, _SOFTMAX_KEY, graphs=self._graphs())
        beliefs = {}
        for name in concept_names:
            try:
                entry = store.concept_matrix(name)
            except Exception:  # noqa: BLE001 - a concept without predictions
                continue
            if entry['matrix'] is not None and entry['dns']:
                beliefs[name] = entry
        return beliefs

    # ------------------------------------------------------------------ #
    # Write-back
    # ------------------------------------------------------------------ #

    def _write_back(self, builder, datanode, name, entry, refined):
        """Publish refined beliefs to whichever consumers ``belief_flow`` allows."""
        dns = entry['dns']

        # (a) constraint side — the datanode softmax key. inferLocal only fills
        # this when absent, so writing it here survives later recomputation.
        key = f'<{name}>{_SOFTMAX_KEY}'
        for row, dn in enumerate(dns):
            if row < refined.shape[0]:
                dn.attributes[key] = refined[row]

        # Skeleton (batched) layout keeps one stacked tensor on the root.
        variable_set = datanode.attributes.get('variableSet')
        if isinstance(variable_set, dict):
            for vkey in list(variable_set):
                if vkey.endswith(f'/<{name}>{_SOFTMAX_KEY}'):
                    variable_set[vkey] = refined

        if self.belief_flow != 'write_back':
            return

        # (b) supervised side — the sensor cache. dict.__setitem__ bypasses
        # DataNodeBuilder.__setitem__, which would otherwise re-run datanode
        # construction for an already-built graph.
        for prop in self._concept_properties().get(name, ()):
            for output_sensor, _target in self.find_sensors(prop):
                if output_sensor in builder:
                    dict.__setitem__(builder, output_sensor, refined)

    def _concept_properties(self):
        """``{concept name: [Property]}`` over the concept-valued POI properties.

        A property declared as ``ent[a] = ModuleLearner(...)`` stores the
        *Concept object* in ``prop_name`` (plain string properties like ``'emb'``
        keep a ``str``), so matching on that is an identity check rather than
        string-munging the property's fullname.
        """
        if self._concept_props is None:
            mapping = {}
            # Cache the complete schema.  Activation can change between
            # forwards, so filtering here would permanently omit concepts that
            # become active in a later step.
            for prop in self.poi:
                prop_name = getattr(prop, 'prop_name', None)
                if isinstance(prop_name, Concept):
                    mapping.setdefault(prop_name.name, []).append(prop)
            self._concept_props = mapping
        return self._concept_props

    # ------------------------------------------------------------------ #
    # Structure
    # ------------------------------------------------------------------ #

    def _apply_refinement(self, beliefs, factors):
        """R4B: violation-gradient message passing over the batched beliefs."""
        if self._refiner is None:
            # Only reachable when the graph declared no factor-bearing
            # constraint; fall back to widths observed at runtime.
            widths = {name: entry['matrix'].shape[1] for name, entry in beliefs.items()}
            self._refiner = ConstraintRefinement(widths, **self._refine_config)
        # ``self.device`` may still be the unresolved sentinel 'auto' at this
        # point, so place the refiner on the beliefs' own device instead.
        reference = next(iter(beliefs.values()))['matrix']
        self._refiner = self._refiner.to(reference.device)

        # ConstraintRefinement softmaxes its input, so hand it log-probabilities
        # and the identity softmax(log p) = p keeps the beliefs unchanged when
        # no message fires.
        logits = {name: torch.log(entry['matrix'].clamp_min(1e-30))
                  for name, entry in beliefs.items()}
        refined_logits = self._refiner(logits, factors)
        return {name: torch.softmax(t, dim=-1) for name, t in refined_logits.items()}

    def _apply_factor_graph(self, beliefs, factors):
        """R3: replace beliefs with exact constrained marginals, per grounding.

        One circuit per *formula shape*, not per row. Every row of a factor has
        the same structure and differs only in its weights, so variables are
        named by slot (``v0``, ``v1``, ...) rather than by ``(concept, node)``:
        the manager then hash-conses the same nodes for every row and the
        compiled structure is reused — across rows and across training steps.
        Naming variables per node instead rebuilt the whole circuit every row.

        A row whose circuit exceeds the budget keeps its unrefined beliefs and
        is reported, never silently approximated.
        """
        current = {name: entry['matrix'] for name, entry in beliefs.items()}
        # Groundings that share a node contribute several refined beliefs for it,
        # so results are accumulated and averaged. That is the standard
        # factor-graph aggregation, and it also removes an order-dependence the
        # old row-at-a-time update had: whichever grounding happened to be
        # enumerated last used to win.
        accum = {name: torch.zeros_like(m) for name, m in current.items()}
        counts = {name: torch.zeros(m.shape[0], 1, dtype=m.dtype, device=m.device)
                  for name, m in current.items()}

        for factor in factors:
            if any(lit.concept not in beliefs for lit in factor.literals):
                self.report.fallback.append(f'{factor.name} (missing beliefs)')
                continue
            n_rows = factor.literals[0].node_index.shape[0]
            exact_rows = 0

            for shape_key, batch in self._row_batches(factor, beliefs).items():
                sizes, slots, slot_concept, slot_nodes = batch
                head = self._head_for(factor, sizes, slots, shape_key)
                block = {slot: current[concept][slot_nodes[slot]]
                         for slot, concept in slot_concept.items()}
                out = head(block)
                if head.report.fallback:
                    continue
                rows_here = next(iter(slot_nodes.values())).shape[0]
                exact_rows += rows_here
                ones = torch.ones(rows_here, 1, dtype=counts[
                    next(iter(slot_concept.values()))].dtype)
                for slot, concept in slot_concept.items():
                    index = slot_nodes[slot]
                    accum[concept] = accum[concept].index_add(0, index, out[slot])
                    counts[concept] = counts[concept].index_add(
                        0, index, ones.to(counts[concept].device))

            if exact_rows == n_rows and n_rows:
                self.report.exact.append(factor.name or 'phi')
                self.enforced_constraints.add(factor.name)
            elif exact_rows:
                self.report.fallback.append(
                    f'{factor.name} ({n_rows - exact_rows}/{n_rows} rows fell back)')
            else:
                self.report.fallback.append(f'{factor.name} (no row compiled)')

        refined = {}
        for name, matrix in current.items():
            count = counts[name]
            averaged = accum[name] / count.clamp_min(1.0)
            refined[name] = torch.where(count > 0, averaged, matrix)
        return refined

    def _row_batches(self, factor, beliefs):
        """Group a factor's groundings by formula shape.

        Every row with the same shape is scored through **one** circuit
        evaluation with ``[R, K]`` weights instead of R scalar passes, which is
        what makes exact per-grounding inference affordable at graph scale.
        Returns ``{shape_key: (sizes, slots, {slot: concept}, {slot: node_index})}``.
        """
        batches = {}
        n_rows = factor.literals[0].node_index.shape[0]
        for r in range(n_rows):
            spec = self._row_slots(factor, beliefs, r)
            if spec is None:
                continue
            sizes, slots, group_src, shape_key = spec
            entry = batches.get(shape_key)
            if entry is None:
                entry = (sizes, slots,
                         {slot: concept for slot, (concept, _n) in group_src.items()},
                         {slot: [] for slot in group_src})
                batches[shape_key] = entry
            for slot, (_concept, node) in group_src.items():
                entry[3][slot].append(node)

        return {
            key: (sizes, slots, slot_concept,
                  {slot: torch.tensor(nodes, dtype=torch.long)
                   for slot, nodes in slot_nodes.items()})
            for key, (sizes, slots, slot_concept, slot_nodes) in batches.items()
        }

    def _head_for(self, factor, sizes, slots, shape_key):
        """A head (and its processor) per formula shape, cached on the model.

        Caching across forward passes means a constraint's circuit is compiled
        once for the whole run rather than once per row per step.
        """
        head = self._fg_cache.get(shape_key)
        if head is None:
            from domiknows.solver.circuitBooleanMethods import circuitBooleanMethods
            processor = circuitBooleanMethods(
                backend=self.circuit_backend, max_nodes=self.circuit_max_nodes,
                size_limit_action='raise')
            head = FactorGraphHead(
                [VariableGroup(g, size) for g, size in sizes.items()],
                _row_builder(factor, slots),
                backend=self.circuit_backend, max_nodes=self.circuit_max_nodes,
                name=factor.name or 'phi', processor=processor)
            self._fg_cache[shape_key] = head
        return head

    @staticmethod
    def _row_slots(factor, beliefs, r):
        """Slot-named variable groups for one grounding row.

        Deduped by ``(concept, node)`` — two literals hitting the same node are
        the *same* variable — but named ``v0``, ``v1``, ... so the formula shape
        is row-independent. ``shape_key`` captures everything the compiled
        circuit depends on: the factor, the dedup pattern and the group widths.
        """
        sizes, group_src, slots, index_of = {}, {}, [], {}
        for lit in factor.literals:
            node = int(lit.node_index[r])
            matrix = beliefs[lit.concept]['matrix']
            if node >= matrix.shape[0]:
                return None
            source = (lit.concept, node)
            if source not in index_of:
                name = f'v{len(index_of)}'
                index_of[source] = name
                sizes[name] = matrix.shape[1]
                group_src[name] = source
            slots.append((index_of[source], lit.class_index))
        shape_key = (factor.name, factor.kind, tuple(slots),
                     tuple(sorted(sizes.items())))
        return sizes, slots, group_src, shape_key

    # ------------------------------------------------------------------ #
    # R6 — one semantics end-to-end: MAP decoding replaces ILP
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def inferMAPResults(self, datanode):
        """Decode with max-product MAP, writing ``<concept>/MAP`` one-hots.

        Constraint-respecting *by construction* for everything that compiles, so
        it replaces ILP there; ILP stays available as the cross-check.

        **Never decodes by argmax of the marginals.** Those marginals are exact,
        but maximum-posterior-marginals is not MAP: a per-variable argmax is a
        factorised readout of a distribution that is not factorised, and can
        select an assignment of zero posterior probability (measured 73/3000 on
        an ``exactly-one`` constraint). Only ``map_assignment`` is sound.
        """
        datanode.inferLocal(keys=('softmax',))
        factors, skipped = build_constraint_factors(datanode, self.graph)
        concept_names = {lit.concept for factor in factors for lit in factor.literals}
        beliefs = self._harvest(datanode, concept_names)
        if not beliefs:
            return datanode

        rows_of = {name: list(entry['matrix'].unbind(0))
                   for name, entry in beliefs.items()}
        for factor in factors:
            if any(lit.concept not in beliefs for lit in factor.literals):
                continue
            n_rows = factor.literals[0].node_index.shape[0]
            for r in range(n_rows):
                spec = self._row_slots(factor, beliefs, r)
                if spec is None:
                    continue
                sizes, slots, group_src, shape_key = spec
                head = self._head_for(factor, sizes, slots, shape_key)
                row_beliefs = {g: rows_of[c][n] for g, (c, n) in group_src.items()}
                try:
                    assignment = head.map_predict(row_beliefs)
                except CircuitSizeLimitExceeded:
                    continue
                for g, (c, n) in group_src.items():
                    one_hot = torch.zeros_like(rows_of[c][n])
                    one_hot[assignment[g]] = 1.0
                    rows_of[c][n] = one_hot

        for name, entry in beliefs.items():
            key = f'<{name}>/MAP'
            for row, dn in enumerate(entry['dns']):
                if row < len(rows_of[name]):
                    dn.attributes[key] = rows_of[name][row]
        return datanode

    # ------------------------------------------------------------------ #
    # populate = sensors -> structure -> loss
    # ------------------------------------------------------------------ #

    def populate(self, builder, run=True):
        from domiknows.program.model.base import Mode

        datanode = self.inference(builder)

        # Count *training* steps only. Counting every forward would make the
        # effective warmup depend on the evaluation schedule — a run that
        # validates often would leave warmup early having barely trained.
        if self.mode() == Mode.TRAIN:
            self._train_steps += 1

        use_structure = (
            (self.refine or self.factor_graph)
            and self._train_steps >= self.structure_warmup
        )
        if use_structure:
            try:
                self._apply_structure(builder, datanode)
            except CircuitSizeLimitExceeded as exc:
                # The one *expected* failure: a grounding too big to compile.
                # Everything else is a bug and must surface rather than silently
                # degrade the model to "no structure" for the rest of training.
                self.logger.warning(
                    'StructuredModel: structure skipped this step (%s)', exc)
                self.report.fallback.append('circuit-size-limit')

        lose, metric = super(SolverModel, self).populate(
            builder, datanode=datanode, run=False)
        return datanode, lose, metric

    def _apply_structure(self, builder, datanode):
        self.report = FactorGraphReport()
        self.enforced_constraints = set()

        datanode.inferLocal(keys=('softmax',))
        factors, skipped = build_constraint_factors(datanode, self.graph)
        self.report.fallback.extend(skipped)
        if not factors:
            return

        concept_names = {lit.concept for factor in factors for lit in factor.literals}
        beliefs = self._harvest(datanode, concept_names)
        if not beliefs:
            return

        usable = [f for f in factors
                  if all(lit.concept in beliefs for lit in f.literals)]
        if not usable:
            return

        refined = None
        if self.refine:
            refined = self._apply_refinement(beliefs, usable)
        if self.factor_graph:
            source = beliefs
            if refined is not None:
                source = {name: {**entry, 'matrix': refined[name]}
                          for name, entry in beliefs.items()}
            refined = self._apply_factor_graph(source, usable)

        if refined is None:
            return
        for name, entry in beliefs.items():
            if name in refined:
                self._write_back(builder, datanode, name, entry, refined[name])


def _row_builder(factor, slots):
    """A ``build_constraint`` callable reproducing ``factor``'s semantics.

    Mirrors the violation kinds :mod:`refinement` defines, but as *logic* rather
    than a penalty — this is the R2/R3 distinction: the constraint becomes
    structure the distribution conditions on, not a term scored against it.
    """

    def build(processor, leaves):
        lits = [leaves[group][class_index] for group, class_index in slots]
        if factor.kind == 'implication':
            consequent = (processor.andVar(None, *lits[1:])
                          if len(lits) > 2 else lits[1])
            return processor.ifVar(None, lits[0], consequent)
        if factor.kind == 'exclusion':
            return processor.andVar(
                None, *[processor.nandVar(None, lits[i], lits[j])
                        for i in range(len(lits)) for j in range(i + 1, len(lits))])
        if factor.kind == 'at_least_one':
            return processor.orVar(None, *lits)
        if factor.kind == 'exactly_one':
            return processor.xorVar(None, *lits)
        raise ValueError(f'unsupported factor kind {factor.kind!r}')

    return build
