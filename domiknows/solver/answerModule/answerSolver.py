import logging
from collections import OrderedDict
from itertools import product
from time import perf_counter

from gurobipy import GRB, Model, Env
import torch

from domiknows.graph.logicalConstrain import (
    LogicalConstrain, queryL, miotaL, existsL, sumL, greaterL, atLeastL, exactL,
    notL, andL,
)
from domiknows.utils import setup_logger


# Map from constraint type name (as used in "execute(...)") to class
_TYPE_MAP = {
    'queryL': queryL,
    'miotaL': miotaL,
    'existsL': existsL,
    'sumL': sumL,
    'greaterL': greaterL,
    'atLeastL': atLeastL,
    'exactL': exactL,
}

logger = setup_logger({
    'log_name': 'answerSolver',
    'log_level': logging.INFO,
    'log_filename': 'answerSolver.log',
    'log_filesize': 50*1024*1024,  # 50MB
    'log_backupCount': 5,
    'log_fileMode': 'a',
})


class AnswerSolver:
    def __init__(self, graph, solver=None):
        if solver is not None:
            self.solver = solver
        else:
            from domiknows.solver import ilpOntSolverFactory
            self.solver = ilpOntSolverFactory.getOntSolverInstance({graph})

    # ── ILP cache management ────────────────────────────────────────────

    @staticmethod
    def _clear_ilp_cache(dn, visited=None):
        """Remove cached ILP variables from the DataNode tree.

        ``createILPVariable`` (gurobiILPOntSolver) stores Gurobi variables
        on each DataNode under keys like ``<concept>/ILP/x``.  When we
        need a *different* Gurobi model (e.g. a fresh build per
        hypothesis), the stale cache must be cleared so that variables
        are created on the new model.
        """
        if visited is None:
            visited = set()
        if id(dn) in visited:
            return
        visited.add(id(dn))

        ilp_keys = [k for k in dn.attributes if '/ILP/' in k]
        for k in ilp_keys:
            del dn.attributes[k]

        # Recurse into explicit children
        if hasattr(dn, 'childDataNodes'):
            for children in dn.childDataNodes.values():
                for child in children:
                    AnswerSolver._clear_ilp_cache(child, visited)

        # Recurse through graph links as well. The solver uses findDatanodes(),
        # which traverses links rather than only childDataNodes, so stale ILP
        # vars can survive on linked relation/object nodes unless we clear them.
        if hasattr(dn, 'getLinks'):
            for linked_dns in dn.getLinks().values():
                for linked_dn in linked_dns:
                    AnswerSolver._clear_ilp_cache(linked_dn, visited)

    @staticmethod
    def _safe_var_name(var):
        try:
            return var.VarName
        except Exception:
            return f"<{type(var).__name__}:name-unavailable>"

    def _log_var_not_in_model_details(self, error, dn, model, stage):
        """Log candidate variable names and return a short diagnostic summary."""
        if "Variable not in model" not in str(error):
            return None

        try:
            model_vars = model.getVars()
            names = []
            for mv in model_vars[:10]:
                try:
                    names.append(mv.VarName)
                except Exception:
                    continue

            logger.error(
                "Variable not in model during %s. model_var_count=%d, sample_model_vars=%s",
                stage,
                len(model_vars),
                names,
            )
            sample = ",".join(names)
            return f"stage={stage}; model_var_count={len(model_vars)}; sample_model_vars={sample}"
        except Exception as debug_error:
            logger.error(
                "Variable-not-in-model debug logging failed during %s: %s",
                stage,
                debug_error,
            )
            return f"stage={stage}; debug_logging_failed={debug_error}"

    # ── public API ──────────────────────────────────────────────────────

    @staticmethod
    def _format_executable_constraint(elc, graph):
        """Return a readable identifier/expression tuple for executable constraints."""
        elc_id = getattr(elc, 'lcName', '<unknown-elc>')
        elc_label = getattr(elc, 'name', elc_id)

        # Find registry key (usually same as lcName) for extra traceability.
        registry_key = elc_id
        try:
            for key, value in graph.executableLCs.items():
                if value is elc:
                    registry_key = key
                    break
        except Exception:
            pass

        try:
            expression = elc.strEs()
        except Exception:
            expression = '<expression-unavailable>'

        return registry_key, elc_id, elc_label, expression

    def answer(self, question, dn):
        """Answer an executable constraint question.

        Args:
            question: String in "execute(<type>)" format, e.g. "execute(queryL)".
            dn: Root DataNode containing predictions.

        Returns:
            The answer: str for single-answer queryL, an aligned class-ID list
            for multi-answer queryL, int for sumL, bool for boolean types, a
            multi-hot list for miotaL, or None if no feasible result exists.
        """
        answer_started = perf_counter()

        if not question.startswith('execute(') or not question.endswith(')'):
            raise ValueError(f"Invalid question format: {question}")

        constraint_name = question[len('execute('):-1].strip()

        try:
            elc = self._resolve_constraint(constraint_name, dn.graph)
            lc = elc.innerLC
            reg_key, elc_id, elc_label, elc_expr = self._format_executable_constraint(elc, dn.graph)

            result = self.solve_active_constraints(
                dn,
                [reg_key],
                populate=False,
                raise_on_infeasible=False,
            )
            answer_value = (
                None
                if result is None
                else result['hypotheses'][reg_key]
            )

            elapsed_ms = (perf_counter() - answer_started) * 1000.0
            logger.info(
                "AnswerSolver answered in %.2f ms | question=%s | constraint_lookup=%s | elc_id=%s | elc_name=%s | lc_type=%s | constraint=%s | answer=%r",
                elapsed_ms,
                question,
                reg_key,
                elc_id,
                elc_label,
                type(lc).__name__,
                elc_expr,
                answer_value,
            )
            return answer_value
        except Exception as e:
            elapsed_ms = (perf_counter() - answer_started) * 1000.0
            logger.error(
                "AnswerSolver failed after %.2f ms | question=%s | parsed_constraint=%s | error=%s",
                elapsed_ms,
                question,
                constraint_name,
                e,
            )
            if "Variable not in model" not in str(e):
                raise

            details = f"constraint={constraint_name}"

            raise RuntimeError(f"{e}; {details}") from e
        finally:
            self._clear_ilp_cache(dn)

    # ── constraint resolution ───────────────────────────────────────────

    def _resolve_constraint(self, constraint_name, graph):
        """Find the execute wrapper in graph.executableLCs.

        Supports two lookup modes:
            1. By ELC name (e.g. 'ELC0') — direct lookup in graph.executableLCs.
            2. By type name (e.g. 'queryL') — finds the first matching ELC whose
               innerLC is an instance of the named type class.
        """
        # 1. Try direct ELC name lookup
        if constraint_name in graph.executableLCs:
            return graph.executableLCs[constraint_name]

        # 2. Try type name lookup
        cls = _TYPE_MAP.get(constraint_name)
        if cls is None:
            raise ValueError(
                f"Unknown constraint name or type: '{constraint_name}'. "
                f"Not found as an ELC name and not a known type ({', '.join(_TYPE_MAP)})"
            )

        for _, elc in graph.executableLCs.items():
            if isinstance(elc.innerLC, cls) and elc.active:
                return elc

        # If no active one found, try any matching one
        for _, elc in graph.executableLCs.items():
            if isinstance(elc.innerLC, cls):
                return elc

        raise ValueError(
            f"No executable constraint of type '{constraint_name}' found in graph.executableLCs"
        )

    def _sum_hypothesis_upper_bound(self, lc, dn, concepts_relations):
        """Return a safe entity-count bound for a ``sumL`` hypothesis."""
        concept_names = lc.getLcConcepts()
        candidate_counts = []

        for concept_relation in concepts_relations:
            concept_name = self.solver.getConceptName(concept_relation)
            if concept_name not in concept_names:
                continue

            root_concept = dn.findRootConceptOrRelation(concept_relation[0])
            candidate_counts.append(
                len(dn.findDatanodes(select=root_concept))
            )

        if not candidate_counts:
            logger.warning(
                "No ILP candidates found for sumL concepts: %s",
                concept_names,
            )
            return 0

        # Sum the relevant positive-variable domains. This can overestimate
        # conjunction counts, but it remains a safe upper bound for relation
        # groundings and constraints spanning more than one entity domain.
        return sum(candidate_counts)

    def _hypothesis_spec(self, elc, dn, concepts_relations):
        """Return ordered hypothesis values and an LC compiler for one ELC."""
        lc = elc.innerLC
        graph = lc.graph

        if isinstance(lc, queryL):
            subclass_names = list(lc._subclass_names)
            subclasses = list(lc._subclasses)
            iota_elements = list(lc.e)

            def build_query(subclass_name):
                idx = subclass_names.index(subclass_name)
                concept, name, subclass_index = subclasses[idx]

                from domiknows.graph.concept import EnumConcept
                if isinstance(lc.concept, EnumConcept):
                    subclass_tuple = (
                        lc.concept,
                        name,
                        subclass_index,
                        len(subclass_names),
                    )
                else:
                    subclass_tuple = (concept, concept.name, None, 1)

                return self._compile_hypothesis(
                    andL,
                    [subclass_tuple] + iota_elements,
                    graph,
                )

            return subclass_names, build_query

        if isinstance(lc, sumL):
            max_count = self._sum_hypothesis_upper_bound(
                lc,
                dn,
                concepts_relations,
            )
            base_elements = [e for e in lc.e if not isinstance(e, int)]

            def build_sum(count):
                return self._compile_hypothesis(
                    exactL,
                    list(base_elements) + [count],
                    graph,
                )

            return list(range(0, max_count + 1)), build_sum

        if isinstance(lc, (existsL, greaterL, atLeastL, exactL)):
            lc_class = type(lc)
            constructor_kwargs = {}
            explicit_limit = getattr(lc, '_explicitLimit', None)
            if explicit_limit is not None:
                constructor_kwargs['limit'] = explicit_limit

            def build_boolean(hypothesis):
                inner = self._compile_hypothesis(
                    lc_class,
                    lc.e,
                    graph,
                    **constructor_kwargs,
                )
                if hypothesis:
                    return inner
                return self._compile_hypothesis(notL, [inner], graph)

            return [True, False], build_boolean

        raise ValueError(
            "Unsupported executable constraint type "
            f"'{type(lc).__name__}' for {getattr(elc, 'lcName', elc)}"
        )

    def _decode_miota(self, lc, dn, key=("local", "softmax")):
        """Decode a miotaL directly; it has no powerset hypotheses to search."""
        key_text = "/" + "/".join(key) if isinstance(key, (tuple, list)) else key
        processor = self.solver.myLcLossBooleanMethods
        processor.current_device = dn.current_device
        self.solver.constraintConstructor.current_device = dn.current_device
        self.solver.constraintConstructor.myGraph = self.solver.myGraph
        output, _ = self.solver.constraintConstructor.constructLogicalConstrains(
            lc, processor, None, dn, 0, key=key_text,
            headLC=False, loss=True, sample=False,
        )
        tensors = []

        def collect(value):
            if torch.is_tensor(value):
                tensors.append(value.reshape(-1))
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        collect(output)
        if not tensors:
            return []
        probabilities = torch.cat(tensors)
        return (probabilities >= lc.threshold).to(torch.int64).detach().cpu().tolist()

    def _decode_multi_query(self, lc, dn, key=("local", "softmax")):
        """Decode every candidate row without enumerating class products."""
        key_text = "/" + "/".join(key) if isinstance(key, (tuple, list)) else key
        processor = self.solver.myLcLossBooleanMethods
        processor.current_device = dn.current_device
        self.solver.constraintConstructor.current_device = dn.current_device
        self.solver.constraintConstructor.myGraph = self.solver.myGraph
        output, _ = self.solver.constraintConstructor.constructLogicalConstrains(
            lc, processor, None, dn, 0, key=key_text,
            headLC=False, loss=True, sample=False,
        )
        matrices = []

        def collect(value):
            if torch.is_tensor(value) and value.dim() >= 2:
                if value.shape[-1] == lc.num_subclasses:
                    matrices.append(value.reshape(-1, lc.num_subclasses))
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        collect(output)
        if not matrices:
            return []
        distribution = matrices[0] if len(matrices) == 1 else torch.cat(matrices)
        membership = distribution.sum(dim=-1)
        answers = distribution.argmax(dim=-1).long()
        answers = torch.where(
            membership >= lc.threshold,
            answers,
            torch.full_like(answers, -1),
        )
        return answers.detach().cpu().tolist()

    def _decode_multi_query_ilp(self, lc, dn):
        """Decode candidate/class conjunctions from a populated ILP solution."""
        processor = self.solver.booleanMethodsCalculator
        processor.current_device = dn.current_device
        self.solver.constraintConstructor.current_device = dn.current_device
        self.solver.constraintConstructor.myGraph = self.solver.myGraph
        output, _ = self.solver.constraintConstructor.constructLogicalConstrains(
            lc, processor, None, dn, 0, key="/ILP",
            headLC=False, loss=False, sample=False, verify=True,
        )
        rows = []

        def collect(value):
            if isinstance(value, (list, tuple)):
                if (
                    len(value) == lc.num_subclasses
                    and all(not isinstance(item, (list, tuple)) for item in value)
                ):
                    rows.append([
                        float(item.item()) if hasattr(item, 'item') else float(item)
                        for item in value
                    ])
                else:
                    for item in value:
                        collect(item)

        collect(output)
        if not rows:
            return None
        return [
            max(range(lc.num_subclasses), key=row.__getitem__)
            if any(value > 0.5 for value in row) else -1
            for row in rows
        ]

    @staticmethod
    def _snapshot_ilp_attributes(dn):
        """Capture ILP attributes so populate=False remains non-mutating."""
        snapshot = []
        visited = set()

        def visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            snapshot.append((
                node,
                {key: value for key, value in node.attributes.items() if '/ILP' in key},
            ))
            for child in node.getChildDataNodes() or []:
                visit(child)
            for linked in getattr(node, 'relationLinks', {}).values():
                for relation_node in linked:
                    visit(relation_node)

        visit(dn)
        return snapshot

    @staticmethod
    def _restore_ilp_attributes(snapshot):
        for node, saved in snapshot:
            for key in [key for key in node.attributes if '/ILP' in key]:
                del node.attributes[key]
            node.attributes.update(saved)

    @staticmethod
    def _is_infeasible_error(error):
        return "infeasible" in str(error).lower()

    def solve_active_constraints(
        self,
        dn,
        active_constraint_names,
        concepts_relations=None,
        *,
        key=("local", "softmax"),
        fun=None,
        epsilon=0.00001,
        minimize_objective=False,
        ignore_pin_lcs=False,
        populate=True,
        raise_on_infeasible=True,
    ):
        """Solve every joint hypothesis for the active executable constraints.

        Returns a dictionary containing the selected hypotheses, objective, and
        detached variable assignment.  Only the winning assignment is written
        to ``dn`` when ``populate`` is true. Winning hypothesis answers are
        also written to the sample's constraint DataNode as
        ``<constraint-name>/answer`` attributes.
        """
        answer_target_available = False
        if populate:
            # Clear answers before validation/model construction so failures
            # cannot leave a previous inference's hypothesis visible.
            answer_target_available = (
                dn._clearExecutableConstraintAnswers()
            )
            if not answer_target_available:
                logger.warning(
                    "No constraint DataNode found; executable hypothesis "
                    "answers will not be persisted"
                )

        requested_names = set(active_constraint_names)
        unknown_names = requested_names.difference(dn.graph.executableLCs)
        if unknown_names:
            unknown_text = ", ".join(sorted(unknown_names))
            raise ValueError(
                f"Unknown active executable constraint name(s): {unknown_text}"
            )

        ordered_names = [
            name
            for name in dn.graph.executableLCs
            if name in requested_names
        ]
        if not ordered_names:
            return None

        required_solver_api = (
            '_calculateILPSelection',
            'populateILPSelection',
        )
        missing_api = [
            api for api in required_solver_api if not hasattr(self.solver, api)
        ]
        if missing_api:
            raise TypeError(
                "Hypothesis-aware executable inference requires a Gurobi "
                f"ILP solver supporting {', '.join(missing_api)}"
            )

        if concepts_relations is None:
            concepts_relations = dn.collectConceptsAndRelations()
        concepts_relations = tuple(concepts_relations)

        specs = []
        direct_decode_names = []
        for name in ordered_names:
            elc = dn.graph.executableLCs[name]
            if (
                isinstance(elc.innerLC, miotaL)
                or (
                    isinstance(elc.innerLC, queryL)
                    and elc.innerLC.is_multi_answer
                )
            ):
                direct_decode_names.append(name)
                continue
            values, builder = self._hypothesis_spec(
                elc,
                dn,
                concepts_relations,
            )
            specs.append((name, values, builder))

        best_result = None
        best_hypotheses = None

        try:
            for hypothesis_values in product(
                *(values for _, values, _ in specs)
            ):
                self._clear_ilp_cache(dn)
                hypothesis_lcs = [
                    builder(value)
                    for (_, _, builder), value in zip(
                        specs,
                        hypothesis_values,
                    )
                ]

                try:
                    candidate = self.solver._calculateILPSelection(
                        dn,
                        *concepts_relations,
                        key=key,
                        fun=fun,
                        epsilon=epsilon,
                        minimizeObjective=minimize_objective,
                        ignorePinLCs=ignore_pin_lcs,
                        extraLogicalConstraints=hypothesis_lcs,
                        populate=False,
                        forceFreshModel=True,
                        raiseOnInfeasible=False,
                    )
                except Exception as error:
                    if self._is_infeasible_error(error):
                        logger.debug(
                            "Joint hypothesis %r is structurally infeasible: %s",
                            hypothesis_values,
                            error,
                        )
                        continue
                    raise

                if candidate is None:
                    continue

                if best_result is None:
                    is_better = True
                elif minimize_objective:
                    is_better = (
                        candidate['objective'] < best_result['objective']
                    )
                else:
                    is_better = (
                        candidate['objective'] > best_result['objective']
                    )

                # Strict comparison intentionally preserves the first
                # graph/hypothesis-order candidate on exact objective ties.
                if is_better:
                    best_result = {
                        'objective': candidate['objective'],
                        'values': dict(candidate['values']),
                    }
                    best_hypotheses = OrderedDict(
                        (name, value)
                        for (name, _values, _builder), value in zip(
                            specs, hypothesis_values
                        )
                    )
        finally:
            self._clear_ilp_cache(dn)

        if best_result is None:
            message = (
                "All joint hypotheses were infeasible for active executable "
                f"constraints: {', '.join(ordered_names)}"
            )
            logger.warning(message)
            if raise_on_infeasible:
                raise RuntimeError(message)
            return None

        populated_winner = False
        temporary_ilp_snapshot = None
        multi_query_names = [
            name for name in direct_decode_names
            if isinstance(dn.graph.executableLCs[name].innerLC, queryL)
        ]
        if multi_query_names:
            if not populate:
                temporary_ilp_snapshot = self._snapshot_ilp_attributes(dn)
            self.solver.populateILPSelection(
                dn,
                concepts_relations,
                best_result['values'],
            )
            populated_winner = populate

        if direct_decode_names:
            decoded_direct = {}
            try:
                for name in direct_decode_names:
                    inner = dn.graph.executableLCs[name].innerLC
                    if isinstance(inner, miotaL):
                        decoded_direct[name] = self._decode_miota(inner, dn, key=key)
                    else:
                        decoded = self._decode_multi_query_ilp(inner, dn)
                        decoded_direct[name] = (
                            decoded if decoded is not None
                            else self._decode_multi_query(inner, dn, key=key)
                        )
            finally:
                if temporary_ilp_snapshot is not None:
                    self._restore_ilp_attributes(temporary_ilp_snapshot)
            best_hypotheses = OrderedDict(
                (name, decoded_direct[name] if name in decoded_direct else best_hypotheses[name])
                for name in ordered_names
            )

        if populate and not populated_winner:
            self.solver.populateILPSelection(
                dn,
                concepts_relations,
                best_result['values'],
            )
        if populate and answer_target_available:
            dn._writeExecutableConstraintAnswers(best_hypotheses)

        result = {
            'hypotheses': best_hypotheses,
            'objective': best_result['objective'],
            'values': best_result['values'],
        }
        logger.info(
            "Selected executable hypotheses %s with objective %.6f",
            dict(best_hypotheses),
            best_result['objective'],
        )
        return result

    # ── hypothesis compilation ──────────────────────────────────────────

    def _compile_hypothesis(self, lc_class, elements, graph, **kwargs):
        """Create a hypothesis LogicalConstrain object without registering it in the graph.

        Constructing any LogicalConstrain automatically registers it in
        graph.logicalConstrains and sets headLC=False on nested LCs.
        This method bypasses registration by temporarily swapping
        graph.logicalConstrains with a disposable dict during construction,
        and restores any mutated headLC flags afterwards.

        Args:
            lc_class: The LogicalConstrain subclass to instantiate (e.g. notL, exactL).
            elements: The elements (*e) to pass to the constructor.
            graph: The Graph instance (needed for context during LC construction).
            **kwargs: Additional keyword arguments for the constructor (e.g. p, active).

        Returns:
            The newly created LogicalConstrain instance, detached from the graph.
        """
        # Save headLC state of any nested LCs before construction mutates them
        saved_head = {id(e): e.headLC for e in elements if isinstance(e, LogicalConstrain)}

        # Swap the backing store so auto-registration goes into a temporary
        # dict — graph.logicalConstrains is a read-only property.
        original_lcs = graph._logicalConstrains
        graph._logicalConstrains = OrderedDict()

        # Push graph onto the shared context stack so LcElement.__init__
        # can resolve the graph from element._context (which is empty
        # outside a ``with graph:`` block).
        graph._context.append(graph)
        try:
            hyp_lc = lc_class(*elements, **kwargs)
        finally:
            graph._context.pop()
            graph._logicalConstrains = original_lcs

        # Restore headLC on any nested LCs that were mutated by __init__
        for e in elements:
            if isinstance(e, LogicalConstrain) and id(e) in saved_head:
                e.headLC = saved_head[id(e)]

        # Mark as head so addLogicalConstrains will process it
        hyp_lc.headLC = True
        hyp_lc.active = True

        return hyp_lc

    # ── base ILP model construction ─────────────────────────────────────

    def _build_base_model(self, dn):
        """Build a complete ILP model (variables + constraints + objective) following
        the pattern from gurobiILPOntSolver.calculateILPSelection.

        Returns:
            (m, x, conceptsRelations) where m is the Gurobi Model, x is the
            variable dict, and conceptsRelations is the tuple list.
        """
        self.solver.current_device = dn.current_device

        conceptsRelations = dn.collectConceptsAndRelations()

        gurobiEnv = Env("", empty=True)
        gurobiEnv.setParam('OutputFlag', 0)
        gurobiEnv.start()

        m = Model("answerSolver", gurobiEnv)
        m.params.outputflag = 0
        x = OrderedDict()

        # Create ILP variables and objective
        Q = self.solver.createILPVariables(m, x, dn, *conceptsRelations)

        # Add structural constraints
        self.solver.addOntologyConstrains(m, dn, *conceptsRelations)
        self.solver.addGraphConstrains(m, dn, *conceptsRelations)
        self.solver.addMulticlassExclusivity(conceptsRelations, dn, m)

        # Collect and add all active *structural* logical constraints.
        lcs = []
        for graph in self.solver.myGraph:
            for _, lc_item in graph.logicalConstrains.items():
                if lc_item.headLC and lc_item.active:
                    lcs.append(lc_item)

        if lcs:
            try:
                self.solver.addLogicalConstrains(m, dn, lcs, 100, key="/ILP/x")
            except Exception as e:
                details = self._log_var_not_in_model_details(e, dn, m, "base-logical-constraints")
                if details:
                    raise RuntimeError(f"{e}; {details}") from e
                raise

        # Set objective
        if Q is None:
            Q = 0
            logger.warning("No ILP variables created — empty model")

        m.setObjective(Q, GRB.MAXIMIZE)
        m.update()

        return m, x, conceptsRelations

    # ── generic hypothesis loop ─────────────────────────────────────────

    def _solve_with_hypotheses(self, dn, hypotheses, build_hypothesis_lc_fn):
        """Test each hypothesis by building a fresh ILP model, adding the
        hypothesis constraint, solving, and selecting the feasible result
        with the best objective

        Args:
            dn: Root DataNode.
            hypotheses: Iterable of hypothesis values to test.
            build_hypothesis_lc_fn: callable(hypothesis) -> LogicalConstrain
                Returns a compiled hypothesis LC for the given hypothesis value.

        Returns:
            (best_hypothesis, best_obj) or (None, None) if all infeasible.
        """
        best_hypothesis = None
        best_obj = None

        for hyp in hypotheses:
            # Clear ILP cache so _build_base_model creates fresh variables
            self._clear_ilp_cache(dn)

            # Build a fresh model (variables are cached on dn for THIS model)
            m, x, _ = self._build_base_model(dn)

            # Build hypothesis as a proper LogicalConstrain
            hyp_lc = build_hypothesis_lc_fn(hyp)

            # Add hypothesis constraint to the fresh model
            try:
                self.solver.addLogicalConstrains(m, dn, [hyp_lc], 100, key="/ILP/x")
            except Exception as e:
                if "ILP model is infeasible" in str(e):
                    # The hypothesis is structurally infeasible (e.g. NOT(1)
                    # when the inner expression resolved to a fixed True).
                    # Treat the same as an infeasible solve — skip this
                    # hypothesis.
                    logger.debug(
                        "Hypothesis %r is structurally infeasible: %s", hyp, e
                    )
                    continue
                details = self._log_var_not_in_model_details(e, dn, m, f"hypothesis={hyp}")
                if details:
                    raise RuntimeError(f"{e}; {details}") from e
                raise

            m.update()
            m.optimize()

            if m.status == GRB.Status.OPTIMAL:
                obj = m.ObjVal
                if best_obj is None or obj > best_obj:
                    best_obj = obj
                    best_hypothesis = hyp

        # Clear cache after loop so dn isn't left with stale vars
        self._clear_ilp_cache(dn)

        if best_hypothesis is None:
            logger.warning("All hypotheses were infeasible")

        return best_hypothesis, best_obj

    # ── per-type answer handlers ────────────────────────────────────────

    def _answer_existsL(self, lc, dn, m, x):
        """Answer an existsL constraint: "Does there exist ...?"

        Hypotheses:
            True  -> existsL(sub-elements)   (the original constraint itself)
            False -> notL(existsL(sub-elements))
        """
        graph = lc.graph

        def build_hyp(hyp):
            if hyp:
                # True hypothesis: the existsL constraint itself
                return self._compile_hypothesis(existsL, lc.e, graph)
            else:
                # False hypothesis: negate the existsL
                # First create the inner existsL, then wrap in notL
                inner = self._compile_hypothesis(existsL, lc.e, graph)
                return self._compile_hypothesis(notL, [inner], graph)

        best, _ = self._solve_with_hypotheses(dn, [True, False], build_hyp)
        return best

    def _answer_queryL(self, lc, dn, m, x):
        """Answer a queryL constraint: "What is the <concept> of THE selected entity?"

        Hypotheses are the subclass names of lc.concept.
        For each hypothesis, compile an existsL constraint that requires the
        selected entity (via iotaL sub-expression) to have the specific subclass.
        """
        graph = lc.graph
        subclass_names = lc._subclass_names
        subclasses = lc._subclasses  # list of (concept, name, index) tuples

        # The iotaL sub-expression that selects the entity
        # lc.e contains the elements passed to queryL (excluding the concept)
        iota_elements = lc.e  # e.g. [iotaL(...)]

        def build_hyp(subclass_name):
            idx = subclass_names.index(subclass_name)
            concept, name, i = subclasses[idx]

            # Build: andL(subclass_concept_tuple, iotaL_element)
            # The subclass tuple pins the concept value
            from domiknows.graph.concept import EnumConcept
            if isinstance(lc.concept, EnumConcept):
                subclass_tuple = (lc.concept, name, i, len(subclass_names))
            else:
                subclass_tuple = (concept, concept.name, None, 1)

            # Combine subclass pin with the entity selection from iotaL
            hyp_elements = [subclass_tuple] + list(iota_elements)
            return self._compile_hypothesis(andL, hyp_elements, graph)

        best, _ = self._solve_with_hypotheses(dn, subclass_names, build_hyp)
        return best

    def _answer_sumL(self, lc, dn, m, x):
        """Answer a sumL constraint: "How many ...?"

        Hypotheses: range(0, max_count + 1).
        For each count n, compile an exactL constraint with the same
        sub-elements and limit n.
        """
        graph = lc.graph

        # Determine max_count from datanodes for the relevant concepts
        concept_names = lc.getLcConcepts()
        max_count = 0
        for k, v in x.items():
            if v is not None and k[0].name in concept_names:
                max_count += 1

        if max_count == 0:
            logger.warning("No ILP variables found for sumL concepts: %s", concept_names)
            return 0

        # Elements without any trailing int (which would be a limit in _CountBaseL)
        base_elements = [e for e in lc.e if not isinstance(e, int)]

        def build_hyp(n):
            # Hypothesis: exactL(sub_elements..., n) — exactly n satisfy the condition
            hyp_elements = list(base_elements) + [n]
            return self._compile_hypothesis(exactL, hyp_elements, graph)

        best, _ = self._solve_with_hypotheses(dn, range(0, max_count + 1), build_hyp)
        return best

    def _answer_greaterL(self, lc, dn, m, x):
        """Answer a greaterL constraint: "Are there more X than Y?"

        Hypotheses:
            True  -> greaterL(X, Y)   (the original constraint)
            False -> notL(greaterL(X, Y))
        """
        graph = lc.graph

        def build_hyp(hyp):
            if hyp:
                return self._compile_hypothesis(greaterL, lc.e, graph)
            else:
                inner = self._compile_hypothesis(greaterL, lc.e, graph)
                return self._compile_hypothesis(notL, [inner], graph)

        best, _ = self._solve_with_hypotheses(dn, [True, False], build_hyp)
        return best

    def _answer_atLeastL(self, lc, dn, m, x):
        """Answer an atLeastL constraint: "Are there at least N ...?"

        Hypotheses:
            True  -> atLeastL(sub-elements, N)   (the original constraint)
            False -> notL(atLeastL(sub-elements, N))
        """
        graph = lc.graph

        def build_hyp(hyp):
            if hyp:
                return self._compile_hypothesis(atLeastL, lc.e, graph)
            else:
                inner = self._compile_hypothesis(atLeastL, lc.e, graph)
                return self._compile_hypothesis(notL, [inner], graph)

        best, _ = self._solve_with_hypotheses(dn, [True, False], build_hyp)
        return best

    def _answer_exactL(self, lc, dn, m, x):
        """Answer an exactL constraint: "Are there exactly N ...?"

        Hypotheses:
            True  -> exactL(sub-elements, N)   (the original constraint)
            False -> notL(exactL(sub-elements, N))
        """
        graph = lc.graph

        def build_hyp(hyp):
            if hyp:
                return self._compile_hypothesis(exactL, lc.e, graph)
            else:
                inner = self._compile_hypothesis(exactL, lc.e, graph)
                return self._compile_hypothesis(notL, [inner], graph)

        best, _ = self._solve_with_hypotheses(dn, [True, False], build_hyp)
        return best
