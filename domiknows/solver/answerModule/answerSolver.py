import logging
from collections import OrderedDict
from time import perf_counter

from gurobipy import GRB, Model, Env

from domiknows.graph.logicalConstrain import (
    LogicalConstrain, queryL, existsL, sumL, greaterL, atLeastL, exactL,
    notL, andL,
)


# Map from constraint type name (as used in "execute(...)") to class
_TYPE_MAP = {
    'queryL': queryL,
    'existsL': existsL,
    'sumL': sumL,
    'greaterL': greaterL,
    'atLeastL': atLeastL,
    'exactL': exactL,
}

logger = logging.getLogger(__name__)


class AnswerSolver:
    def __init__(self, graph, solver=None):
        if solver is not None:
            self.solver = solver
        else:
            from domiknows.solver import ilpOntSolverFactory
            self.solver = ilpOntSolverFactory.getOntSolverInstance({graph})

    # ── public API ──────────────────────────────────────────────────────

    def answer(self, question, dn):
        """Answer an executable constraint question.

        Args:
            question: String in "execute(<type>)" format, e.g. "execute(queryL)".
            dn: Root DataNode containing predictions.

        Returns:
            The answer: str for queryL, int for sumL, bool for boolean types,
            or None if no feasible hypothesis exists.
        """
        if not question.startswith('execute(') or not question.endswith(')'):
            raise ValueError(f"Invalid question format: {question}")

        constraint_name = question[len('execute('):-1].strip()

        elc = self._resolve_constraint(constraint_name, dn.graph)
        lc = elc.innerLC

        m, x, conceptsRelations = self._build_base_model(dn)

        # Dispatch to per-type handler
        if isinstance(lc, queryL):
            return self._answer_queryL(lc, dn, m, x)
        elif isinstance(lc, existsL):
            return self._answer_existsL(lc, dn, m, x)
        elif isinstance(lc, sumL):
            return self._answer_sumL(lc, dn, m, x)
        elif isinstance(lc, greaterL):
            return self._answer_greaterL(lc, dn, m, x)
        elif isinstance(lc, atLeastL):
            return self._answer_atLeastL(lc, dn, m, x)
        elif isinstance(lc, exactL):
            return self._answer_exactL(lc, dn, m, x)
        else:
            raise ValueError(f"Unsupported constraint type: {type(lc).__name__}")

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

        # Swap graph.logicalConstrains with a temp dict so auto-registration
        # goes into the void — the real constraint set stays untouched
        original_lcs = graph.logicalConstrains
        graph.logicalConstrains = OrderedDict()
        try:
            hyp_lc = lc_class(*elements, **kwargs)
        finally:
            graph.logicalConstrains = original_lcs

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

        # Collect and add all active logical constraints
        lcs = []
        for graph in self.solver.myGraph:
            for _, lc_item in graph.allLogicalConstrains:
                if lc_item.headLC and lc_item.active:
                    lcs.append(lc_item)

        if lcs:
            self.solver.addLogicalConstrains(m, dn, lcs, 100, key="/ILP/x")

        # Set objective
        if Q is None:
            Q = 0
            logger.warning("No ILP variables created — empty model")

        m.setObjective(Q, GRB.MAXIMIZE)
        m.update()

        return m, x, conceptsRelations

    # ── model copy helper ───────────────────────────────────────────────

    def _copy_model_and_vars(self, m, x):
        """Copy a Gurobi model and remap the variable dictionary."""
        mCopy = m.copy()
        xCopy = OrderedDict()
        for k, v in x.items():
            if v is not None:
                xCopy[k] = mCopy.getVarByName(v.VarName)
            else:
                xCopy[k] = None
        return mCopy, xCopy

    # ── generic hypothesis loop ─────────────────────────────────────────

    def _solve_with_hypotheses(self, m, x, dn, hypotheses, build_hypothesis_lc_fn):
        """Test each hypothesis by copying the model, compiling the hypothesis
        as a LogicalConstrain, adding it via addLogicalConstrains, solving,
        and selecting the feasible result with the best objective.

        Args:
            m: Base Gurobi model (already has variables, constraints, objective).
            x: Variable dictionary {(concept, label, instanceID, labelIndex): GurobiVar}.
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
            mCopy, xCopy = self._copy_model_and_vars(m, x)

            # Build hypothesis as a proper LogicalConstrain
            hyp_lc = build_hypothesis_lc_fn(hyp)

            # Add hypothesis constraint to the copied model via the solver infrastructure
            # This correctly translates nested LCs into ILP constraints
            self.solver.addLogicalConstrains(mCopy, dn, [hyp_lc], 100, key="/ILP/x")

            mCopy.update()
            mCopy.optimize()

            if mCopy.status == GRB.Status.OPTIMAL:
                obj = mCopy.ObjVal
                if best_obj is None or obj > best_obj:
                    best_obj = obj
                    best_hypothesis = hyp

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

        best, _ = self._solve_with_hypotheses(m, x, dn, [True, False], build_hyp)
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

        best, _ = self._solve_with_hypotheses(m, x, dn, subclass_names, build_hyp)
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

        best, _ = self._solve_with_hypotheses(m, x, dn, range(0, max_count + 1), build_hyp)
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

        best, _ = self._solve_with_hypotheses(m, x, dn, [True, False], build_hyp)
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

        best, _ = self._solve_with_hypotheses(m, x, dn, [True, False], build_hyp)
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

        best, _ = self._solve_with_hypotheses(m, x, dn, [True, False], build_hyp)
        return best
