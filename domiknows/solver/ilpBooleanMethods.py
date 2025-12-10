import abc

# Resolve relative / absolute import so the interface can be reused from
# within domiknows *or* as a standalone module during testing.
if __package__ in (None, ""):
    from domiknows.solver.ilpConfig import ilpConfig
else:
    from .ilpConfig import ilpConfig


class ilpBooleanProcessor(object, metaclass=abc.ABCMeta):
    """Abstract base class that specifies *logical* building blocks for
    Integer‑Linear‑Programming encodings.

    Every concrete subclass must implement each Boolean operator as *either*
    a *reified* form (returning a fresh binary that represents the truth
    value of the expression) *or* a *hard* form ("onlyConstrains=True") that
    merely adds the appropriate constraints and returns nothing.

    All operators assume their inputs are **binary literals**:
        • a Gurobi Var of type *BINARY*  (0/1)
        • the Python integers 0 or 1
        • or ``None`` (treated as an unknown literal: 1 for positive context,
          0 for negative, exactly as in the original implementation).
    """

    # ---------------------------------------------------------------------
    # Unary operator
    # ---------------------------------------------------------------------
    @abc.abstractmethod
    def notVar(self, m, _var, *, onlyConstrains: bool = False):
        """Logical **negation**.

        Reified form:   create binary *varNOT* and add
            1 − _var  ==  varNOT             (two‑way equivalence)
        so *varNOT* equals the logical *NOT(_var)*.

        Constraint‑only form: simply force ``_var == 0`` so that NOT(_var)
        would be *True* without introducing *varNOT*.
        """

    # ------------------------------------------------------------------
    # N‑ary conjunction (AND)
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def andVar(self, m, *_var, onlyConstrains: bool = False):
        """General **N‑ary conjunction**.

        Reified form:
            varAND ≤ v_i               for every i
            Σ v_i ≤ varAND + N − 1

        Constraint‑only: enforce ``Σ v_i ≥ N`` (all inputs are 1).
        """

    # ------------------------------------------------------------------
    # N‑ary disjunction (OR)
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def orVar(self, m, *_var, onlyConstrains: bool = False):
        """General **N‑ary disjunction**.

        Reified form:
            v_i ≤ varOR                for every i
            Σ v_i ≥ varOR

        Constraint‑only: enforce ``Σ v_i ≥ 1``.
        """

    # ------------------------------------------------------------------
    # NAND (NOT‑AND)
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def nandVar(self, m, *_var, onlyConstrains: bool = False):
        """General **N‑ary NAND**.

        Reified form:
            NOT(varNAND) ≤ v_i          for every i
            Σ v_i ≤ NOT(varNAND) + N − 1

        Constraint‑only: enforce ``Σ v_i ≤ N − 1`` (not all can be True).
        """

    # ------------------------------------------------------------------
    # NOR (NOT‑OR)
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def norVar(self, m, *_var, onlyConstrains: bool = False):
        """General **N‑ary NOR**.

        Reified form:
            v_i ≤ NOT(varNOR)           for every i
            Σ v_i ≥ NOT(varNOR)

        Constraint‑only: enforce ``Σ v_i ≤ 0`` (all inputs 0).
        """

    # ------------------------------------------------------------------
    # XOR / XNOR
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def xorVar(self, m, *var, onlyConstrains: bool = False):
        """Two‑input **exclusive‑or**.

        Reified form (returns *varXOR*): standard 4‑constraint encoding
        ensuring *varXOR* = 1 exactly when the inputs differ.

        Constraint‑only: enforce ``Σ v_i == 1`` (one True, others False).
        """

    # ------------------------------------------------------------------
    # Implication
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def ifVar(self, m, _var1, _var2, *, onlyConstrains: bool = False):
        """Logical **implication**  (var1 ⇒ var2).

        Reified form (returns *varIF*):
            1 − var1 ≤ varIF
            var2     ≤ varIF
            1 − var1 + var2 ≥ varIF
        so *varIF* = 1 unless var1 = 1 and var2 = 0.

        Constraint‑only: enforce ``var1 ≤ var2``.
        """

    # ------------------------------------------------------------------
    # Equivalence (XNOR)
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def equivalenceVar(self, m, *var, onlyConstrains: bool = False):
        """Logical **equivalence** (biconditional/if-and-only-if).

        Returns true when all input variables have the same truth value 
        (all true or all false).

        For binary case: equiv(a, b) = (a ↔ b) = (a → b) ∧ (b → a)
        For n-ary case: equiv(a, b, c, ...) = (all true) ∨ (all false)

        Reified form (returns *varEQ*): constraints ensure *varEQ* = 1 
        exactly when all inputs are equivalent.

        Constraint‑only: enforce that all variables have the same truth value.
        
        Args:
            m: Model context
            *var: Variable number of boolean variables to compare
            onlyConstrains: If True, return loss (constraint violation);
                            if False, return success (truth degree)
        
        Returns:
            Truth degree of equivalence or constraint violation measure
        """

    # ------------------------------------------------------------------
    # Counting primitives
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def countVar(
        self, m, *_var,
        onlyConstrains: bool = False,
        limitOp: str = "None",
        limit: int = 1,
        logicMethodName: str = "COUNT",
    ):
        """Compare the **number of True literals** in *var* against a constant.

        Supports three relations via *limitOp*:
            • '>='  (at least *limit* Trues)
            • '<='  (at most  *limit* Trues)
            • '=='  (exactly *limit* Trues)

        Reified form returns a binary *varCOUNT* that is 1 when the chosen
        relation is satisfied. Constraint‑only mode merely imposes the count
        without introducing *varCOUNT*.
        """

    @abc.abstractmethod
    def compareCountsVar(
        self,
        m,
        varsA,
        varsB,
        *,
        compareOp: str = ">",
        diff: int = 0,
        onlyConstrains: bool = False,
        logicMethodName: str = "COUNT_CMP",
    ):
        """Compare the counts of **two sets** of literals.

        Encodes the relation:
            Σ(varsA)   compareOp   Σ(varsB) + diff

        where ``compareOp ∈ {'>', '>=', '<', '<=', '==', '!='}``.
        With *onlyConstrains=False* the method returns a fresh binary that is
        1 when the relation holds. Otherwise it just adds the constraints.
        """

    # ------------------------------------------------------------------
    # Fixed label helper
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def fixedVar(self, m, _var, *, onlyConstrains: bool = False):
        """Fix an ILP literal to its ground‑truth label.

        • If the data node says the variable is *true*, constrain ``_var == 1``.
        • If labelled *false*, constrain ``_var == 0``.
        • If the label is missing (e.g. VTag = "-100"), simply return 1 so
          the downstream logic treats it as satisfied.
        """
        
    # ------------------------------------------------------------------
    # Summation
    # ------------------------------------------------------------------
    def summationVar(self, m, *_var):
        "Sums up a list of binary literals to an integer literal."
        
    # ------------------------------------------------------------------
    # Definite Description (Iota)
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def iotaVar(
        self, 
        m, 
        *_var, 
        onlyConstrains: bool = False,
        temperature: float = 1.0,
        logicMethodName: str = "IOTA"
    ):
        """Definite description operator - selects THE unique entity satisfying condition.
        
        Implements Russell's iota operator (ι): ιx.φ(x) denotes "the unique x such that φ(x)".
        
        Semantics:
            - Presupposes existence: at least one entity satisfies φ
            - Presupposes uniqueness: at most one entity satisfies φ
            - Returns: the entity that satisfies φ (or a distribution over entities)
        
        Implementation varies by processor:
            - ILP: Creates selection variables with exactly-one constraint
            - Loss: Returns softmax distribution over satisfaction scores
            - Sample: Returns index of satisfying entity in each sample
            - Verify: Returns index if exactly one satisfies, -1 otherwise
        
        Args:
            m: Model context (Gurobi model for ILP, None for others)
            *_var: Variables representing condition satisfaction for each entity
                - ILP: Binary Gurobi variables
                - Loss: Probability tensors in [0,1]
                - Sample: Binary sampled tensors
                - Verify: Discrete 0/1 values
            onlyConstrains: 
                - If True: Add constraints / return loss (constraint violation measure)
                - If False: Return selection variables / distribution / index
            temperature: Softmax temperature for differentiable selection (lower = harder)
            logicMethodName: Name for logging and constraint naming
        
        Returns:
            - ILP (onlyConstrains=False): List of binary selection variables [s_0, ..., s_n]
            - ILP (onlyConstrains=True): None (constraints added to model)
            - Loss (onlyConstrains=False): Tensor [n] representing selection distribution
            - Loss (onlyConstrains=True): Scalar loss tensor
            - Sample (onlyConstrains=False): Tensor [sample_size] of selected indices
            - Sample (onlyConstrains=True): Boolean tensor [sample_size] of violations
            - Verify (onlyConstrains=False): Integer index of selected entity, -1 if invalid
            - Verify (onlyConstrains=True): 1 if valid (exactly one), 0 if violated
        
        Raises:
            Exception: In ILP mode, if model is infeasible (no entity can satisfy)
        """