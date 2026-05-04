"""DomiKnowS graph encoder for token-generation tasks.

This module bridges the generation vocabulary and constraints subsystems with
the DomiKnowS graph/concept layer.  It provides:

:class:`GenerationBundle`
    A plain dataclass holding all graph objects built by
    :meth:`GenerationEncoder.build_graph`, plus the vocabulary and constraints
    used during construction.

:class:`GenerationGraphContext`
    A concrete implementation of the
    :class:`~.constraints.DomiKnowSGenerationContext` protocol.  Wraps the
    graph objects from a :class:`GenerationBundle` and translates vocabulary
    predicates into DomiKnowS logical expressions.

:class:`GenerationEncoder`
    Factory that constructs the shared DomiKnowS graph for a generation task
    and returns ``(graph, bundle)``.

Typical usage::

    encoder = GenerationEncoder(vocab=my_tokens, eos_token="<eos>")
    graph, bundle = encoder.build_graph(constraints=[max_non_eos(10)])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .constraints import GenerationConstraint
from .vocabulary import TokenVocabulary


@dataclass
class GenerationBundle:
    """Container for all DomiKnowS graph objects produced by :class:`GenerationEncoder`.

    Instances are returned by :meth:`GenerationEncoder.build_graph` and passed
    to DomiKnowS programs, sensors, and solvers.

    Attributes:
        text: The top-level ``text`` :class:`~domiknows.graph.Concept`
            representing a whole generated sequence.
        token: The ``token`` :class:`~domiknows.graph.Concept` representing
            one position in the sequence.
        contains: The ``contains`` relation linking ``text`` to ``token``
            (one-to-many).
        generated_token: An :class:`~domiknows.graph.EnumConcept` attached to
            ``token`` with one enum value per vocabulary label index.  Sensors
            write predicted label distributions here.
        is_before_rel: A ``Concept`` encoding the pairwise ordering relation
            between token positions (used by ordering constraints).
        first_token: The first argument role of ``is_before_rel``.
        second_token: The second argument role of ``is_before_rel``.
        context: The :class:`GenerationGraphContext` that was used to compile
            DomiKnowS constraints during graph construction.
        constraints: Tuple of :class:`~.constraints.GenerationConstraint`
            objects that were registered when the graph was built.
        vocabulary: The :class:`~.vocabulary.TokenVocabulary` used to build
            the graph.
    """

    text: object
    token: object
    contains: object
    generated_token: object
    is_before_rel: object
    first_token: object
    second_token: object
    context: object
    constraints: tuple[GenerationConstraint, ...]
    vocabulary: TokenVocabulary


class GenerationGraphContext:
    """Concrete implementation of the DomiKnowS generation context protocol.

    Wraps the graph objects produced by :class:`GenerationEncoder` and
    translates vocabulary-level token predicates into DomiKnowS logical
    expressions.  Passed to each constraint's ``apply_domiknows`` method
    during graph construction.

    Attributes:
        vocabulary: The :class:`~.vocabulary.TokenVocabulary` for the current
            generation task.
        generated_token: The ``generated_token`` EnumConcept node.
        is_before_rel: The ``is_before_rel`` concept encoding pair ordering.
        first_token: The first argument role of ``is_before_rel``.
        second_token: The second argument role of ``is_before_rel``.
    """

    def __init__(
        self,
        vocabulary: TokenVocabulary,
        generated_token,
        is_before_rel,
        first_token,
        second_token,
    ):
        """Store graph objects needed to build DomiKnowS predicates.

        Args:
            vocabulary: Token vocabulary providing label ↔ token mappings.
            generated_token: ``EnumConcept`` node for per-position labels.
            is_before_rel: Pairwise ordering concept.
            first_token: First argument role of *is_before_rel*.
            second_token: Second argument role of *is_before_rel*.
        """
        self.vocabulary = vocabulary
        self.generated_token = generated_token
        self.is_before_rel = is_before_rel
        self.first_token = first_token
        self.second_token = second_token

    def token_value(self, token: str, variable: str, path=None):
        """Return a DomiKnowS predicate asserting *variable* holds *token*.

        Resolves *token* to its integer label via the vocabulary, then
        retrieves the corresponding enum-concept attribute on
        ``generated_token`` and wraps it in a concept-call expression.

        Args:
            token: Surface-form token string to match.
            variable: DomiKnowS variable name (e.g. ``"x"``).
            path: Optional ``path=`` argument forwarded to the concept call;
                used by relational constraints that reference tokens through
                a relation (e.g. ``is_before_rel``).

        Returns:
            A DomiKnowS concept-call expression ``generated_token.<label>(variable)``.
        """
        # Look up the enum attribute by label index (stored as a string name).
        token_concept = getattr(self.generated_token, str(self.vocabulary.label_for_token(token)))
        if path is None:
            return token_concept(variable)
        return token_concept(variable, path=path)

    def non_eos(self, variable: str):
        """Return a DomiKnowS predicate asserting *variable* is not the EOS token.

        Implemented as ``notL(token_value(eos_token, variable))``.

        Args:
            variable: DomiKnowS variable name (e.g. ``"x"``).

        Returns:
            A DomiKnowS ``notL`` expression.
        """
        from domiknows.graph.logicalConstrain import notL

        return notL(self.token_value(self.vocabulary.eos_token, variable))


class GenerationEncoder:
    """Build the common DomiKnowS graph used for token generation.

    :class:`GenerationEncoder` owns the :class:`~.vocabulary.TokenVocabulary`
    and constructs the DomiKnowS graph (concepts, relations, enum values, and
    logical constraints) on demand via :meth:`build_graph`.

    The graph layout produced by :meth:`build_graph` is:

    .. code-block:: text

        text  --contains-->  token
                              |
                              +-- generated_token  (EnumConcept, one value per label)

        is_before_rel  --has_a(arg1)-->  token  (first_token)
                       --has_a(arg2)-->  token  (second_token)

    Attributes:
        vocabulary: The :class:`~.vocabulary.TokenVocabulary` built from the
            supplied *vocab* and *eos_token* arguments.
        graph_name: Name passed to the DomiKnowS :class:`~domiknows.graph.Graph`
            constructor.
        clear_graph: Whether to clear the global DomiKnowS concept/relation
            registries before each call to :meth:`build_graph`.
    """

    def __init__(
        self,
        vocab: Sequence[str],
        eos_token: str,
        graph_name: str = "main",
        tokenizer: object | None = None,
        clear_graph: bool = True,
    ):
        """Initialise the encoder with a vocabulary.

        Args:
            vocab: Ordered sequence of surface-form tokens that defines the
                generation vocabulary (label 0 = ``vocab[0]``, etc.).
            eos_token: The end-of-sequence token string.  Must be present in
                *vocab* or handled by
                :class:`~.vocabulary.TokenVocabulary`.
            graph_name: Name given to the DomiKnowS graph.  Defaults to
                ``"main"``.
            tokenizer: Optional HuggingFace (or compatible) tokenizer.  When
                provided, the vocabulary can map raw tokenizer IDs to labels
                and vice-versa.
            clear_graph: If ``True`` (default), calls
                ``Graph.clear()`` / ``Concept.clear()`` / ``Relation.clear()``
                before building the graph so previous graphs do not interfere.
        """
        self.vocabulary = TokenVocabulary(vocab, eos_token=eos_token, tokenizer=tokenizer)
        self.graph_name = graph_name
        self.clear_graph = clear_graph

    def build_graph(self, constraints: Sequence[GenerationConstraint] = ()) -> tuple[object, GenerationBundle]:
        """Construct the DomiKnowS graph and compile constraints into it.

        Steps performed:
        1. Optionally clear global DomiKnowS registries.
        2. Create the ``text``, ``token``, and ``is_before_rel`` concepts.
        3. Attach a ``generated_token`` :class:`~domiknows.graph.EnumConcept`
           with one value per vocabulary label.
        4. Build a :class:`GenerationGraphContext` and call
           ``apply_domiknows`` on each constraint that supports it.
        5. Return the graph and a :class:`GenerationBundle`.

        Args:
            constraints: Sequence of :class:`~.constraints.GenerationConstraint`
                objects to register.  Constraints with
                ``supports_domiknows = False`` are stored in the bundle but
                not compiled into the graph.

        Returns:
            A ``(graph, bundle)`` tuple where *graph* is the DomiKnowS
            :class:`~domiknows.graph.Graph` instance and *bundle* is a
            :class:`GenerationBundle` holding all concept/relation references.
        """
        from domiknows.graph import Concept, EnumConcept, Graph, Relation

        if self.clear_graph:
            # Reset global registries to avoid concept name collisions with
            # graphs built in previous calls or test runs.
            Graph.clear()
            Concept.clear()
            Relation.clear()

        with Graph(self.graph_name) as graph:
            text = Concept(name="text")
            token = Concept(name="token")
            # `contains` is unpacked from a one-element tuple returned by `.contains()`.
            contains, = text.contains(token)

            # Pairwise ordering relation between two token positions.
            is_before_rel = Concept(name="is_before_rel")
            first_token, second_token = is_before_rel.has_a(arg1=token, arg2=token)

            # EnumConcept with one value per label index (stored as string names).
            generated_token = token(
                name="generated_token",
                ConceptClass=EnumConcept,
                values=[str(i) for i in range(self.vocabulary.label_count)],
            )

            # Build the context and compile each DomiKnowS-compatible constraint.
            context = GenerationGraphContext(
                self.vocabulary,
                generated_token,
                is_before_rel,
                first_token,
                second_token,
            )
            for constraint in constraints:
                if constraint.supports_domiknows:
                    constraint.apply_domiknows(context)

        bundle = GenerationBundle(
            text=text,
            token=token,
            contains=contains,
            generated_token=generated_token,
            is_before_rel=is_before_rel,
            first_token=first_token,
            second_token=second_token,
            context=context,
            constraints=tuple(constraints),
            vocabulary=self.vocabulary,
        )
        return graph, bundle
