"""DomiKnowS graph encoder for token-generation tasks.

This module bridges the generation vocabulary and constraints subsystems with
the DomiKnowS graph/concept layer.  It provides:

:class:`GenerationBundle`
    A plain dataclass holding all graph objects built by
    :meth:`GenerationEncoder.build_graph`, plus the vocabulary used during
    construction.

:class:`GenerationGraphContext`
    A concrete implementation of the
    :class:`~.dfa.DomiKnowSGenerationContext` protocol.  Wraps the
    graph objects from a :class:`GenerationBundle` and translates vocabulary
    predicates into DomiKnowS logical expressions.

:class:`GenerationEncoder`
    Factory that constructs the shared DomiKnowS graph for a generation task
    and returns ``(graph, bundle)``.

Typical usage::

    encoder = GenerationEncoder(vocab=my_tokens, eos_token="<eos>")
    graph, bundle = encoder.build_graph()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
        vocabulary: The :class:`~.dfa.vocabulary.TokenVocabulary` used to build
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
    vocabulary: TokenVocabulary


class GenerationGraphContext:
    """Concrete implementation of the DomiKnowS generation context protocol.

    Wraps the graph objects produced by :class:`GenerationEncoder` and
    translates vocabulary-level token predicates into DomiKnowS logical
    expressions.  Passed to each constraint's ``apply_domiknows`` method
    during graph construction.

    Attributes:
        vocabulary: The :class:`~.dfa.vocabulary.TokenVocabulary` for the current
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
        token_concept = getattr(self.generated_token, self._enum_name_for_token(token))
        if path is None:
            return token_concept(variable)
        return token_concept(variable, path=path)

    def _enum_name_for_token(self, token: str) -> str:
        """Return the generated-token enum value name for a vocabulary token."""
        label = self.vocabulary.label_for_token(token)
        enum_values = tuple(getattr(self.generated_token, "enum", ()))
        if 0 <= label < len(enum_values):
            return enum_values[label]
        return str(label)

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


def generation_bundle_from_graph(
    graph,
    *,
    vocab: Sequence[str],
    eos_token: str,
    tokenizer: object | None = None,
    text_name: str = "text",
    token_name: str = "token",
    generated_token_name: str = "generated_token",
    before_relation_name: str = "is_before_rel",
    first_role_name: str = "arg1",
    second_role_name: str = "arg2",
) -> GenerationBundle:
    """Wrap an existing traditional DomiKnowS graph as a generation bundle.

    The graph must already contain the standard generation shape:

    .. code-block:: text

        text  --contains-->  token
        token --is_a-->      generated_token(EnumConcept)

        is_before_rel --has_a(arg1)--> token
                      --has_a(arg2)--> token

    Unlike :class:`GenerationEncoder`, this helper does not create graph
    objects.  It resolves the existing objects by name, validates that they
    match the compact :class:`TokenVocabulary`, and returns a
    :class:`GenerationBundle` pointing at those original graph objects.

    Args:
        graph: Existing DomiKnowS graph.
        vocab: Ordered known token strings.  ``_other`` is added by
            :class:`TokenVocabulary` and must be present in the graph enum as
            the final compact label.  Enum values may be either numeric label
            names (``"0"``, ``"1"``, ...) or readable semantic names ordered
            exactly like ``vocab + [_other]``.
        eos_token: End-of-sequence token string.
        tokenizer: Optional tokenizer for token-id mappings.
        text_name: Name of the sequence/root concept.
        token_name: Name of the per-position token concept.
        generated_token_name: Name of the token enum concept.
        before_relation_name: Name of the pairwise ordering relation concept.
        first_role_name: Role name for the first endpoint of
            ``before_relation_name``.
        second_role_name: Role name for the second endpoint of
            ``before_relation_name``.
        The helper assumes graph-level raw constraints are already written on
        *graph*.

    Returns:
        A :class:`GenerationBundle` backed by the existing graph objects.

    Raises:
        ValueError: If a required concept/relation is missing or if the
            generated-token enum does not exactly match the compact label
            space.
    """
    vocabulary = TokenVocabulary(vocab, eos_token=eos_token, tokenizer=tokenizer)

    text = _required_concept(graph, text_name)
    token = _required_concept(graph, token_name)
    generated_token = _required_concept(graph, generated_token_name)
    is_before_rel = _required_concept(graph, before_relation_name)

    contains = _find_contains_relation(text, token)
    first_token = _find_has_a_role(is_before_rel, first_role_name, token)
    second_token = _find_has_a_role(is_before_rel, second_role_name, token)
    _validate_generated_token_enum(generated_token, vocabulary)

    context = GenerationGraphContext(
        vocabulary,
        generated_token,
        is_before_rel,
        first_token,
        second_token,
    )
    return GenerationBundle(
        text=text,
        token=token,
        contains=contains,
        generated_token=generated_token,
        is_before_rel=is_before_rel,
        first_token=first_token,
        second_token=second_token,
        context=context,
        vocabulary=vocabulary,
    )


def _required_concept(graph, name: str):
    concept = graph.findConcept(name) if hasattr(graph, "findConcept") else None
    if concept is None:
        raise ValueError(f"generation graph is missing required concept {name!r}")
    return concept


def _find_contains_relation(text, token):
    relations = list(text.contains())
    matches = [relation for relation in relations if relation.dst is token]
    if len(matches) != 1:
        raise ValueError(
            "generation graph must contain exactly one "
            f"{text.name}.contains({token.name}) relation; found {len(matches)}"
        )
    return matches[0]


def _find_has_a_role(relation_concept, role_name: str, token):
    relations = list(relation_concept.has_a())
    matches = [relation for relation in relations if relation.name == role_name and relation.dst is token]
    if len(matches) != 1:
        raise ValueError(
            "generation graph must contain exactly one "
            f"{relation_concept.name}.has_a({role_name}={token.name}) role; found {len(matches)}"
        )
    return matches[0]


def _validate_generated_token_enum(generated_token, vocabulary: TokenVocabulary) -> None:
    enum_values = tuple(getattr(generated_token, "enum", ()))
    expected_numeric = tuple(str(label) for label in range(vocabulary.label_count))
    if enum_values == expected_numeric:
        return
    if len(enum_values) == vocabulary.label_count:
        return
    if enum_values != expected_numeric:
        raise ValueError(
            f"generated_token enum must contain {vocabulary.label_count} ordered labels "
            f"for {vocabulary.labels}; got {enum_values}. Include the reserved "
            f"{vocabulary.other_token!r} label."
        )


class GenerationEncoder:
    """Build the common DomiKnowS graph used for token generation.

    :class:`GenerationEncoder` owns the :class:`~.dfa.vocabulary.TokenVocabulary`
    and constructs the DomiKnowS graph concepts, relations, and enum values on
    demand via :meth:`build_graph`.

    The graph layout produced by :meth:`build_graph` is:

    .. code-block:: text

        text  --contains-->  token
                              |
                              +-- generated_token  (EnumConcept, one value per label)

        is_before_rel  --has_a(arg1)-->  token  (first_token)
                       --has_a(arg2)-->  token  (second_token)

    Attributes:
        vocabulary: The :class:`~.dfa.vocabulary.TokenVocabulary` built from the
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
                :class:`~.dfa.vocabulary.TokenVocabulary`.
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

    def build_graph(self) -> tuple[object, GenerationBundle]:
        """Construct the DomiKnowS graph.

        Steps performed:
        1. Optionally clear global DomiKnowS registries.
        2. Create the ``text``, ``token``, and ``is_before_rel`` concepts.
        3. Attach a ``generated_token`` :class:`~domiknows.graph.EnumConcept`
           with one value per vocabulary label.
        4. Build a :class:`GenerationGraphContext`.
        5. Return the graph and a :class:`GenerationBundle`.

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

            # Build the context used by graph-constraint helper functions.
            context = GenerationGraphContext(
                self.vocabulary,
                generated_token,
                is_before_rel,
                first_token,
                second_token,
            )

        bundle = GenerationBundle(
            text=text,
            token=token,
            contains=contains,
            generated_token=generated_token,
            is_before_rel=is_before_rel,
            first_token=first_token,
            second_token=second_token,
            context=context,
            vocabulary=self.vocabulary,
        )
        return graph, bundle
