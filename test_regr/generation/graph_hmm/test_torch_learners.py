"""
Tests for PyTorch generation heads - integrating HMM/automaton with neural networks.

This module tests the integration of graph-based automata with PyTorch for text generation:
- GraphHMMGenerationHead: PyTorch module wrapping DomiKnowSAwareHMM
- GraphSpectralGenerationHead: PyTorch module wrapping GraphSpectralAutomaton
- Parameter management: Converting learned models to trainable PyTorch parameters
- Constrained decoding: Generating text while respecting constraints
- Factory methods: Creating generation heads from fitted models

Generation heads enable constraint-respecting neural text generation by combining:
1. Learned HMM/automaton probabilities as initial parameters
2. PyTorch parameter management for end-to-end training
3. Constrained decoding that enforces DFA constraints during generation
"""
import torch

from domiknows.generation import constrained_label_greedy_decode
from domiknows.generation.automata import DFA
from domiknows.generation.graph_hmm import (
    DomiKnowSAwareHMM,
    GraphHMMGenerationHead,
    GraphSpectralAutomaton,
    GraphSpectralGenerationHead,
)
from domiknows.generation.vocabulary import TokenVocabulary


def test_graph_hmm_generation_head_forward_and_masks_are_finite():
    """
    Test that GraphHMMGenerationHead computes finite log-probabilities with valid gradients.
    
    The generation head should:
    - Produce valid log-probability outputs
    - Maintain zero probabilities for masked transitions/emissions
    - Support backpropagation for gradient-based training
    """
    head = GraphHMMGenerationHead(
        n_hidden_states=2,
        label_count=2,
        transition_mask=torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        emission_mask=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        pad_size=3,
        trainable=True,
    )

    output = head(None, torch.tensor([[9]]), torch.tensor([0, 1, 1]))
    loss = -output[0, 0]
    loss.backward()

    assert output.shape == (3, 2)
    assert torch.isfinite(output).all()
    assert head.transition_probs[0, 1].item() == 0.0
    assert head.emission_probs[0, 1].item() == 0.0
    assert head.trainable_parameter_names() == ["initial_logits", "transition_logits", "emission_logits"]


def test_graph_hmm_generation_head_from_fitted_hmm_copies_parameters():
    """
    Test that generation head can be initialized from a fitted DomiKnowSAwareHMM.
    
    This allows warm-starting neural training with learned HMM parameters.
    """
    learner = DomiKnowSAwareHMM(
        graph=None,
        n_hidden_states=2,
        state_names=["A", "B"],
        symbols=[0, 1],
        transition_mask=torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64),
        emission_mask=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        random_seed=3,
    ).fit([[0, 1], [0, 1, 1]], max_iter=2)

    head = GraphHMMGenerationHead.from_graph_hmm(learner, trainable=False, pad_size=3)

    assert torch.allclose(head.transition_probs.double(), learner.transition_, atol=1e-5)
    assert torch.allclose(head.emission_probs.double(), learner.emission_, atol=1e-5)
    assert head.trainable_parameter_names() == []


def test_graph_hmm_to_torch_learner_factory():
    """Test the factory method that converts DomiKnowSAwareHMM directly to GraphHMMGenerationHead."""
    learner = DomiKnowSAwareHMM(
        graph=None,
        n_hidden_states=2,
        state_names=["A", "B"],
        symbols=[0, 1],
        random_seed=5,
    ).fit([[0, 1], [0, 0]], max_iter=1)

    head = learner.to_torch_learner(trainable=True, pad_size=2)

    assert isinstance(head, GraphHMMGenerationHead)
    assert head.trainable_parameter_names()


def test_graph_hmm_generation_head_can_condition_initial_state_on_prompt():
    """Prompt-conditioned graph-HMM heads should let prompt ids change label probabilities."""
    head = GraphHMMGenerationHead(
        n_hidden_states=2,
        label_count=2,
        initial=torch.tensor([0.5, 0.5]),
        transition=torch.eye(2),
        emission=torch.eye(2),
        emission_mask=torch.eye(2),
        pad_size=2,
        trainable=True,
        prompt_conditioning="initial",
        prompt_vocab_size=8,
        prompt_hidden_size=4,
        label_to_token_id=(0, 1),
    )
    with torch.no_grad():
        head.prompt_initial_projector.weight.zero_()
        head.prompt_initial_projector.bias.zero_()
        head.prompt_embedding.weight.zero_()
        head.prompt_embedding.weight[2, 0] = 10.0
        head.prompt_embedding.weight[3, 0] = -10.0
        head.prompt_initial_projector.weight[0, 0] = 1.0
        head.prompt_initial_projector.weight[1, 0] = -1.0

    prompt_two = head.next_label_logits(torch.tensor([2]))
    prompt_three = head.next_label_logits(torch.tensor([3]))

    assert prompt_two[0] > prompt_two[1]
    assert prompt_three[1] > prompt_three[0]
    assert "prompt_embedding.weight" in head.trainable_parameter_names()


def test_graph_hmm_generation_head_decodes_with_label_dfa():
    """
    Test constrained decoding using a DFA that filters generated sequences.
    
    The DFA enforces that certain symbols can only appear after certain prefixes,
    ensuring generated text respects complex structural constraints.
    """
    vocab = TokenVocabulary(["<eos>", " The"], eos_token="<eos>")
    dfa = DFA(
        states=frozenset({"start", "seen", "dead"}),
        alphabet=frozenset({0, 1, 2}),
        transitions={
            ("start", 0): "dead",
            ("start", 1): "seen",
            ("start", 2): "start",
            ("seen", 0): "seen",
            ("seen", 1): "seen",
            ("seen", 2): "seen",
            ("dead", 0): "dead",
            ("dead", 1): "dead",
            ("dead", 2): "dead",
        },
        start_state="start",
        accepting_states=frozenset({"seen"}),
        dead_states=frozenset({"dead"}),
    )
    head = GraphHMMGenerationHead(
        n_hidden_states=3,
        label_count=3,
        initial=torch.tensor([0.01, 1.0, 0.01]),
        transition=torch.eye(3),
        emission=torch.eye(3),
        emission_mask=torch.eye(3),
        pad_size=2,
        trainable=False,
        label_to_token_id=(0, 1, 2),
    )

    result = constrained_label_greedy_decode(head, torch.tensor([2]), vocab, dfa, max_new_tokens=2)

    assert result.accepted
    assert 1 in result.labels


def test_graph_spectral_generation_head_forward_and_signed_logits():
    """
    Test that GraphSpectralGenerationHead computes finite logits from signed operators.
    
    Spectral automata use signed matrices (with positive and negative entries),
    unlike HMMs which use probability matrices. The generation head must handle
    this signed representation and convert it to logits.
    """
    head = GraphSpectralGenerationHead(
        label_count=2,
        state_count=2,
        initial=torch.tensor([1.0, -0.5]),
        final=torch.tensor([0.3, -0.2]),
        operators=torch.tensor(
            [
                [[0.5, -0.1], [0.2, 0.3]],
                [[-0.4, 0.6], [0.1, -0.2]],
            ]
        ),
        pad_size=3,
        trainable=True,
    )

    output = head(None, torch.tensor([[9]]), torch.tensor([0, 1, 0]))
    logits = head.next_label_logits(torch.tensor([0, 1]))

    assert output.shape == (3, 2)
    assert torch.isfinite(output).all()
    assert torch.isfinite(logits).all()
    assert head.trainable_parameter_names() == ["initial", "final", "operators"]


def test_graph_spectral_generation_head_from_fitted_automaton():
    """Test that generation head can be initialized from a fitted GraphSpectralAutomaton."""
    automaton = GraphSpectralAutomaton(symbols=[0, 1])
    automaton.fit(
        [[0], [1], [0, 1], [0, 1]],
        prefixes=[(), (0,), (1,)],
        suffixes=[(), (0,), (1,)],
        rank=2,
    )

    head = GraphSpectralGenerationHead.from_graph_spectral(automaton, trainable=False, pad_size=3)

    assert torch.allclose(head.initial, automaton.initial.float())
    assert torch.allclose(head.final, automaton.final.float())
    assert head.operators.shape == (2, 2, 2)
    assert head.trainable_parameter_names() == []


def test_graph_spectral_to_torch_learner_factory():
    """Test the factory method that converts GraphSpectralAutomaton to GraphSpectralGenerationHead."""
    automaton = GraphSpectralAutomaton(symbols=[0, 1])
    automaton.fit(
        [[0], [1], [0, 1]],
        prefixes=[(), (0,), (1,)],
        suffixes=[(), (0,), (1,)],
        rank=2,
    )

    head = automaton.to_torch_learner(trainable=True, pad_size=2)

    assert isinstance(head, GraphSpectralGenerationHead)
    assert head.trainable_parameter_names()


def test_graph_hmm_sequence_log_probs_respects_lengths_mask():
    head = GraphHMMGenerationHead(
        n_hidden_states=2,
        label_count=2,
        pad_size=4,
        trainable=False,
    )

    outputs = head.sequence_log_probs(torch.tensor([[1, 1, 0, 0]]), lengths=torch.tensor([2]))

    assert outputs.shape == (1, 4, 2)
    assert torch.allclose(outputs[0, 2], torch.zeros(2, dtype=outputs.dtype))
    assert torch.allclose(outputs[0, 3], torch.zeros(2, dtype=outputs.dtype))


def test_graph_hmm_head_dynamic_all_zero_row_stays_blocked():
    """Torch head transition projection must preserve dynamic hard blocks."""

    def dynamic_transition(context):
        if context.prefix == (0,):
            return torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        return None

    head = GraphHMMGenerationHead(
        n_hidden_states=2,
        label_count=2,
        symbols=[0, 1],
        transition=torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32),
        emission=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        initial=torch.tensor([1.0, 0.0], dtype=torch.float32),
        dynamic_transition=dynamic_transition,
        trainable=False,
    )

    transition = head._transition_for_prefix(
        step=0,
        prefix=(0,),
        belief=torch.tensor([1.0, 0.0], dtype=torch.float32),
    )

    assert torch.allclose(transition[0], torch.zeros(2, dtype=transition.dtype))


def test_graph_spectral_sequence_log_probs_respects_lengths_mask():
    head = GraphSpectralGenerationHead(
        label_count=2,
        state_count=2,
        pad_size=4,
        trainable=False,
    )

    outputs = head.sequence_log_probs(torch.tensor([[1, 1, 0, 0]]), lengths=torch.tensor([2]))

    assert outputs.shape == (1, 4, 2)
    assert torch.allclose(outputs[0, 2], torch.zeros(2, dtype=outputs.dtype))
    assert torch.allclose(outputs[0, 3], torch.zeros(2, dtype=outputs.dtype))
