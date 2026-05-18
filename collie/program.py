"""Collie Program Builder - Constructs learning programs for sequence generation with graph-based constraints.

This module provides utilities to build DomiKnowS programs for constrained sequence generation tasks,
supporting both traditional neural models (TinyModel) and graph-based HMM learners for integrating
structural constraints into the learning process.
"""

from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program import LearningBasedProgram, SolverPOIProgram

from domiknows.program.metric import MacroAverageTracker
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.generation import (
    GenerationLossWeights,
    GraphHMMGenerationHead,
    GraphSpectralGenerationHead,
    compute_generation_training_loss,
    constraints_to_dfa_from_graph,
    discover_generation_enforcement,
    generation_bundle_from_graph,
)

from tokens import TokenMap, tokenize
import torch
from typing import Any, Literal, TYPE_CHECKING
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
else:
    PreTrainedModel = Any
    PreTrainedTokenizer = Any

from graph import EOS_TOKEN, build_graph
from model import TinyModel


def build_program(
        label_map: TokenMap,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        vocab: list[str],
        pad_size: int = 32,
        model_mode: Literal['tf', 'generate'] = 'generate',
        ilp: bool = False,
        constrained_decoding: bool = False,
        graph_hmm_learner: Literal['none', 'hmm', 'spectral'] = 'none',
        graph_hmm_source: Literal['target', 'generated'] = 'target',
    ) -> LearningBasedProgram:
    """Build a DomiKnowS learning program for constrained sequence generation.
    
    Args:
        label_map: TokenMap object mapping token IDs to vocabulary indices.
        model: Pretrained HuggingFace model for text generation.
        tokenizer: Pretrained tokenizer matching the model.
        vocab: List of vocabulary tokens to predict.
        pad_size: Maximum sequence length for generation (default: 32).
        model_mode: 'tf' for teacher-forcing, 'generate' for greedy decoding (default: 'generate').
        ilp: Whether to enable ILP inference for constraint satisfaction (default: False).
        constrained_decoding: Whether to mask logits with DFA constraints (default: False).
        graph_hmm_learner: Type of graph-based learner: 'none', 'hmm', or 'spectral' (default: 'none').
        graph_hmm_source: Training source for graph HMM: 'target' labels or 'generated' labels (default: 'target').
    
    Returns:
        A PrimalDualProgram instance configured with the specified learning setup.
    
    Raises:
        ValueError: If graph_hmm_source != 'target' when graph_hmm_learner is 'none'.
    """
    if graph_hmm_learner == 'none' and graph_hmm_source != 'target':
        raise ValueError("graph_hmm_source only applies when graph_hmm_learner is enabled")

    # Convert vocabulary tokens to their token IDs using the tokenizer
    vocab_ids = [tokenizer.encode(v)[0] for v in vocab]

    # Build constraint graph and extract generation bundle with node/relation types
    graph, _graph_parts = build_graph(label_map, tokenizer, vocab)
    bundle = generation_bundle_from_graph(
        graph,
        vocab=vocab,
        eos_token=EOS_TOKEN,
        tokenizer=tokenizer,
    )
    # Unpack bundle components - text, token, and relation nodes/predicates from the graph
    text = bundle.text
    token = bundle.token
    contains = bundle.contains
    generated_token = bundle.generated_token
    is_before_rel = bundle.is_before_rel
    first_token = bundle.first_token
    second_token = bundle.second_token
    token_vocabulary = bundle.vocabulary
    # Compile constraints into a DFA for constrained decoding if enabled
    constrained_dfa = (
        constraints_to_dfa_from_graph(graph, bundle)
        if constrained_decoding
        else None
    )
    # Discover constraint enforcement strategy (soft constraints, hard constraints, etc.)
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="ignore")

    # Register input sensors for reading instruction tokens, target sequences, and testing sequences
    text["instruction_tokens"] = ReaderSensor(keyword="instruction_tokens")
    text["target_tokens"] = ReaderSensor(keyword="target_tokens")
    text["_testing_generated_tokens"] = ReaderSensor(keyword="_testing_generated_tokens")

    def _add_sequence(target, testing_seq):
        """Pad sequences to fixed size and create aligned labels and indices."""
        assert len(target[0]) <= pad_size, f"target sequence is too long: {len(target[0])}"
        assert len(testing_seq[0]) <= pad_size, f"debug sequence is too long: {len(testing_seq[0])}"

        # Pad target and testing sequences to pad_size with EOS token IDs
        # Input format: target, testing_seq are (1, seq_length) tensors
        target_out = torch.cat([
            target[0],
            torch.ones((pad_size - len(target[0]),)) * tokenizer.eos_token_id
        ], dim=0)

        testing_out = torch.cat([
            testing_seq[0],
            torch.ones((pad_size - len(testing_seq[0]),)) * tokenizer.eos_token_id
        ], dim=0)

        # Convert token IDs to label indices using vocab mapping
        target_labels = map_labels(target_out.long())

        # Return: containment mask, target tokens, testing tokens, mapped labels, and token indices
        return torch.ones((pad_size, 1)), target_out, testing_out, target_labels, torch.arange(pad_size)

    # Register joint sensor for sequence alignment and label mapping
    token[contains, 'target', '_testing_generated', 'target_labels', 'token_index'] = JointSensor(text["target_tokens"], text["_testing_generated_tokens"], forward=_add_sequence)

    # Note: Loss computation currently only applies to known vocabulary tokens

    def map_labels(label_vals):
        """Map token IDs to vocabulary indices (OOV tokens get index len(vocab_ids))."""
        new_labels = []
        for label in label_vals:
            if label in vocab_ids:
                # Map known token IDs to their vocabulary index
                new_labels.append(vocab_ids.index(label))
            else:
                # Map unknown tokens to the OOV index
                new_labels.append(len(vocab_ids))
        
        return torch.tensor(new_labels)

    # Register the label mapping sensor as the ground truth for learning
    token[generated_token] = FunctionalSensor(token[contains], "target", forward=lambda _, x: map_labels(x), label=True)

    print(vocab)
    # Choose learning pathway: standard neural model vs. graph-based HMM learner
    if graph_hmm_learner == 'none':
        # Initialize standard neural model wrapper with optional DFA constraint masking
        model = TinyModel(
            model,
            tokenizer,
            label_map,
            vocab=vocab,
            eos_idx=tokenizer.eos_token_id,
            pad_size=pad_size,
            mode=model_mode,
            token_vocabulary=token_vocabulary,
            constrained_dfa=constrained_dfa,
        )

        # Register the neural model as the learner
        token[generated_token] = ModuleLearner(
            token[contains],
            text["instruction_tokens"],
            'target',
            '_testing_generated',
            module=model
        )
    else:
        # Use graph-based HMM learner: prepare transition/emission masks and label mappings
        transition_mask, emission_mask = _collie_graph_hmm_masks(token_vocabulary)
        label_to_token_id = _label_token_id_map(token_vocabulary)
        # Instantiate the appropriate graph HMM head based on learner type
        if graph_hmm_learner == 'hmm':
            graph_head = GraphHMMGenerationHead(
                graph=graph,
                n_hidden_states=token_vocabulary.label_count,
                label_count=token_vocabulary.label_count,
                symbols=tuple(range(token_vocabulary.label_count)),
                state_names=token_vocabulary.labels,
                transition_mask=transition_mask,
                emission_mask=emission_mask,
                pad_size=pad_size,
                label_to_token_id=label_to_token_id,
                trainable=True,
            )
        elif graph_hmm_learner == 'spectral':
            graph_head = GraphSpectralGenerationHead(
                label_count=token_vocabulary.label_count,
                state_count=token_vocabulary.label_count,
                symbols=tuple(range(token_vocabulary.label_count)),
                pad_size=pad_size,
                label_to_token_id=label_to_token_id,
                trainable=True,
            )
        else:
            raise ValueError("graph_hmm_learner must be 'none', 'hmm', or 'spectral'")

        # Configure graph HMM training source: target labels or generated labels from teacher
        if graph_hmm_source == 'target':
            # Train directly on ground-truth target labels
            model = graph_head
            token[generated_token] = ModuleLearner(
                token[contains],
                text["instruction_tokens"],
                'target_labels',
                module=model,
            )
        elif graph_hmm_source == 'generated':
            # Train graph HMM to imitate labels from a frozen neural teacher model
            teacher = TinyModel(
                model,
                tokenizer,
                label_map,
                vocab=vocab,
                eos_idx=tokenizer.eos_token_id,
                pad_size=pad_size,
                mode='generate',
                token_vocabulary=token_vocabulary,
                constrained_dfa=constrained_dfa,
            )
            # Combine teacher predictions with graph HMM head for imitation learning
            model = _GeneratedTokenGraphLearner(teacher, graph_head)
            token[generated_token] = ModuleLearner(
                token[contains],
                text["instruction_tokens"],
                'target',
                '_testing_generated',
                'target_labels',
                module=model,
            )
        else:
            raise ValueError("graph_hmm_source must be 'target' or 'generated'")

    # Register edge sensors for token ordering relations
    def is_before_edges(*args, arg1, arg2):
        """Check if first token appears before second token in sequence."""
        return arg1.getAttribute('token_index') < arg2.getAttribute('token_index')
    
    # Create is_before relation sensor for enforcing token ordering constraints
    is_before_rel[first_token.reversed, second_token.reversed] = CompositionCandidateSensor(
        relations=(first_token.reversed, second_token.reversed),
        forward=is_before_edges
    )

    # return SolverPOIProgram(
    #     graph,
    #     poi=(text, token, is_before_rel),
    #     inferTypes=['local/argmax', 'ILP'] if ilp else ['local/argmax']
    # )

    # Construct the main DomiKnowS program with constraint-aware inference and learning
    program = PrimalDualProgram(
        graph,
        SolverModel,
        poi=(text, token, is_before_rel),
        inferTypes=['local/argmax', 'ILP'] if ilp else ['local/argmax'],  # Enable ILP solver if requested
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        beta=10, device='cpu', tnorm="P", counting_tnorm="P"  # Primal-Dual configuration
    )
    # Attach generation metadata and constraint enforcement info to program
    program.generation_bundle = bundle
    program.generation_enforcement = enforcement
    program.graph_hmm_learner = graph_hmm_learner
    program.graph_hmm_source = graph_hmm_source
    if graph_hmm_learner != 'none':
        program.graph_hmm_model = model
    return program


class _GeneratedTokenGraphLearner(torch.nn.Module):
    """Train a compact graph-HMM head from labels emitted by a frozen generator.
    
    This module implements imitation learning where a lightweight graph HMM head learns
    to predict the same sequence labels as a frozen neural teacher model, enabling
    knowledge distillation of generated sequences into graph-constrained predictions.
    """

    def __init__(self, teacher: TinyModel, head: torch.nn.Module):
        super().__init__()
        # Store teacher model (frozen, not trainable)
        object.__setattr__(self, "_teacher", teacher)
        self.head = head  # Trainable graph HMM head
        self.last_teacher_labels = None  # Cache teacher outputs for debugging
        self.last_teacher_token_ids = None
        self.last_imitation_loss = None  # Cache loss for monitoring
        # Freeze teacher parameters
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.eval()

    def forward(
            self,
            _contains,
            instruction_tokens: torch.Tensor,
            target_tokens: torch.Tensor,
            testing_tokens: torch.Tensor,
            target_labels: torch.Tensor,
        ) -> torch.Tensor:
        """Forward pass: get teacher labels and compute imitation loss.
        
        Args:
            _contains: Containment mask tensor
            instruction_tokens: Instruction/prefix tokens
            target_tokens: Target sequence tokens
            testing_tokens: Testing/generation tokens
            target_labels: Ground truth labels (unused in imitation path)
        
        Returns:
            Log probabilities from the graph HMM head over teacher label sequence.
        """
        # Get frozen teacher predictions without gradient computation
        with torch.no_grad():
            teacher_logprobs = self._teacher(_contains, instruction_tokens, target_tokens, testing_tokens)
            teacher_labels = self._teacher.last_generated_labels
            if teacher_labels is None:
                teacher_labels = torch.argmax(teacher_logprobs, dim=-1).long()
            teacher_token_ids = self._teacher.last_generated_token_ids

        # Compute graph HMM head predictions and loss on teacher-generated sequence
        teacher_labels = teacher_labels.to(dtype=torch.long)
        logprobs = self.head.sequence_log_probs(teacher_labels)
        # Cache outputs for monitoring and debugging
        self.last_teacher_labels = teacher_labels.detach().clone()
        self.last_teacher_token_ids = None if teacher_token_ids is None else teacher_token_ids.detach().clone()
        # Compute negative log-likelihood: how well does head predict teacher's labels?
        self.last_imitation_loss = -logprobs.gather(-1, teacher_labels.unsqueeze(-1)).squeeze(-1).mean()
        return logprobs

    def token_id_for_label(self, label: int) -> int:
        return self.head.token_id_for_label(label)

    def next_label_logits(self, input_ids):
        return self.head.next_label_logits(input_ids)

    def sequence_log_probs(self, target_labels, *, lengths=None):
        return self.head.sequence_log_probs(target_labels, lengths=lengths)

    def trainable_parameter_names(self):
        """Get list of trainable parameter names from the graph HMM head."""
        if hasattr(self.head, "trainable_parameter_names"):
            return self.head.trainable_parameter_names()
        return [name for name, parameter in self.head.named_parameters() if parameter.requires_grad]


def _label_token_id_map(token_vocabulary):
    """Map label indices to their corresponding token IDs.
    
    Args:
        token_vocabulary: Token vocabulary object with label-to-token conversion.
    
    Returns:
        Tuple of token IDs indexed by label (None for unmapped labels).
    """
    token_ids = []
    for label in range(token_vocabulary.label_count):
        try:
            token_ids.append(token_vocabulary.token_id_for_label(label))
        except ValueError:
            # Label has no corresponding token ID
            token_ids.append(None)
    return tuple(token_ids)


def _collie_graph_hmm_masks(token_vocabulary):
    """Create HMM transition and emission masks for graph-constrained generation.
    
    Transition mask: Prevents any transitions out of EOS state (forces sequence termination).
    Emission mask: Forces one-to-one correspondence between hidden states and emissions.
    
    Args:
        token_vocabulary: Token vocabulary with label and EOS information.
    
    Returns:
        Tuple of (transition_mask, emission_mask) as float tensors.
    """
    label_count = token_vocabulary.label_count
    # Initialize all transitions as allowed
    transition_mask = torch.ones((label_count, label_count), dtype=torch.float32)
    # Block all transitions from EOS state except self-loop
    eos_label = token_vocabulary.eos_label
    transition_mask[eos_label, :] = 0.0
    transition_mask[eos_label, eos_label] = 1.0
    # Emission: each state can only emit its corresponding symbol (identity matrix)
    emission_mask = torch.eye(label_count, dtype=torch.float32)
    return transition_mask, emission_mask


def print_tkns(input_tkns, cutoff_idx, tkns, tokenizer, label_map):
    """Pretty-print instruction tokens and decoded predictions.
    
    Args:
        input_tkns: Full input token sequence.
        cutoff_idx: Index separating instruction from generation.
        tkns: Label indices to decode and display.
        tokenizer: Tokenizer for decoding token IDs.
        label_map: Vocabulary mapper for converting labels to tokens.
    """
    print(
        '\t'.join([tokenizer.decode(x) for x in input_tkns[0,:cutoff_idx]]) + '\t' +
        color('\t'.join([tokenizer.decode(x) for x in label_map.unmap_vocab(tkns)]), fg='red')
    )

def viz_inference(program, sample_data, ILP=False):
    """Run inference on sample data and display predictions vs. ground truth.
    
    Args:
        program: DomiKnowS program to run inference.
        sample_data: Input data with instruction and target tokens.
        ILP: Whether to display ILP constraint-satisfaction predictions.
    """
    preds, labels = [], []  # Greedy predictions and ground truth
    ilp_preds = []  # ILP solver predictions if enabled

    # Populate the program graph with sample data
    node = program.populate_one(sample_data)

    # Extract token-level predictions and ground truth labels
    for token_node in node.getChildDataNodes():
        if ILP:
            ilp_preds.append(torch.argmax(token_node.getAttribute('<generated_token>/ILP')))

        preds.append(torch.argmax(token_node.getAttribute('<generated_token>'), dim=0).item())
        labels.append(token_node.getAttribute('<generated_token>/label').item())

    # Display results with color coding
    print(color('Ground-truth tokens:', fg='green'))
    print_tkns(sample_tkn, cutoff_idx, labels, tokenizer, label_map)

    print(color('Predicted tokens:', fg='green'))
    print_tkns(sample_tkn, cutoff_idx, preds, tokenizer, label_map)

    if args.ILP:
        print(color('ILP predictions:', fg='green'))
        print_tkns(sample_tkn, cutoff_idx, ilp_preds, tokenizer, label_map)

    # Check constraint satisfaction
    constr_names = [
        'no non-EOS tokens can follow an EOS token',
        'at most 4 non-EOS tokens are generated',
    ]

    # Output constraint violations
    print(color('Constraint satisfaction rate:', fg='green', style='bold'))
    verify = node.verifyResultsLC()
    for i, (k, v) in enumerate(verify.items()):
        constraint_name = constr_names[i] if i < len(constr_names) else str(k)
        print(color('Constraint:', fg='green', style='bold'), color(constraint_name, fg='green'))
        print(k, v['satisfied'])


def _generated_probs_from_datanode(node):
    """Stack token-level generated_token probabilities from a populated DataNode.
    
    Extracts logits/probabilities from each token node in sequence order for computing
    generation latent loss terms based on soft constraint violations.
    
    Args:
        node: Populated DataNode from the program graph.
    
    Returns:
        Stacked tensor of shape (num_tokens, num_labels) with probability values.
    
    Raises:
        ValueError: If any token node lacks generated_token probabilities or no tokens found.
    """
    rows = []
    for token_node in node.getChildDataNodes():
        # Try both attribute names for compatibility
        value = token_node.getAttribute('<generated_token>')
        if value is None:
            value = token_node.getAttribute('generated_token')
        if value is None:
            raise ValueError("token DataNode is missing generated_token probabilities")
        rows.append(value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32))
    if not rows:
        raise ValueError("no token DataNodes found for latent loss")
    return torch.stack(rows, dim=0)


if __name__ == "__main__":
    """Main execution: Train a constrained generation program using Primal-Dual learning."""
    from pathlib import Path
    from transformers import AutoTokenizer
    import pickle
    from colors import color
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', type=str, default='data/vocab_val_10k.pkl', help="Path to the vocabulary file, generated from build_vocab.py")
    parser.add_argument('--pad_size', type=int, default=32, help="Maximum length of generation")
    parser.add_argument('--model_mode', type=str, default='generate', choices=['tf', 'generate'], help="tf: Teacher-forcing during the forward pass, generate: Greedy decoding during the forward pass")
    parser.add_argument('--constrained_decoding', default=False, action='store_true', help="Mask generation logits with the DFA compiled from the active constraints")
    parser.add_argument('--graph_hmm_learner', type=str, default='none', choices=['none', 'hmm', 'spectral'], help="Use a PMD-compatible graph_hmm Torch learner instead of the HF TinyModel")
    parser.add_argument('--graph_hmm_source', type=str, default='target', choices=['target', 'generated'], help="For graph_hmm learners, train from target labels or from labels generated by a frozen HF TinyModel teacher")
    parser.add_argument('--graph_hmm_generated_weight', type=float, default=1.0, help="Weight for the graph_hmm imitation loss when --graph_hmm_source generated is used")
    parser.add_argument('--latent_weight', type=float, default=0.0, help="Weight for generation latent soft constraints")
    parser.add_argument('--ILP', default=False, action='store_true', help="Add this flag to enable ILP inference")
    parser.add_argument('--max_vocab_size', type=int, default=None, required=False, help="Maximum size of the vocabulary")
    parser.add_argument('--steps', type=int, default=100, help="Number of PMD training steps")

    args = parser.parse_args()

    # Resolve vocabulary file path with fallback logic
    task_dir = Path(__file__).resolve().parent
    vocab_file = Path(args.vocab_file)
    if not vocab_file.is_absolute() and not vocab_file.exists():
        # Try task-relative path if not absolute
        task_relative = task_dir / vocab_file
        if task_relative.exists():
            vocab_file = task_relative
    if not vocab_file.exists() and Path(args.vocab_file).name == "vocab_val.pkl":
        # Fall back to vocab_val_10k.pkl if requested file doesn't exist
        fallback = task_dir / "data" / "vocab_val_10k.pkl"
        if fallback.exists():
            print(color('Vocabulary file not found:', fg='yellow'), args.vocab_file)
            print(color('Using available fallback:', fg='yellow'), str(fallback))
            vocab_file = fallback
    if not vocab_file.exists():
        available = ", ".join(path.name for path in sorted((task_dir / "data").glob("*.pkl")))
        raise FileNotFoundError(
            f"could not find vocabulary file {args.vocab_file!r}. "
            f"Paths are resolved from the current directory and {task_dir}. "
            f"Available Collie vocab files: {available or 'none'}"
        )

    if args.graph_hmm_learner == 'none' and args.graph_hmm_source != 'target':
        raise ValueError("--graph_hmm_source only applies when --graph_hmm_learner is hmm or spectral")

    # Load tokenizer and optionally model.
    # Note: Pure target-conditioned graph-HMM paths do not need the HuggingFace model.
    print(color('Loading tokenizer', fg='green'))
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    needs_hf_model = args.graph_hmm_learner == 'none' or args.graph_hmm_source == 'generated'
    if needs_hf_model:
        from transformers import AutoModelForCausalLM

        print(color('Loading model', fg='green'))
        model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
    else:
        print(color('Skipping HF model load', fg='green'), f'using graph_hmm {args.graph_hmm_learner} Torch learner')
        model = torch.nn.Identity()

    # Load vocabulary and create token map
    with open(vocab_file, 'rb') as f_in:
        vocab_data = pickle.load(f_in)

        # Ensure EOS token is in the vocabulary
        max_idx = max(vocab_data.values())
        vocab_idx = tokenizer.eos_token_id
        if vocab_idx not in vocab_data:
            vocab_data[vocab_idx] = max_idx + 1

        # Build vocabulary from data with optional max size constraint
        label_map = TokenMap(vocab_data, max_length=args.max_vocab_size)

    # Define the vocabulary tokens to predict (subset of full vocabulary)
    vocab = ['<|endoftext|>', ' The', ' slide']

    print(color('Vocabulary size:', fg='green'), len(label_map))

    # Build the DomiKnowS program with specified configuration
    print(color('Learning path:', fg='green'), 'PrimalDualProgram/cmodel uses DomiKnowS graph constraint loss')
    print(
        color('Enforcement path:', fg='green'),
        'DFA logits masking is ' + ('enabled' if args.constrained_decoding else 'disabled') +
        ' via --constrained_decoding'
    )
    print(
        color('Generation learner:', fg='green'),
        'HF TinyModel' if args.graph_hmm_learner == 'none' else f'graph_hmm {args.graph_hmm_learner} Torch learner from {args.graph_hmm_source} labels'
    )
    # Instantiate the program with all configuration parameters
    program = build_program(
        label_map,
        model,
        tokenizer,
        vocab,
        pad_size=args.pad_size,
        model_mode=args.model_mode,
        ilp=args.ILP,
        constrained_decoding=args.constrained_decoding,
        graph_hmm_learner=args.graph_hmm_learner,
        graph_hmm_source=args.graph_hmm_source,
    )

    # Prepare training sample data
    sample = "At the end, she was happy."
    sample_tkn = tokenize(sample, tokenizer)  # Tokenize the sample text

    print(color('Running inference', fg='green'))
    cutoff_idx = 4  # Instruction tokens up to this index, rest are targets
    sample_data = {
        'target_tokens': sample_tkn[:,cutoff_idx:],  # Tokens to generate
        'instruction_tokens': sample_tkn[:,:cutoff_idx],  # Conditioning context
        '_testing_generated_tokens': sample_tkn[:,cutoff_idx:]  # For validation
    }

    viz_inference(program, sample_data, ILP=args.ILP)

    # Set up optimizers for model and constraint model in Primal-Dual framework
    opt = torch.optim.Adam(program.model.parameters(), lr=1e-3)  # Primal optimizer
    copt = torch.optim.Adam(program.cmodel.parameters(), lr=1e-3)  # Constraint optimizer

    # Main training loop with Primal-Dual Method (PMD)
    for _ in tqdm(range(args.steps), desc="Training with PMD"):
        # Training step: update both model and constraint parameters
        # Zero gradients for both optimizers
        opt.zero_grad()
        copt.zero_grad()
        # Forward pass: get model loss and constraint satisfaction
        mloss, _, *output = program.model(sample_data)
        closs, *_ = program.cmodel(output[1])
        # Compute latent loss from soft constraint violations
        latent_breakdown = program.generation_enforcement.latent_breakdown(
            _generated_probs_from_datanode(output[0]),
            eos_label=program.generation_bundle.vocabulary.eos_label,
        )

        # Compute combined loss: supervised (disabled), constraint, and latent soft constraint losses
        breakdown = compute_generation_training_loss(
            supervised_loss=mloss,
            pmd_loss=closs,
            latent_loss=latent_breakdown.total,
            weights=GenerationLossWeights(supervised=0.0, pmd=1.0, latent=args.latent_weight),
            latent_items=latent_breakdown.items,
        )
        # Add imitation loss if using graph_hmm with generated labels
        graph_hmm_generated_loss = getattr(getattr(program, "graph_hmm_model", None), "last_imitation_loss", None)
        loss = breakdown.total
        if graph_hmm_generated_loss is not None and args.graph_hmm_generated_weight:
            loss = loss + args.graph_hmm_generated_weight * graph_hmm_generated_loss
        # Extract scalar values for logging
        model_loss_value = mloss.detach().item() if torch.is_tensor(mloss) else mloss
        constraint_loss_value = closs.detach().item() if torch.is_tensor(closs) else closs
        graph_hmm_generated_loss_value = (
            graph_hmm_generated_loss.detach().item()
            if torch.is_tensor(graph_hmm_generated_loss)
            else graph_hmm_generated_loss
        )
        # Log loss components for monitoring training progress
        print(
            "Loss",
            loss.item(),
            "(optimized = constraint_loss + latent_weight * latent_loss"
            + (" + graph_hmm_generated_weight * generated_imitation_loss" if graph_hmm_generated_loss is not None else "")
            + "; DomiKnowS model_loss disabled)",
            "model_loss=",
            model_loss_value,
            "constraint_loss=",
            constraint_loss_value,
            "generated_imitation_loss=",
            graph_hmm_generated_loss_value,
            "latent_loss=",
            breakdown.latent.detach().item() if torch.is_tensor(breakdown.latent) else breakdown.latent,
            "latent_terms=",
            len(latent_breakdown.items),
        )
        # Safety check for invalid loss values
        if loss.item() < 0:
            print("Negative loss", loss.item())
            break
        # Backward pass and parameter updates
        if loss:
            loss.backward()
            opt.step()  # Update model parameters
            copt.step()  # Update constraint model parameters

        # Display predictions after each update
        viz_inference(program, sample_data, ILP=args.ILP)
