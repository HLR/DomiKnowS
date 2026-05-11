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

from graph import build_generation_bundle
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
    ) -> LearningBasedProgram:

    vocab_ids = [tokenizer.encode(v)[0] for v in vocab]

    graph, bundle = build_generation_bundle(tokenizer, vocab)
    text = bundle.text
    token = bundle.token
    contains = bundle.contains
    generated_token = bundle.generated_token
    is_before_rel = bundle.is_before_rel
    first_token = bundle.first_token
    second_token = bundle.second_token
    token_vocabulary = bundle.vocabulary
    constrained_dfa = (
        constraints_to_dfa_from_graph(graph, bundle)
        if constrained_decoding
        else None
    )
    enforcement = discover_generation_enforcement(graph, bundle, on_unsupported="ignore")

    text["instruction_tokens"] = ReaderSensor(keyword="instruction_tokens")
    text["target_tokens"] = ReaderSensor(keyword="target_tokens")

    text["_testing_generated_tokens"] = ReaderSensor(keyword="_testing_generated_tokens")

    def _add_sequence(target, testing_seq):
        assert len(target[0]) <= pad_size, f"target sequence is too long: {len(target[0])}"
        assert len(testing_seq[0]) <= pad_size, f"debug sequence is too long: {len(testing_seq[0])}"

        # expect target, testing_seq to be (1, seq_length)
        target_out = torch.cat([
            target[0],
            torch.ones((pad_size - len(target[0]),)) * tokenizer.eos_token_id
        ], dim=0)

        testing_out = torch.cat([
            testing_seq[0],
            torch.ones((pad_size - len(testing_seq[0]),)) * tokenizer.eos_token_id
        ], dim=0)

        target_labels = map_labels(target_out.long())

        return torch.ones((pad_size, 1)), target_out, testing_out, target_labels, torch.arange(pad_size)

    token[contains, 'target', '_testing_generated', 'target_labels', 'token_index'] = JointSensor(text["target_tokens"], text["_testing_generated_tokens"], forward=_add_sequence)

    # issue: can't have loss over "other" tokens

    def map_labels(label_vals):
        new_labels = []

        for label in label_vals:
            if label in vocab_ids:
                new_labels.append(vocab_ids.index(label))
            else:
                new_labels.append(len(vocab_ids))
        
        return torch.tensor(new_labels)

    token[generated_token] = FunctionalSensor(token[contains], "target", forward=lambda _, x: map_labels(x), label=True)

    print(vocab)
    if graph_hmm_learner == 'none':
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

        token[generated_token] = ModuleLearner(
            token[contains],
            text["instruction_tokens"],
            'target',
            '_testing_generated',
            module=model
        )
    else:
        transition_mask, emission_mask = _collie_graph_hmm_masks(token_vocabulary)
        label_to_token_id = _label_token_id_map(token_vocabulary)
        if graph_hmm_learner == 'hmm':
            model = GraphHMMGenerationHead(
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
            model = GraphSpectralGenerationHead(
                label_count=token_vocabulary.label_count,
                state_count=token_vocabulary.label_count,
                symbols=tuple(range(token_vocabulary.label_count)),
                pad_size=pad_size,
                label_to_token_id=label_to_token_id,
                trainable=True,
            )
        else:
            raise ValueError("graph_hmm_learner must be 'none', 'hmm', or 'spectral'")

        token[generated_token] = ModuleLearner(
            token[contains],
            text["instruction_tokens"],
            'target_labels',
            module=model,
        )

    # edge sensors
    def is_before_edges(*args, arg1, arg2):
        # print('is_before_edges', arg1.getAttribute('token_index'), arg2.getAttribute('token_index'))
        return arg1.getAttribute('token_index') < arg2.getAttribute('token_index')
    
    is_before_rel[first_token.reversed, second_token.reversed] = CompositionCandidateSensor(
        relations=(first_token.reversed, second_token.reversed),
        forward=is_before_edges
    )

    # return SolverPOIProgram(
    #     graph,
    #     poi=(text, token, is_before_rel),
    #     inferTypes=['local/argmax', 'ILP'] if ilp else ['local/argmax']
    # )

    program = PrimalDualProgram(
        graph,
        SolverModel,
        poi=(text, token, is_before_rel),
        inferTypes=['local/argmax', 'ILP'] if ilp else ['local/argmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        beta=10, device='cpu', tnorm="P", counting_tnorm="P"
    )
    program.generation_bundle = bundle
    program.generation_enforcement = enforcement
    program.graph_hmm_learner = graph_hmm_learner
    return program


def _label_token_id_map(token_vocabulary):
    token_ids = []
    for label in range(token_vocabulary.label_count):
        try:
            token_ids.append(token_vocabulary.token_id_for_label(label))
        except ValueError:
            token_ids.append(None)
    return tuple(token_ids)


def _collie_graph_hmm_masks(token_vocabulary):
    label_count = token_vocabulary.label_count
    transition_mask = torch.ones((label_count, label_count), dtype=torch.float32)
    eos_label = token_vocabulary.eos_label
    transition_mask[eos_label, :] = 0.0
    transition_mask[eos_label, eos_label] = 1.0
    emission_mask = torch.eye(label_count, dtype=torch.float32)
    return transition_mask, emission_mask


def print_tkns(input_tkns, cutoff_idx, tkns, tokenizer, label_map):
    print(
        '\t'.join([tokenizer.decode(x) for x in input_tkns[0,:cutoff_idx]]) + '\t' +
        color('\t'.join([tokenizer.decode(x) for x in label_map.unmap_vocab(tkns)]), fg='red')
    )

def viz_inference(program, sample_data, ILP=False):
    preds, labels = [], []
    ilp_preds = []

    node = program.populate_one(sample_data)

    for token_node in node.getChildDataNodes():
        if ILP:
            ilp_preds.append(torch.argmax(token_node.getAttribute('<generated_token>/ILP')))

        preds.append(torch.argmax(token_node.getAttribute('<generated_token>'), dim=0).item())
        labels.append(token_node.getAttribute('<generated_token>/label').item())

    print(color('Ground-truth tokens:', fg='green'))
    print_tkns(sample_tkn, cutoff_idx, labels, tokenizer, label_map)

    print(color('Predicted tokens:', fg='green'))
    print_tkns(sample_tkn, cutoff_idx, preds, tokenizer, label_map)

    if args.ILP:
        print(color('ILP predictions:', fg='green'))
        print_tkns(sample_tkn, cutoff_idx, ilp_preds, tokenizer, label_map)

    constr_names = [
        'no non-EOS tokens can follow an EOS token',
        'at most 4 non-EOS tokens are generated',
    ]

    # output constraint violations
    print(color('Constraint satisfaction rate:', fg='green', style='bold'))
    verify = node.verifyResultsLC()
    for i, (k, v) in enumerate(verify.items()):
        constraint_name = constr_names[i] if i < len(constr_names) else str(k)
        print(color('Constraint:', fg='green', style='bold'), color(constraint_name, fg='green'))
        print(k, v['satisfied'])


def _generated_probs_from_datanode(node):
    """Stack token-level generated_token probabilities from a populated DataNode."""
    rows = []
    for token_node in node.getChildDataNodes():
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
    from pathlib import Path
    from transformers import AutoTokenizer
    import pickle
    from colors import color
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', type=str, default='data/vocab_val_10k.pkl', help="Path to the vocabulary file, generated from build_vocab.py")
    parser.add_argument('--pad_size', type=int, default=32, help="Maximum length of generation")
    parser.add_argument('--model_mode', type=str, default='generate', choices=['tf', 'generate'], help="tf: Teacher-forcing during the forward pass, generate: Greedy decoding during the forward pass")
    parser.add_argument('--constrained_decoding', default=False, action='store_true', help="Mask generation logits with the DFA compiled from the active constraints")
    parser.add_argument('--graph_hmm_learner', type=str, default='none', choices=['none', 'hmm', 'spectral'], help="Use a PMD-compatible graph_hmm Torch learner instead of the HF TinyModel")
    parser.add_argument('--latent_weight', type=float, default=0.0, help="Weight for generation latent soft constraints")
    parser.add_argument('--ILP', default=False, action='store_true', help="Add this flag to enable ILP inference")
    parser.add_argument('--max_vocab_size', type=int, default=None, required=False, help="Maximum size of the vocabulary")
    parser.add_argument('--steps', type=int, default=100, help="Number of PMD training steps")

    args = parser.parse_args()

    task_dir = Path(__file__).resolve().parent
    vocab_file = Path(args.vocab_file)
    if not vocab_file.is_absolute() and not vocab_file.exists():
        task_relative = task_dir / vocab_file
        if task_relative.exists():
            vocab_file = task_relative
    if not vocab_file.exists() and Path(args.vocab_file).name == "vocab_val.pkl":
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

    # load tokenizer / model. Graph-HMM learner paths do not need the HF model.
    print(color('Loading tokenizer', fg='green'))
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    if args.graph_hmm_learner == 'none':
        from transformers import AutoModelForCausalLM

        print(color('Loading model', fg='green'))
        model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
    else:
        print(color('Skipping HF model load', fg='green'), f'using graph_hmm {args.graph_hmm_learner} Torch learner')
        model = torch.nn.Identity()

    with open(vocab_file, 'rb') as f_in:
        vocab_data = pickle.load(f_in)

        # add eos token to the vocabulary
        max_idx = max(vocab_data.values())
        vocab_idx = tokenizer.eos_token_id
        if vocab_idx not in vocab_data:
            vocab_data[vocab_idx] = max_idx + 1

        # build vocabulary from data
        label_map = TokenMap(vocab_data, max_length=args.max_vocab_size)

    vocab = ['<|endoftext|>', ' The', ' slide']

    print(color('Vocabulary size:', fg='green'), len(label_map))

    # build program
    print(color('Learning path:', fg='green'), 'PrimalDualProgram/cmodel uses DomiKnowS graph constraint loss')
    print(
        color('Enforcement path:', fg='green'),
        'DFA logits masking is ' + ('enabled' if args.constrained_decoding else 'disabled') +
        ' via --constrained_decoding'
    )
    print(
        color('Generation learner:', fg='green'),
        'HF TinyModel' if args.graph_hmm_learner == 'none' else f'graph_hmm {args.graph_hmm_learner} Torch learner'
    )
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
    )

    # train
    sample = "At the end, she was happy."    
    sample_tkn = tokenize(sample, tokenizer)

    print(color('Running inference', fg='green'))
    cutoff_idx = 4
    sample_data = {
        'target_tokens': sample_tkn[:,cutoff_idx:],
        'instruction_tokens': sample_tkn[:,:cutoff_idx],
        '_testing_generated_tokens': sample_tkn[:,cutoff_idx:]
    }

    viz_inference(program, sample_data, ILP=args.ILP)

    opt = torch.optim.Adam(program.model.parameters(), lr=1e-3)
    copt = torch.optim.Adam(program.cmodel.parameters(), lr=1e-3)

    for _ in tqdm(range(args.steps), desc="Training with PMD"):
        # train step
        opt.zero_grad()
        copt.zero_grad()
        mloss, _, *output = program.model(sample_data)
        closs, *_ = program.cmodel(output[1])
        latent_breakdown = program.generation_enforcement.latent_breakdown(
            _generated_probs_from_datanode(output[0]),
            eos_label=program.generation_bundle.vocabulary.eos_label,
        )

        breakdown = compute_generation_training_loss(
            supervised_loss=mloss,
            pmd_loss=closs,
            latent_loss=latent_breakdown.total,
            weights=GenerationLossWeights(supervised=0.0, pmd=1.0, latent=args.latent_weight),
            latent_items=latent_breakdown.items,
        )
        loss = breakdown.total
        model_loss_value = mloss.detach().item() if torch.is_tensor(mloss) else mloss
        constraint_loss_value = closs.detach().item() if torch.is_tensor(closs) else closs
        print(
            "Loss",
            loss.item(),
            "(optimized = constraint_loss + latent_weight * latent_loss; model_loss disabled)",
            "model_loss=",
            model_loss_value,
            "constraint_loss=",
            constraint_loss_value,
            "latent_loss=",
            breakdown.latent.detach().item() if torch.is_tensor(breakdown.latent) else breakdown.latent,
            "latent_terms=",
            len(latent_breakdown.items),
        )
        if loss.item() < 0:
            print("Negative loss", loss.item())
            break
        if loss:
            loss.backward()
            opt.step()
            copt.step()

        # output predictions
        viz_inference(program, sample_data, ILP=args.ILP)
