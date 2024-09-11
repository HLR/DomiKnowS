from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program import LearningBasedProgram, SolverPOIProgram

from transformers import PreTrainedModel, PreTrainedTokenizer
from tokens import TokenMap, tokenize
import torch
from typing import Literal

from graph import build_graph
from model import TinyModel


def build_program(
        label_map: TokenMap,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        pad_size: int = 32,
        model_mode: Literal['tf', 'generate'] = 'generate',
        ilp: bool = False
    ) -> LearningBasedProgram:

    graph, (text, token, contains, generated_token, is_before_rel, first_token, second_token) = build_graph(label_map, tokenizer)

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

        return torch.ones((pad_size, 1)), target_out, testing_out, torch.arange(pad_size)

    token[contains, 'target', '_testing_generated', 'token_index'] = JointSensor(text["target_tokens"], text["_testing_generated_tokens"], forward=_add_sequence)
    token[generated_token] = FunctionalSensor(token[contains], "target", forward=lambda _, x: label_map.map_vocab(x), label=True)

    model = TinyModel(
        model,
        tokenizer,
        label_map,
        eos_idx=tokenizer.eos_token_id,
        pad_size=pad_size,
        mode=model_mode
    )

    token[generated_token] = ModuleLearner(
        token[contains],
        text["instruction_tokens"],
        'target',
        '_testing_generated',
        module=model
    )

    # edge sensors
    def is_before_edges(*args, arg1, arg2):
        # print('is_before_edges', arg1.getAttribute('token_index'), arg2.getAttribute('token_index'))
        return arg1.getAttribute('token_index') < arg2.getAttribute('token_index')
    
    is_before_rel[first_token.reversed, second_token.reversed] = CompositionCandidateSensor(
        relations=(first_token.reversed, second_token.reversed),
        forward=is_before_edges
    )

    return SolverPOIProgram(
        graph,
        poi=(text, token, is_before_rel),
        inferTypes=['local/argmax', 'ILP'] if ilp else ['local/argmax']
    )


def print_tkns(input_tkns, cutoff_idx, tkns, tokenizer, label_map):
    print(
        '\t'.join([tokenizer.decode(x) for x in input_tkns[0,:cutoff_idx]]) + '\t' +
        color('\t'.join([tokenizer.decode(x) for x in label_map.unmap_vocab(tkns)]), fg='red')
    )


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import pickle
    from colors import color
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', type=str, default='vocab_val.pkl', help="Path to the vocabulary file, generated from build_vocab.py")
    parser.add_argument('--pad_size', type=int, default=32, help="Maximum length of generation")
    parser.add_argument('--model_mode', type=str, default='generate', choices=['tf', 'generate'], help="tf: Teacher-forcing during the forward pass, generate: Greedy decoding during the forward pass")
    parser.add_argument('--ILP', default=False, action='store_true', help="Add this flag to enable ILP inference")
    parser.add_argument('--max_vocab_size', type=int, default=None, required=False, help="Maximum size of the vocabulary")

    args = parser.parse_args()

    # load model
    print(color('Loading model', fg='green'))
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")

    with open(args.vocab_file, 'rb') as f_in:
        vocab_data = pickle.load(f_in)

        # add eos token to the vocabulary
        max_idx = max(vocab_data.values())
        vocab_idx = tokenizer.eos_token_id
        if vocab_idx not in vocab_data:
            vocab_data[vocab_idx] = max_idx + 1

        # build vocabulary from data
        label_map = TokenMap(vocab_data, max_length=args.max_vocab_size)

    print(color('Vocabulary size:', fg='green'), len(label_map))

    # build program
    program = build_program(
        label_map,
        model,
        tokenizer,
        pad_size=args.pad_size,
        model_mode=args.model_mode,
        ilp=args.ILP
    )

    # forward pass
    sample = "At the end, she was happy."    
    sample_tkn = tokenize(sample, tokenizer)

    print(color('Running inference', fg='green'))
    cutoff_idx = 4
    node = program.populate_one({
        'target_tokens': sample_tkn[:,cutoff_idx:],
        'instruction_tokens': sample_tkn[:,:cutoff_idx],
        '_testing_generated_tokens': sample_tkn[:,cutoff_idx:]
    })

    # output predictions
    preds, labels = [], []
    ilp_preds = []

    for token_node in node.getChildDataNodes():
        if args.ILP:
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
        'at most 16 tokens are generated',
        'at most 32 tokens are generated',
        'at least one of the " The" token is generated',
        'at least one of the " slide" token is generated',
        'if there is a token " The", then there are at most 16 tokens generated total'
    ]

    # output constraint violations
    print(color('Constraint satisfaction rate:', fg='green', style='bold'))
    verify = node.verifyResultsLC()
    for i, (k, v) in enumerate(verify.items()):
        print(color('Constraint:', fg='green', style='bold'), color(constr_names[i], fg='green'))
        print(k, v['satisfied'])

