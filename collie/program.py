from domiknows.sensor.pytorch.sensors import ReaderSensor, JointSensor, FunctionalSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program import LearningBasedProgram, SolverPOIProgram

from transformers import PreTrainedModel, PreTrainedTokenizer
from tokens import TokenMap, tokenize
import torch

from graph import build_graph
from model import TinyModel


def build_program(
        label_map: TokenMap,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer
    ) -> LearningBasedProgram:

    graph, (text, token, contains, generated_token) = build_graph(label_map, tokenizer)

    text["instruction_tokens"] = ReaderSensor(keyword="instruction_tokens")
    text["target_tokens"] = ReaderSensor(keyword="target_tokens")

    text["_testing_generated_tokens"] = ReaderSensor(keyword="_testing_generated_tokens")

    def _add_sequence(target, testing_seq):
        return torch.ones((target.shape[1], 1)), target[0], testing_seq[0]

    token[contains, 'target', '_testing_generated'] = JointSensor(text["target_tokens"], text["_testing_generated_tokens"], forward=_add_sequence)
    token[generated_token] = FunctionalSensor(token[contains], "target", forward=lambda _, x: label_map.map_vocab(x), label=True)

    token[generated_token] = ModuleLearner(
        token[contains],
        text["instruction_tokens"],
        'target',
        '_testing_generated',
        module=TinyModel(model, tokenizer, label_map)
    )

    return SolverPOIProgram(
        graph,
        poi=(text, token),
        inferTypes=['local/argmax']
    )


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import pickle
    from colors import color

    # load model
    with open('data/vocab_val_1k.pkl', 'rb') as f_in:
        label_map = TokenMap(pickle.load(f_in))

    print(color('Vocabulary size:', fg='green'), len(label_map))

    print(color('Loading model', fg='green'))
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")

    # build program
    program = build_program(label_map, model, tokenizer)

    # forward pass
    sample = "One day, a little girl named Lily found an apple in her room."    
    sample_tkn = tokenize(sample, tokenizer)

    print(color('Running inference', fg='green'))
    node = program.populate_one({
        'target_tokens': sample_tkn[:,2:],
        'instruction_tokens': sample_tkn[:,:2],
        '_testing_generated_tokens': sample_tkn[:,2:]
    })

    # output predictions
    preds, labels = [], []

    for token_node in node.getChildDataNodes():
        preds.append(torch.argmax(token_node.getAttribute('<generated_token>'), dim=0).item())
        labels.append(token_node.getAttribute('<generated_token>/label').item())

    print(color('Ground-truth tokens:', fg='green'))
    print(
        '\t'.join([tokenizer.decode(x) for x in sample_tkn[0,:2]]) + '\t' +
        color('\t'.join([tokenizer.decode(x) for x in label_map.unmap_vocab(labels)]), fg='red')
    )

    print(color('Predicted tokens:', fg='green'))
    print(
        '\t'.join([tokenizer.decode(x) for x in sample_tkn[0,:2]]) + '\t' +
        color('\t'.join([tokenizer.decode(x) for x in label_map.unmap_vocab(preds)]), fg='red')
    )

    constr_names = [
        'at most three tokens are generated',
        'at most 20 tokens are generated',
        'at least one of the " The" token is generated',
        'at least one of the " girl" token is generated',
        'if there is a token " The", then there are at most three tokens generated total'
    ]

    # output constraint violations
    print(color('Constraint satisfaction rate:', fg='green'))
    verify = node.verifyResultsLC()
    for i, (k, v) in enumerate(verify.items()):
        print(color(constr_names[i], fg='green'))
        print(k, v['satisfied'])
