import sys
import importlib.util
import torch
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent


def _load_local_module(module_name: str, filename: str):
    """Load an example module without colliding with another test directory."""
    qualified_name = f"domiknows_hard_example_{module_name}"
    existing = sys.modules.get(qualified_name)
    if existing is not None:
        return existing

    module_path = RUN_DIR / filename
    spec = importlib.util.spec_from_file_location(qualified_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name!r} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def _local_graph_module():
    return _load_local_module("graph", "graph.py")


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "domiknows").exists():
            return candidate
    return start


REPO_ROOT = _find_repo_root(RUN_DIR)
for path in (RUN_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


import argparse
from domiknows.program import POIProgram, SolverPOIProgram, IMLProgram, CallbackProgram
from domiknows.program.callbackprogram import ProgramStorageCallback
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.lossprogram import PrimalDualProgram, InferenceProgram
from domiknows.program.model.pytorch import SolverModel, SolverModelDictLoss
from domiknows.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss, NBCrossEntropyDictLoss
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ModuleSensor, ReaderSensor, \
    FunctionalReaderSensor, TorchSensor, cache, TorchCache
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor, EdgeSensor, \
    CompositionCandidateReaderSensor
import re
from reader import conll4_reader
from reward import reward_from_generator
from domiknows.program import ReinforcementProgram
import numpy as np

import spacy

# from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm')  # English()

import logging
import os

# The per-tensor debug_tensor(...) calls log at DEBUG on every sensor/BERT/
# Classifier forward, which floods the console each training step. Default to
# WARNING; set DOMIKNOWS_HARD_DEBUG=1 to restore the verbose DEBUG output.
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DOMIKNOWS_HARD_DEBUG") == "1" else logging.WARNING)

from transformers import BertTokenizerFast, BertModel

TRANSFORMER_MODEL = 'bert-base-uncased'

FEATURE_DIM = 768 + 96


def find_data_file(filename, train_portion=None):
    """Find data file by checking multiple possible locations"""
    current_dir = Path(__file__).parent

    # First, check if extracted portion file exists
    if train_portion:
        extracted_filename = f"{train_portion}.json"
        possible_extracted_paths = [
            current_dir / extracted_filename,
            current_dir / "data" / extracted_filename,
            Path.cwd() / extracted_filename,
            Path.cwd() / "data" / extracted_filename,
        ]
        
        for path in possible_extracted_paths:
            if path.exists():
                print(f"Using extracted data file: {path}")
                return str(path)

    # List of possible locations to check for main file
    possible_paths = [
        current_dir / filename,
        current_dir / "data" / filename,
        current_dir / ".." / filename,
        current_dir / ".." / "data" / filename,
        current_dir / ".." / ".." / filename,
        current_dir / ".." / ".." / "data" / filename,
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    raise FileNotFoundError(f"Could not find {filename} in any of the expected locations: {possible_paths}")


class Tokenizer():
    def __init__(self, device='cpu') -> None:
        self.tokenizer = BertTokenizerFast.from_pretrained(TRANSFORMER_MODEL)
        self.device = device

    def __call__(self, text):
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(text, padding=True, return_tensors='pt', return_offsets_mapping=True)

        ids = tokens['input_ids'].to(self.device)
        mask = tokens['attention_mask'].to(self.device)
        offset = tokens['offset_mapping'].to(self.device)

        idx = mask.nonzero()[:, 0].unsqueeze(-1)
        mapping = torch.zeros(idx.shape[0], idx.max() + 1, device=self.device)
        mapping.scatter_(1, idx, 1)

        mask = mask.bool()
        ids = ids.masked_select(mask)
        offset = torch.stack((offset[:, :, 0].masked_select(mask), offset[:, :, 1].masked_select(mask)), dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return mapping, ids, offset, tokens


class BERT(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.module = BertModel.from_pretrained(TRANSFORMER_MODEL)
        self.device = device
        self.module.to(self.device)
        for param in self.module.base_model.parameters():
            param.requires_grad = False

    def forward(self, input):  # <-- This was incorrectly indented inside __init__
        debug_tensor("BERT input", input)
        if input.device != self.device:
            input = input.to(self.device)

        input = input.unsqueeze(0)
        _out = self.module(input)

        out, *_ = _out

        if (isinstance(out, str)):
            out = _out.last_hidden_state

        assert out.shape[0] == 1
        out = out.squeeze(0)
        debug_tensor("BERT output", out)
        return out


class Classifier(torch.nn.Sequential):
    def __init__(self, in_features, device='cpu') -> None:
        linear = torch.nn.Linear(in_features, 2)
        super().__init__(linear)
        self.to(device)
        logging.debug(f"[DTYPE DEBUG] Classifier initialized: weight dtype={linear.weight.dtype}")

    def forward(self, x):
        debug_tensor("Classifier input", x)
        result = super().forward(x)
        debug_tensor("Classifier output", result)
        return result


def debug_tensor(name, tensor):
    """Log tensor dtype, shape, and device for debugging."""
    if isinstance(tensor, torch.Tensor):
        logging.debug(f"[DTYPE DEBUG] {name}: dtype={tensor.dtype}, shape={tensor.shape}, device={tensor.device}")
    else:
        logging.debug(f"[DTYPE DEBUG] {name}: type={type(tensor)}")
    return tensor

# Asking function (from reader.ASKING_TYPE) -> comparison operator the answer uses.
_ASK_OP = {
    "atLeastAL": ">=", "atLeastL": ">=",
    "atMostAL": "<=", "atMostL": "<=",
    "exactAL": "==", "exactL": "==",
    "sumL": "==",
}

# Relation concept names (graph uses 'orgbase_in' for the orgbase_on variable).
_RELATION_NAMES = {"work_for", "located_in", "live_in", "orgbase_in", "orgbase_on", "kill"}


def _asking_op(logic_str):
    """Leading asking function in a logic_str, e.g. 'atMostAL(...)' -> '<='."""
    m = re.match(r"\s*([A-Za-z]+)\s*\(", logic_str or "")
    return _ASK_OP.get(m.group(1)) if m else None


def _asked_number(data_item, logic_str):
    """The threshold N the question asks about (label count, or trailing int)."""
    if isinstance(data_item, dict):
        ll = data_item.get("logic_label")
        if isinstance(ll, (list, tuple)) and ll:
            try:
                return int(ll[0])
            except (TypeError, ValueError):
                pass
        if isinstance(ll, (int, float)):
            return int(ll)
    nums = re.findall(r"-?\d+", logic_str or "")
    return int(nums[-1]) if nums else 1


def yesno_answer_decoder(samples, targets, datanode, data_item):
    """Map one sampled NER/relation decoding to a yes/no answer (threshold-aware).

    Counts how many asked instances the decoding turns on, then answers the
    *actual* asked question by comparing that count to the asked threshold N via
    the asking type parsed from ``logic_str`` (``atLeast`` -> ``count >= N``,
    ``atMost`` -> ``count <= N``, ``exact``/``sum`` -> ``count == N``).  This makes
    the answer depend on getting the count right, which gives the reward a real
    learning signal (unlike a trivial "yes if any positive" rule, which a random
    model already satisfies).  Falls back to ``count >= 1`` if the asking type is
    unknown.
    """
    logic_str = ""
    if isinstance(data_item, dict):
        logic_str = data_item.get("logic_str") or ""

    asked = [
        c for c in targets
        if re.search(rf"(?<!\w){re.escape(c.name)}(?!\w)", logic_str)
    ]
    if not asked:
        asked = list(targets)

    # Count one scale-appropriate group so the count is comparable to N: prefer
    # the asked relation (the head of the andL pattern), else the asked entity.
    # Summing across all asked concepts would inflate the count far beyond N and
    # make the threshold comparison unsatisfiable.
    asked_rel = [c for c in asked if c.name in _RELATION_NAMES]
    counting = asked_rel if asked_rel else asked

    count = 0
    for concept in counting:
        idx = samples.get(concept)
        if idx is not None:
            # class index 1 == "the concept holds" for binary concepts
            count += int((idx.reshape(-1) == 1).sum().item())

    op = _asking_op(logic_str)
    n = _asked_number(data_item, logic_str)
    if op == ">=":
        answer = count >= n
    elif op == "<=":
        answer = count <= n
    elif op == "==":
        answer = count == n
    else:
        answer = count >= 1
    return "yes" if answer else "no"


def program_declaration(train, args, device='auto'):
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o
    from graph import work_for, located_in, live_in, orgbase_on, kill
    from graph import rel_sentence_contains_word, rel_phrase_contains_word, rel_pair_phrase1, rel_pair_phrase2, \
        rel_sentence_contains_phrase

    graph.detach()
    
    # Set device for all Sensors
    TorchSensor.set_default_device(device)
    
    phrase['text'] = ReaderSensor(keyword='tokens')

    def word2vec(text):
        texts = list(map(lambda x: ' '.join(x.split('/')), text))
        tokens_list = list(nlp.pipe(texts))
        result = torch.tensor(np.array([tokens.vector for tokens in tokens_list]), device=device)
        debug_tensor("word2vec output", result)
        return result
        
    phrase['w2v'] = FunctionalSensor('text', forward=word2vec)

    def merge_phrase(phrase_text):
        ones = torch.ones((1, len(phrase_text)), device=device)  # was torch.device
        debug_tensor("merge_phrase ones", ones)
        return [' '.join(phrase_text)], ones

    # NOTE: defined before its use in the EdgeSensor below (was previously
    # referenced before definition, which raised NameError at declaration time).
    def match_phrase(phrase, word_offset):
        def overlap(a_s, a_e, b_s, b_e):
            return (a_s <= b_s and b_s <= a_e) or (a_s <= b_e and b_e <= a_e)

        ph_offset = 0
        ph_word_overlap = []
        for ph in phrase:
            ph_len = len(ph)
            word_overlap = []
            for word_s, word_e in word_offset:
                if word_e - word_s <= 0:
                    word_overlap.append(False)
                else:
                    word_overlap.append(overlap(ph_offset, ph_offset + ph_len, word_s, word_e))
            ph_word_overlap.append(word_overlap)
            ph_offset += ph_len + 1
        result = torch.tensor(ph_word_overlap, device=device)
        debug_tensor("match_phrase output", result)
        return result

    sentence['text', rel_sentence_contains_phrase.reversed] = JointSensor(phrase['text'], forward=merge_phrase)

    # Create Tokenizer with device parameter. This must be declared before the
    # phrase->word EdgeSensor below, which reads word['offset'].
    tokenizer = Tokenizer(device=device)
    word[rel_sentence_contains_word, 'ids', 'offset', 'text'] = JointSensor(sentence['text'], forward=tokenizer)

    phrase[rel_phrase_contains_word.reversed] = EdgeSensor(
        phrase['text'], word['offset'],
        relation=rel_phrase_contains_word.reversed,
        forward=match_phrase
    )

    # Create BERT with device parameter
    bert_model = BERT(device=device)
    word['bert'] = ModuleSensor('ids', module=bert_model)

    def phrase_bert(bert):
        debug_tensor("phrase_bert input", bert)
        return bert

    phrase['bert'] = FunctionalSensor(rel_phrase_contains_word.reversed(word['bert']), forward=phrase_bert)
    
    def concat_features(bert, w2v):
        debug_tensor("concat_features bert input", bert)
        debug_tensor("concat_features w2v input", w2v)
        result = torch.cat((bert, w2v), dim=-1)
        debug_tensor("concat_features output", result)
        return result
    
    phrase['emb'] = FunctionalSensor('bert', 'w2v', forward=concat_features)

    phrase[people] = ModuleLearner('emb', module=Classifier(FEATURE_DIM, device=device))
    phrase[organization] = ModuleLearner('emb', module=Classifier(FEATURE_DIM, device=device))
    phrase[location] = ModuleLearner('emb', module=Classifier(FEATURE_DIM, device=device))
    phrase[other] = ModuleLearner('emb', module=Classifier(FEATURE_DIM, device=device))
    phrase[o] = ModuleLearner('emb', module=Classifier(FEATURE_DIM, device=device))
    
    def filter_pairs(phrase_text, arg1, arg2, data):
        # `data` is the reader's 'relation' list: (rel_text, head_idx, tail_idx)
        # tuples. Create a pair candidate for each true (head, tail) relation.
        for rel in data or []:
            try:
                head_idx, tail_idx = rel[1], rel[2]
            except (TypeError, IndexError, KeyError):
                continue
            if arg1.instanceID == head_idx and arg2.instanceID == tail_idx:
                return True
        return False

    pair[rel_pair_phrase1.reversed, rel_pair_phrase2.reversed] = CompositionCandidateReaderSensor(
        phrase['text'],
        relations=(rel_pair_phrase1.reversed, rel_pair_phrase2.reversed),
        keyword='relation',
        forward=filter_pairs)
    pair['emb'] = FunctionalSensor(
        rel_pair_phrase1.reversed('emb'), rel_pair_phrase2.reversed('emb'),
        forward=lambda arg1, arg2: torch.cat((arg1, arg2), dim=-1))

    pair[work_for] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    pair[located_in] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    pair[live_in] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    pair[orgbase_on] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))
    pair[kill] = ModuleLearner('emb', module=Classifier(FEATURE_DIM * 2))

    # The decision variables sampled by the reinforcement program: the entity
    # labels on phrases and the relation labels on pairs.
    targets = [people, organization, location, other, o,
               work_for, located_in, live_in, orgbase_on, kill]

    # Each data item carries its own reward function under the 'reward_function'
    # key (see reader.conll4_reader / reward.make_reward_function). The decoder
    # turns a sampled decoding into the yes/no answer the reward scores.
    program = ReinforcementProgram(
        graph,
        targets=targets,
        reward_key='reward_function',
        decoder=yesno_answer_decoder,
        num_samples=getattr(args, 'num_samples', 8),
        estimator=getattr(args, 'estimator', 'importance_weighted'),
        device=device,
        visualize=getattr(args, 'visualize', False),
        visualize_port=getattr(args, 'port', 5000),
    )
    return program, train

    


def _print_reward_summary(args, baseline, trained):
    """Explain the before/after mean reward of the reinforcement run."""
    delta = trained - baseline
    print("\n" + "=" * 64)
    print("Reinforcement training summary (hard example)")
    print("-" * 64)
    print("Reward = fraction of sampled decodings whose decoded yes/no answer")
    print("matches the question label. The decoder counts how many asked")
    print("entity/relation instances a decoding turns on, then answers the asked")
    print("question (atLeast/atMost/exact N) by comparing that count to N.")
    print("-" * 64)
    print(f"  estimator         : {getattr(args, 'estimator', 'importance_weighted')}")
    print(f"  epochs            : {args.epochs}")
    print(f"  samples / step    : {getattr(args, 'num_samples', 8)}")
    print(f"  mean reward before: {baseline:.4f}   (random/initial model)")
    print(f"  mean reward after : {trained:.4f}   (after training)")
    print(f"  improvement       : {delta:+.4f}")
    if delta > 1e-3:
        print("  => training INCREASED the reward: the model learned to produce")
        print("     decodings whose counts better satisfy the asked constraints.")
    elif abs(delta) <= 1e-3:
        print("  => reward ~unchanged: near the ceiling for this subset, or the run")
        print("     is too short (raise --epochs / --num_samples / --train_size).")
    else:
        print("  => reward DECREASED: likely sampling noise or too-high --lr;")
        print("     try more --num_samples or a smaller --lr.")
    print("=" * 64)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Getting the arguments passed")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (the trainable params are the small relation/entity "
                             "classifier heads; BERT is frozen, so ~1e-3 is appropriate)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--load_previous", action='store_true')
    parser.add_argument("--train_size", type=int, default=-1, help="Number of training sample")
    parser.add_argument("--train_portion", type=str, default="entities_with_relation", help="Training subset")
    parser.add_argument("--previous_portion", type=str, default="entities_only_with_1_things_YN", help="Training subset")
    parser.add_argument("--checked_acc", type=float, default=0, help="Accuracy to test")
    parser.add_argument("--counting_tnorm", choices=["G", "P", "L", "SP"], default="G", help="The tnorm method to use for the counting constraints")
    parser.add_argument("--data_path", type=str, default="data2.json", help="Path to data file (can be relative or absolute)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for computation (e.g., 'cuda', 'cpu', 'cuda:0', 'auto')")
    parser.add_argument("--num_samples", type=int, default=8, help="Decodings sampled per reinforcement step")
    parser.add_argument("--estimator", type=str, default="importance_weighted", choices=["importance_weighted", "reinforce"], help="Reward-loss estimator")
    parser.add_argument("--visualize", action='store_true', help="Launch the Flask step-by-step visualizer (training pauses each step)")
    parser.add_argument("--port", type=int, default=5000, help="Visualizer port")
    args = parser.parse_args()

    return args


def main(args):
    graph_module = _local_graph_module()
    graph = graph_module.graph
    sentence = graph_module.sentence
    word = graph_module.word
    phrase = graph_module.phrase
    pair = graph_module.pair
    people = graph_module.people
    organization = graph_module.organization
    location = graph_module.location
    other = graph_module.other
    o = graph_module.o

    # Resolve a concrete device (the sensors/modules cannot take 'auto').
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Find the data file automatically (will use extracted file if exists)
    data_file_path = find_data_file(args.data_path, args.train_portion)

    # Pass reward_function=None so each data item gets its own per-question reward
    # closure (reward.make_reward_function), which the program calls as
    # reward_fn(generator_output).
    train, dev, test = conll4_reader(data_path=data_file_path, dataset_portion=args.train_portion, reward_function=None)

    if args.train_size != -1:
        train = train[:args.train_size]

    program, dataset = program_declaration(train if not args.evaluate else test, args, device=device)

    if not args.evaluate:
        # Use more samples for the before/after readout so it isn't dominated by
        # sampling noise (this only adds cheap extra sampling, not extra forwards).
        eval_samples = max(64, getattr(args, 'num_samples', 8) * 4)
        baseline = program.evaluate_reward(dataset, num_samples=eval_samples, device=device)
        program.train(
            dataset,
            train_epoch_num=args.epochs,
            Optim=lambda params: torch.optim.Adam(
                [p for p in params if p.requires_grad], lr=args.lr),
            device=device,
        )
        trained = program.evaluate_reward(dataset, num_samples=eval_samples, device=device)
        _print_reward_summary(args, baseline, trained)
    else:
        reward = program.evaluate_reward(dataset, device=device)
        print("\n" + "=" * 64)
        print("Reinforcement evaluation (hard example)")
        print(f"  Mean reward over sampled decodings: {reward:.4f}")
        print("  (fraction of sampled decodings whose decoded yes/no answer")
        print("   matches the question label, per the asking-type decoder)")
        print("=" * 64)

    return 0


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
