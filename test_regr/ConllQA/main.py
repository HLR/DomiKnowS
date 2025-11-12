import sys
import torch
from pathlib import Path

sys.path.append('.')
sys.path.append('../..')

import argparse
from domiknows.program.lossprogram import PrimalDualProgram, GumbelPrimalDualProgram
from domiknows.program import POIProgram, SolverPOIProgram, IMLProgram, CallbackProgram
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.callbackprogram import ProgramStorageCallback
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from domiknows.program.lossprogram import PrimalDualProgram, InferenceProgram
from domiknows.program.model.pytorch import SolverModel, SolverModelDictLoss
from domiknows.program.loss import NBCrossEntropyLoss, NBCrossEntropyIMLoss, NBCrossEntropyDictLoss
from domiknows.sensor.pytorch.sensors import FunctionalSensor, JointSensor, ModuleSensor, ReaderSensor, \
    FunctionalReaderSensor, cache, TorchCache
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor, EdgeSensor, \
    CompositionCandidateReaderSensor
from reader import conll4_reader
import numpy as np

import spacy

# from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm')  # English()

import logging

logging.basicConfig(level=logging.INFO)

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
    def __init__(self) -> None:
        self.tokenizer = BertTokenizerFast.from_pretrained(TRANSFORMER_MODEL)

    def __call__(self, text):
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(text, padding=True, return_tensors='pt', return_offsets_mapping=True)

        ids = tokens['input_ids']
        mask = tokens['attention_mask']
        offset = tokens['offset_mapping']

        idx = mask.nonzero()[:, 0].unsqueeze(-1)
        mapping = torch.zeros(idx.shape[0], idx.max() + 1)
        mapping.scatter_(1, idx, 1)

        mask = mask.bool()
        ids = ids.masked_select(mask)
        offset = torch.stack((offset[:, :, 0].masked_select(mask), offset[:, :, 1].masked_select(mask)), dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return mapping, ids, offset, tokens


class BERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = BertModel.from_pretrained(TRANSFORMER_MODEL)
        
        # Freeze ALL BERT parameters
        for param in self.module.parameters():
            param.requires_grad = False
        
        # Set to eval mode permanently
        self.module.eval()
        
        print(f"[INFO] BERT frozen - trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, input):
        # Always use torch.no_grad() for BERT
        with torch.no_grad():
            input = input.unsqueeze(0)
            _out = self.module(input)

            out, *_ = _out

            if (isinstance(out, str)):
                out = _out.last_hidden_state

            assert out.shape[0] == 1
            out = out.squeeze(0)
            return out


class Classifier(torch.nn.Sequential):
    def __init__(self, in_features) -> None:
        linear = torch.nn.Linear(in_features, 2)
        super().__init__(linear)


def program_declaration(train, args, device='auto'):
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o
    from graph import work_for, located_in, live_in, orgbase_on, kill
    from graph import rel_sentence_contains_word, rel_phrase_contains_word, rel_pair_phrase1, rel_pair_phrase2, \
        rel_sentence_contains_phrase

    # Force CUDA
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda:0'
            print(f"[INFO] GPU detected - using {device}")
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("[WARNING] No GPU detected - using CPU (will be slow!)")
    
    graph.detach()

    phrase['text'] = ReaderSensor(keyword='tokens')

    def word2vec(text):
        texts = list(map(lambda x: ' '.join(x.split('/')), text))
        tokens_list = list(nlp.pipe(texts))
        return torch.tensor(np.array([tokens.vector for tokens in tokens_list]))

    phrase['w2v'] = FunctionalSensor('text', forward=word2vec)

    def merge_phrase(phrase_text):
        return [' '.join(phrase_text)], torch.ones((1, len(phrase_text)))

    sentence['text', rel_sentence_contains_phrase.reversed] = JointSensor(phrase['text'], forward=merge_phrase)
    word[rel_sentence_contains_word, 'ids', 'offset', 'text'] = JointSensor(sentence['text'], forward=Tokenizer())
    
    # Move BERT to device immediately
    bert_module = BERT()
    bert_module = bert_module.to(device)
    word['bert'] = ModuleSensor('ids', module=bert_module)

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
        return torch.tensor(ph_word_overlap)

    phrase[rel_phrase_contains_word.reversed] = EdgeSensor(phrase['text'], word['offset'],
                                                           relation=rel_phrase_contains_word.reversed,
                                                           forward=match_phrase)

    def phrase_bert(bert):
        return bert

    phrase['bert'] = FunctionalSensor(rel_phrase_contains_word.reversed(word['bert']), forward=phrase_bert)
    phrase['emb'] = FunctionalSensor('bert', 'w2v', forward=lambda bert, w2v: torch.cat((bert, w2v), dim=-1))

    # Create classifiers and move to device
    phrase[people] = ModuleLearner('emb', module=Classifier(FEATURE_DIM).to(device))
    phrase[organization] = ModuleLearner('emb', module=Classifier(FEATURE_DIM).to(device))
    phrase[location] = ModuleLearner('emb', module=Classifier(FEATURE_DIM).to(device))
    phrase[other] = ModuleLearner('emb', module=Classifier(FEATURE_DIM).to(device))
    phrase[o] = ModuleLearner('emb', module=Classifier(FEATURE_DIM).to(device))

    train_dataset = graph.compile_logic(train, logic_keyword='logic_str', logic_label_keyword='logic_label')

    # During training: only local
    train_infer = ['local/softmax']
    
    # During evaluation: can add ILP if needed
    if args.use_ilp_eval:
        eval_infer = ['local/argmax', 'ILP']
    else:
        eval_infer = ['local/argmax']
    
    program = GumbelPrimalDualProgram(
        graph, 
        SolverModel,
        poi=[phrase, sentence, word, people, organization, location, graph.constraint],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        inferTypes=train_infer,
        use_gumbel=True,
        initial_temp=2.0,      
        final_temp=0.5,        
        beta=10.0,
        device=device,
        tnorm='L',
        counting_tnorm=args.counting_tnorm,
        sample=True,
        sampleSize=50
    )
    
    # Store both inference types
    program.train_inferTypes = train_infer
    program.eval_inferTypes = eval_infer
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in program.model.parameters())
    trainable_params = sum(p.numel() for p in program.model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")
    print(f"[INFO] Frozen parameters: {total_params - trainable_params:,}")
    
    return program, train_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="Getting the arguments passed")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (increased)")
    parser.add_argument("--epochs", type=int, default=5, help="Total epochs (REDUCED)")
    parser.add_argument("--evaluate", action='store_true')
    parser.add_argument("--train_size", type=int, default=-1, help="Number of training sample")
    parser.add_argument("--train_portion", type=str, default="entities_only_with_1_things_YN", help="Training subset")
    parser.add_argument("--checked_acc", type=float, default=0, help="Accuracy to test")
    parser.add_argument("--counting_tnorm", choices=["G", "P", "L", "SP"], default="G", help="T-norm for counting")
    parser.add_argument("--data_path", type=str, default="conllQA.json", help="Path to data file")
    
    # Performance parameters
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Warmup epochs (REDUCED)")
    parser.add_argument("--constraint_epochs", type=int, default=5, help="Constraint epochs (REDUCED)")
    parser.add_argument("--constraint_loss_scale", type=float, default=10.0, help="Constraint loss scale (REDUCED)")
    parser.add_argument("--c_freq", type=int, default=50, help="Constraint update frequency (INCREASED)")
    
    # ILP control
    parser.add_argument("--use_ilp_eval", action='store_true', 
                       help="Use ILP solver during evaluation (slow)")
    parser.add_argument("--no_constraints", action='store_false',
                       help="Train without constraint loss (faster)")
    
    
    # Quick testing
    parser.add_argument("--quick_test", action='store_true', help="Quick test with 100 samples")
    
    args = parser.parse_args()
    return args


def main(args):
    from graph import graph, sentence, word, phrase, pair
    from graph import people, organization, location, other, o

    # Force device
    if args.device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\n{'='*60}")
    print(f"DEVICE CONFIGURATION")
    print(f"{'='*60}")
    print(f"Selected device: {device}")
    if 'cuda' in device:
        # Extract GPU index from device string
        gpu_id = int(device.split(':')[1]) if ':' in device else 0
        print(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")
    
    print(f"\n{'='*60}")
    print(f"DEVICE CONFIGURATION")
    print(f"{'='*60}")
    print(f"Selected device: {device}")
    if 'cuda' in device:
        print(f"GPU: {torch.cuda.get_device_name(torch.device(device))}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")

    data_file_path = find_data_file(args.data_path, args.train_portion)
    train, dev, test = conll4_reader(data_path=data_file_path, dataset_portion=args.train_portion)

    # Quick test mode
    if args.quick_test:
        train = train[:100]
        print(f"[INFO] QUICK TEST MODE - Using only {len(train)} samples")
    
    if args.train_size != -1:
        train = train[:args.train_size]
    
    print(f"[INFO] Dataset size: {len(train)}")

    program, dataset = program_declaration(train if not args.evaluate else test, args, device=device)

    if not args.evaluate:
        print(f"\n[INFO] Starting OPTIMIZED training:")
        print(f"  - Device: {device}")
        print(f"  - Warmup epochs: {args.warmup_epochs}")
        print(f"  - Constraint epochs: {args.constraint_epochs}")
        print(f"  - Total epochs: {args.epochs}")
        print(f"  - Constraint loss scale: {args.constraint_loss_scale}x")
        print(f"  - Learning rate: {args.lr}")
        print(f"  - Constraint update frequency: every {args.c_freq} steps")
        
        program.to(device)
        
        program.train(
            dataset, 
            Optim=torch.optim.Adam,
            train_epoch_num=args.epochs,
            warmup_epochs=args.warmup_epochs,
            constraint_epochs=args.constraint_epochs,
            constraint_only=False,            # Use both losses
            constraint_loss_scale=args.constraint_loss_scale,
            c_lr=args.lr,
            c_warmup_iters=len(dataset) * args.warmup_epochs,  # After warmup
            c_freq=args.c_freq,               # Update less frequently
            batch_size=1,
            device=device,
            print_loss=True                  # Enable verbose logging
        )
    else:
        program.load(f"training_{args.epochs}_lr_{args.lr}.pth")
        program.to(device)

    # Evaluation
    program.inferTypes = ['local/argmax']
    
    print("\n[INFO] Evaluating...")
    results = program.evaluate_condition(dataset, device=device)
    
    # Display results
    output_f = open("result.txt", 'a')
    portion = "Training" if not args.evaluate else "Testing"
    
    print(f"\n{'='*60}")
    print(f"{portion} Results")
    print(f"{'='*60}")
    
    if results['counting_accuracy'] is not None:
        print(f"Counting constraint results:")
        print(f"  MAE: {results['counting_mae']:.3f}")
        print(f"  RMSE: {results['counting_rmse']:.3f}")
        print(f"  Accuracy (±0.5): {results['counting_accuracy']:.2f}%")
        
        print(f"training_{args.epochs}_lr_{args.lr}.pth", file=output_f)
        print(f"{portion} Counting Accuracy: {results['counting_accuracy']:.2f}%", file=output_f)
    
    if results['boolean_accuracy'] is not None:
        print(f"Boolean accuracy: {results['boolean_accuracy']:.2f}%")
        print(f"{portion} Boolean Accuracy: {results['boolean_accuracy']:.2f}%", file=output_f)
    
    print(f"Primary metric: {results['primary_metric']:.2f}%")
    print("#" * 40, file=output_f)
    output_f.close()

    if args.checked_acc:
        assert results['primary_metric'] > args.checked_acc

    return 0

if __name__ == '__main__':
    args = parse_arguments()
    main(args)