import sys, os
# Parse --gpu arg early (before torch import) to set CUDA device
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
        break
else:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_script_dir, '..', '..', '..'))
sys.path.append(os.path.join(_script_dir, '..', '..'))
sys.path.append(os.path.join(_script_dir, '..'))
sys.path.append(_script_dir)
from pathlib import Path
import argparse, torch, logging

try:
    from monitor.constraint_monitor import enable_monitoring

    MONITORING_AVAILABLE = True
    enable_monitoring(slave_mode=True, master_url="http://localhost:8080")
except ImportError:
    MONITORING_AVAILABLE = False

from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor, FunctionalReaderSensor, ModuleSensor
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
# InternVL import is deferred to after arg parsing to avoid loading vllm when not needed
#from peftvllm import InternVLSharedHF as InternVL

try:
    from .preprocess import preprocess_dataset, preprocess_folders_and_files
    from .graph import create_graph
    from .modules import LEFTObjectEMB, LEFTRelationEMB, ResnetLEFT, LinearLayer
    from .dataset import g_relational_concepts, g_attribute_concepts
except ImportError:
    from preprocess import preprocess_dataset, preprocess_folders_and_files
    from graph import create_graph
    from modules import LEFTObjectEMB, LEFTRelationEMB, ResnetLEFT, LinearLayer
    from dataset import g_relational_concepts, g_attribute_concepts

RUN_DIR = Path(__file__).parent.resolve()
MODEL_DIR = RUN_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def ckpt_path(lr, epoch_idx, load_epoch_tag, batch, tnorm, subset, q_type="relation",
              lora_r=4, softmax_temp=1.0, max_objects=None, exp_tag=None):
    if exp_tag:
        return MODEL_DIR / f"program_{q_type}_{exp_tag}_e{epoch_idx}.pth"
    mo = f"_mo{max_objects}" if max_objects else ""
    return MODEL_DIR / f"program_{q_type}_{lr}_{epoch_idx}_{load_epoch_tag}__{batch}_6000_{tnorm}_{subset}_r{lora_r}_t{softmax_temp}{mo}.pth"


class OracleModule(torch.nn.Module):
    """Oracle module for object-level attributes. Returns ground truth from all_objects."""

    def __init__(self, attr_name, relation, device='cpu', confidence=1.0):
        super().__init__()
        self.attr_name = attr_name
        self.device = device
        self.confidence = confidence
        self.category = None
        for cat, values in g_attribute_concepts.items():
            if attr_name in values:
                self.category = cat
                break

    def forward(self, data, bounding_boxes):
        n = len(bounding_boxes)
        c = self.confidence
        yes = torch.tensor([1.0 - c, c], device=self.device)
        no = torch.tensor([c, 1.0 - c], device=self.device)
        results = []

        if self.category is not None:
            for obj in data[:n]:
                is_true = obj.get(self.category, '') == self.attr_name
                results.append(yes.clone() if is_true else no.clone())
        else:
            for _ in range(n):
                results.append(torch.tensor([0.5, 0.5], device=self.device))

        return results


class OracleDummyLearner(torch.nn.Module):
    """Passthrough learner that applies softmax to pre-computed oracle logits.
    Used with same-concept reader sensors (mirrors semantic_conversion.py DummyLearner)."""

    def forward(self, input):
        return torch.softmax(input, dim=-1)


logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Logic-guided VQA training / evaluation using DomiKnows framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Dummy mode test with existsL (default)
    uv run  main.py --dummy --epochs 4

    # Train with iotaL for query questions  
    uv run main.py --dummy --question-type query --epochs 4

    # Full training run
    uv run main.py --train-size 6000 --test-size 1000 --epochs 1 --lr 1e-5 --question-type query

    # Evaluation only
    uv run main.py --eval-only --question-type query --test-size 1000

    Question Types:
    relation       : "Is there a cube right of the sphere?" -> existsL
    query          : "What color is the cube?" -> iotaL  
    query_relation : "What color is the cube right of the sphere?" -> iotaL with relations
    exist          : "Are there any red cubes?" -> existsL
    """
    )

    parser.add_argument("--train-size", type=int, default=None,
                        help="Number of training examples to use (default: use all available)")

    parser.add_argument("--test-size", type=int, default=None,
                        help="Number of test examples to use (default: use all available)")

    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of training epochs (default: 4)")

    parser.add_argument("--lr", "--learning-rate", type=float, default=1e-6,
                        help="Learning rate for optimizer (default: 1e-6)")

    parser.add_argument("--batch-size", type=int, default=1,
                        help="Mini-batch size for training (default: 1)")

    parser.add_argument("--subset", type=int, default=-1,
                        help="Subset index 1-6 for memory-efficient training. "
                             "Splits train-size into 6 parts. -1 uses full set (default: -1)")

    parser.add_argument("--load-epoch", type=int, default=0,
                        help="Starting epoch when resuming training (default: 0)")

    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate a saved checkpoint")

    parser.add_argument("--dummy", action="store_true",
                        help="Use dummy mode with per-instance caching (fast reload)")
    parser.add_argument("--num-instances", type=int, default=300,
                        help="Number of instances to cache/load in dummy mode (default: 300)")
    parser.add_argument("--test-split", type=int, default=0,
                        help="Hold out last N compiled examples as test set (default: 0 = no split)")

    parser.add_argument("--tnorm", choices=["G", "P", "L"], default="G",
                        help="T-norm for fuzzy logic: G=Gödel, P=Product, L=Łukasiewicz (default: G)")

    parser.add_argument("--load_previous_save", action="store_true",
                        help="Load checkpoint from previous subset/epoch before training")

    parser.add_argument("--question-type",
                        choices=["relation", "query", "query_relation", "exist", "complex_relation", "counting"],
                        default="relation",
                        help="Type of questions to train on:\n"
                             "  relation      - existsL for relational questions (default)\n"
                             "  query         - iotaL for attribute query questions\n"
                             "  query_relation- iotaL for queries with spatial relations\n"
                             "  exist         - existsL for existence questions only\n"
                             "  complex_relation - query with more than one relationship\n"
                             "  counting      - counting question including greater, lesser, and count")

    parser.add_argument("--use-vlm", default=False, action="store_true", help="use InternVL for predictions")
    parser.add_argument("--peft", action="store_true",
                        help="Use PEFT (LoRA) fine-tuning with HuggingFace InternVL instead of vLLM inference")
    parser.add_argument("--oracle-mode", action="store_true",
                        help="Use ground truth answers instead of VLM/ResNet for debugging")
    parser.add_argument("--infer-only", action="store_true", help="Skip training, only evaluate a the model as is")
    parser.add_argument("--max-objects", type=int, default=None,
                        help="Filter dataset to images with at most N objects (reduces VLM forward passes)")
    parser.add_argument("--load-4bit", action="store_true",
                        help="Use QLoRA 4-bit quantization for the VLM (saves VRAM)")
    parser.add_argument("--softmax-temp", type=float, default=1.0,
                        help="Temperature for VLM softmax output (>1 = softer, prevents gradient vanishing)")
    parser.add_argument("--oracle-confidence", type=float, default=1.0,
                        help="Oracle confidence level (0.5=random, 1.0=perfect). E.g. 0.6 means 60%%/40%% probabilities")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU index to use (parsed early before torch import)")
    parser.add_argument("--max-num-patches", type=int, default=1,
                        help="Maximum number of image patches/tiles for InternVL (1-12, more=better quality, more VRAM)")
    parser.add_argument("--lora-r", type=int, default=4,
                        help="LoRA rank for PEFT fine-tuning (default: 4)")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha for PEFT fine-tuning (default: 2 * lora_r)")
    parser.add_argument("--min-objects", type=int, default=None,
                        help="Filter dataset to images with at least N objects (for generalization eval)")
    parser.add_argument("--exp-tag", type=str, default=None,
                        help="Experiment tag for checkpoint/results filenames (prevents collisions)")
    parser.add_argument("--no-lora", action="store_true",
                        help="Skip LoRA application (eval-only with base model weights, no random LoRA noise)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override model path (default: InternVL3_5-1B for --peft, InternVL3_5-8B otherwise)")

    args = parser.parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r

    if args.peft:
        args.use_vlm = True  # PEFT requires VLM

    if args.use_vlm and not args.oracle_mode:
        if args.peft:
            from peftvllm import InternVLSharedHF as InternVL
        else:
            from internVLvLLM import InternVLShared as InternVL

    CACHE_DIR = preprocess_folders_and_files(args.dummy)
    NUM_INSTANCES = args.num_instances
    device = "cuda" if torch.cuda.is_available() else "cpu"


    def filter_relation(property, arg1, arg2):
        return arg1.getAttribute("image_id") == arg2.getAttribute("image_id")


    # Load dataset with question type filter
    dataset = preprocess_dataset(args, NUM_INSTANCES, CACHE_DIR, question_type=args.question_type)

    # Report max-objects stats (filtering applied to train set only, after train/test split)
    if args.max_objects is not None:
        n_within = sum(1 for d in dataset if len(d.get('all_objects', [])) <= args.max_objects)
        print(f"Dataset: {len(dataset)} total, {n_within} with <={args.max_objects} objects "
              f"(filtering will be applied to train set after split)")

    # Filter by min objects if specified (for generalization evaluation)
    if args.min_objects is not None:
        before = len(dataset)
        dataset = [d for d in dataset if len(d.get('all_objects', [])) >= args.min_objects]
        print(f"Filtered dataset: {before} -> {len(dataset)} images (min {args.min_objects} objects)")

    # Create graph - use iotaL for query questions
    include_query = args.question_type in ['query', 'query_relation']
    results = create_graph(dataset, include_query_questions=include_query)

    questions_executions = results[0]
    graph = results[1]
    image = results[2]
    object = results[3]
    image_object_contains = results[4]
    obj1 = results[5]
    obj2 = results[6]
    relaton_2_obj = results[7]
    attribute_names_dict = results[8]
    query_types = results[9] if len(results) > 9 else [None] * len(dataset)

    # Answer-to-index mappings for query questions
    ATTRIBUTE_TO_INDEX = {}
    for attr, values in g_attribute_concepts.items():
        for idx, val in enumerate(values):
            ATTRIBUTE_TO_INDEX[val] = idx

    print(f"Dataset length: {len(dataset)}")
    print(f"Question type: {args.question_type}")

    # Set up logic labels
    for i in range(len(dataset)):
        dataset[i]["logic_str"] = questions_executions[i]

        if query_types[i] is not None:
            # Query question (iotaL) - answer is an attribute value
            answer = dataset[i].get('answer', '')
            if isinstance(answer, str):
                label_idx = ATTRIBUTE_TO_INDEX.get(answer.lower(), 0)
                dataset[i]["logic_label"] = torch.LongTensor([label_idx]).to(device)
            else:
                dataset[i]["logic_label"] = torch.LongTensor([0]).to(device)
            dataset[i]["query_type"] = query_types[i]
        else:
            # Existence/count question (existsL) - boolean label
            dataset[i]["logic_label"] = torch.LongTensor([bool(dataset[i]['answer'])]).to(device)
            dataset[i]["query_type"] = None

    # Print sample
    for i in range(min(3, len(dataset))):
        print(f"\nSample question: {dataset[i].get('question_raw', '')}")
        print(f"Sample answer: {dataset[i].get('answer', '')}")
        print(f"Sample execution:\n{questions_executions[i]}")

    print(f"\nSample question: {dataset[-1].get('question_raw', '')}")
    print(f"Sample answer: {dataset[-1].get('answer', '')}")
    print(f"Sample execution:\n{questions_executions[-1]}")

    # Pre-compute oracle ground truth and inject into data dict
    if args.oracle_mode:
        import math
        oracle_conf = args.oracle_confidence
        # Convert confidence to logit for softmax: softmax([0, logit])[1] = confidence
        if oracle_conf >= 0.999:
            oracle_logit = 100  # effectively 1.0
        elif oracle_conf <= 0.001:
            oracle_logit = -100  # effectively 0.0
        else:
            oracle_logit = math.log(oracle_conf / (1.0 - oracle_conf))
        print(f"Oracle confidence: {oracle_conf:.2f} (logit={oracle_logit:.3f})")

        spatial_list = g_relational_concepts.get("spatial_relation", [])
        for i in range(len(dataset)):
            all_objs = dataset[i].get('all_objects', [])
            n = len(all_objs)

            gt_spatial = dataset[i].get('relation_spatial_relation', None)

            if gt_spatial is not None:
                # gt_spatial[j*n+i, col] = 1.0 means "obj_j has relation col to obj_i"
                # The formula (from convert_CLEVR_domiKnowS) places RESULT at obj1
                # and SOURCE at obj2, so left(pair) = True means "obj1 is left of obj2",
                # which matches gt_spatial directly (no transpose needed).
                for s_idx, s_name in enumerate(spatial_list):
                    oracle_data = []
                    for pair_idx in range(n * n):
                        if gt_spatial[pair_idx][s_idx] > 0.5:
                            oracle_data.append([0, oracle_logit])
                        else:
                            oracle_data.append([oracle_logit, 0])
                    dataset[i][f"oracle_is_{s_name}"] = oracle_data

            for attr in ['size', 'color', 'material', 'shape']:
                oracle_data = []
                for obj_i in range(n):
                    for obj_j in range(n):
                        if all_objs[obj_i].get(attr) == all_objs[obj_j].get(attr):
                            oracle_data.append([0, oracle_logit])
                        else:
                            oracle_data.append([oracle_logit, 0])
                dataset[i][f"oracle_is_same_{attr}"] = oracle_data

    # Set up sensors
    image["pil_image"] = FunctionalReaderSensor(keyword="pil_image", forward=lambda data: [data])
    image["image_id"] = FunctionalReaderSensor(keyword='image_index', forward=lambda data: [data])

    object["bounding_boxes"] = FunctionalReaderSensor(keyword="objects_raw",
                                                      forward=lambda data: torch.Tensor(data).to(device))
    object["properties"] = ReaderSensor(keyword="all_objects")
    object["image_id"] = FunctionalSensor(image["image_id"], "bounding_boxes",
                                          forward=lambda data, data2: data * len(data2))
    if not args.use_vlm and not args.oracle_mode:
        resnet_model = ResnetLEFT(device=device)
        image["emb"] = ModuleSensor("image_id", "pil_image", module=resnet_model, device=device)

        object_feature_extraction_model = LEFTObjectEMB(device=device)
        object["feature_emb"] = ModuleLearner(image["emb"], "bounding_boxes", module=object_feature_extraction_model,
                                              device=device)

        object_feature_fc = LinearLayer(128 * 32 * 32, 1024, device=device)
        object["emb"] = ModuleLearner("feature_emb", "bounding_boxes", module=object_feature_fc, device=device)

    object[image_object_contains] = EdgeSensor(object["bounding_boxes"], image["pil_image"],
                                               relation=image_object_contains,
                                               forward=lambda b, _: torch.ones(len(b)).unsqueeze(-1))

    relaton_2_obj[obj1.reversed, obj2.reversed] = CompositionCandidateSensor(
        object['image_id'],
        relations=(obj1.reversed, obj2.reversed),
        forward=filter_relation)
    if not args.use_vlm and not args.oracle_mode:
        object_relation_extraction = LEFTRelationEMB(input_size=256, output_size=1024, device=device)
        relaton_2_obj["emb"] = ModuleLearner(image["emb"], object["bounding_boxes"], object["feature_emb"],
                                             module=object_relation_extraction, device=device)

    # Set up learners for attributes and relations
    spatial_relations = g_relational_concepts.get("spatial_relation", [])

    for attr_name, attr_variable in attribute_names_dict.items():
        if args.oracle_mode:
            if attr_name in spatial_relations:
                relaton_2_obj[f"{attr_variable}_label"] = FunctionalReaderSensor(
                    keyword=f"oracle_is_{attr_name}",
                    forward=lambda data: torch.Tensor(data).to(device))
                relaton_2_obj[attr_variable] = ModuleLearner(
                    f"{attr_name}_label", module=OracleDummyLearner(), device=device)
            elif attr_name.startswith("same_"):
                relaton_2_obj[f"{attr_variable}_label"] = FunctionalReaderSensor(
                    keyword=f"oracle_is_{attr_name}",
                    forward=lambda data: torch.Tensor(data).to(device))
                relaton_2_obj[attr_variable] = ModuleLearner(
                    f"{attr_name}_label", module=OracleDummyLearner(), device=device)
            else:
                object[attr_variable] = ModuleLearner(
                    object["properties"], object["bounding_boxes"],
                    module=OracleModule(attr_name, relation=1, device=device,
                                        confidence=args.oracle_confidence), device=device)
        elif not args.use_vlm:
            if attr_name in spatial_relations:
                relaton_2_obj[attr_variable] = ModuleLearner("emb", module=torch.nn.Linear(1024, 2).to(device),
                                                             device=device)
            elif attr_name.startswith("same_"):
                relaton_2_obj[attr_variable] = ModuleLearner("emb", module=torch.nn.Linear(1024, 2).to(device),
                                                             device=device)
            else:
                object[attr_variable] = ModuleLearner("emb", module=torch.nn.Linear(1024, 2).to(device), device=device)
        else:
            # VLM mode (both PEFT and inference) — VLM predicts everything
            MODEL_PATH = args.model_path or ("OpenGVLab/InternVL3_5-1B" if args.peft else "OpenGVLab/InternVL3_5-8B")
            vlm_extra = dict(use_llm_lora=not args.no_lora, use_vision_lora=False, load_4bit=args.load_4bit,
                            softmax_temperature=args.softmax_temp,
                            lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                            max_num=args.max_num_patches) if args.peft else {}
            if attr_name in spatial_relations:
                relaton_2_obj[attr_variable] = ModuleLearner(image["pil_image"], object["bounding_boxes"],
                                                             module=InternVL(model_path=MODEL_PATH, device=device,
                                                                             relation=2, attr=attr_name, **vlm_extra), device=device)
            elif attr_name.startswith("same_"):
                relaton_2_obj[attr_variable] = ModuleLearner(image["pil_image"], object["bounding_boxes"],
                                                             module=InternVL(model_path=MODEL_PATH, device=device,
                                                                             relation=2, attr=attr_name, **vlm_extra), device=device)
            else:
                object[attr_variable] = ModuleLearner(image["pil_image"], object["bounding_boxes"],
                                                      module=InternVL(model_path=MODEL_PATH, device=device, relation=1,
                                                                      attr=attr_name, **vlm_extra), device=device)

    # Train/test split (before compile; each dict already has logic_str/logic_label)
    test_data = None
    test_data_filtered = None
    if args.test_split > 0 and args.test_split < len(dataset):
        test_raw = dataset[-args.test_split:]
        dataset = dataset[:-args.test_split]
        # Create filtered test set (max-objects) if applicable
        if args.max_objects is not None:
            test_raw_filtered = [d for d in test_raw
                                 if len(d.get('all_objects', [])) <= args.max_objects]
            print(f"Train/test split: {len(dataset)} train, {len(test_raw)} test "
                  f"({len(test_raw_filtered)} with <={args.max_objects} objects)")
        else:
            test_raw_filtered = None
            print(f"Train/test split: {len(dataset)} train, {len(test_raw)} test")
    else:
        test_raw = None
        test_raw_filtered = None

    # Apply max-objects filtering to training data only (test set keeps all examples)
    if args.max_objects is not None:
        before_train = len(dataset)
        dataset = [d for d in dataset if len(d.get('all_objects', [])) <= args.max_objects]
        print(f"Training set after max-objects filter: {before_train} -> {len(dataset)} "
              f"(removed {before_train - len(dataset)} with >{args.max_objects} objects)")

    # Compile and create program
    dataset = graph.compile_executable(dataset, logic_keyword='logic_str', logic_label_keyword='logic_label')
    if test_raw is not None:
        test_data = graph.compile_executable(test_raw, logic_keyword='logic_str', logic_label_keyword='logic_label')
    if test_raw_filtered is not None and len(test_raw_filtered) < len(test_raw):
        test_data_filtered = graph.compile_executable(test_raw_filtered, logic_keyword='logic_str', logic_label_keyword='logic_label')

    poi = [image, object, *attribute_names_dict.values(), graph.constraint, relaton_2_obj]
    program = InferenceProgram(graph, SolverModel, poi=poi, device=device, tnorm=args.tnorm)

    _ckpt_extra = dict(lora_r=args.lora_r, softmax_temp=args.softmax_temp,
                       max_objects=args.max_objects, exp_tag=args.exp_tag)
    _results_file = f"results_{args.exp_tag}.txt" if args.exp_tag else "results.txt"

    save_file = ckpt_path(args.lr, 1, args.load_epoch, args.batch_size, args.tnorm, args.subset, args.question_type, **_ckpt_extra)

    if args.infer_only:
        with torch.no_grad():
            acc = program.evaluate_condition(dataset, device=device)
        print(f"Accuracy on Test: {acc:.2f}%")
        with open(_results_file, 'a') as f:
            print(save_file, file=f)
            print(f"Question type: {args.question_type}", file=f)
            print(f"Epoch: {args.load_epoch}", file=f)
            print(f"Learning rate: {args.lr}", file=f)
            print(f"Accuracy: {acc:.2f}%", file=f)
    else:
        if not args.eval_only:
            if args.load_previous_save and args.subset > 1:
                previous_save = ckpt_path(args.lr, 1, args.load_epoch, args.batch_size, args.tnorm, args.subset - 1,
                                          args.question_type, **_ckpt_extra)
                if previous_save.exists():
                    program.load(previous_save)
            elif args.load_previous_save and args.load_epoch > 0:
                previous_save = ckpt_path(args.lr, 1, args.load_epoch - 1, args.batch_size, args.tnorm, args.subset,
                                          args.question_type, **_ckpt_extra)
                if previous_save.exists():
                    program.load(previous_save)

            # Evaluate before training to establish baseline
            with torch.no_grad():
                acc = program.evaluate_condition(dataset, device=device)
            print(f"Accuracy before training: {acc:.2f}%")

            # Train using DomiKnowS constraint loss (program.train handles backprop)
            import gc

            # ================================================================
            # DIAGNOSTIC INSTRUMENTATION: Trace gradient chain through pipeline
            # ================================================================
            import types
            import numpy as np

            # --- Patch 1: InferenceModel.forward → log closs value/grad_fn ---
            from domiknows.program.model.lossModel import InferenceModel
            _orig_inf_forward = InferenceModel.forward
            _inf_step = [0]
            _closs_log = []  # (step, value, has_grad_fn, n_constraints)

            def _debug_inf_forward(self_model, builder, **kwargs):
                loss, datanode, builder_out = _orig_inf_forward(self_model, builder, **kwargs)
                if torch.is_tensor(loss):
                    has_fn = loss.grad_fn is not None
                    _closs_log.append((_inf_step[0], loss.item(), has_fn))
                    print(f"    [CLOSS] step={_inf_step[0]}: value={loss.item():.8f}, "
                          f"requires_grad={loss.requires_grad}, has_grad_fn={has_fn}")
                    if not has_fn:
                        print(f"    [CLOSS] *** NO GRAD_FN — backward will NOT reach VLM! ***")
                else:
                    _closs_log.append((_inf_step[0], float(loss) if loss else 0.0, False))
                    print(f"    [CLOSS] step={_inf_step[0]}: not a tensor: {type(loss).__name__}={loss}")
                _inf_step[0] += 1
                return loss, datanode, builder_out
            InferenceModel.forward = _debug_inf_forward

            # --- Patch 2: _fixVar → detect gradient chain breaks ---
            from domiknows.solver.lcLossBooleanMethods import lcLossBooleanMethods
            _orig_fixVar = lcLossBooleanMethods._fixVar
            _fixvar_detach_count = [0]
            _fixvar_total_count = [0]

            def _debug_fixVar(self_methods, var):
                result = _orig_fixVar(self_methods, var)
                for v_orig, v_fixed in zip(var, result):
                    if torch.is_tensor(v_orig):
                        _fixvar_total_count[0] += 1
                        orig_fn = v_orig.grad_fn is not None
                        fixed_fn = v_fixed.grad_fn is not None
                        if orig_fn and not fixed_fn:
                            _fixvar_detach_count[0] += 1
                            if _fixvar_detach_count[0] <= 5:
                                print(f"    [FIXVAR] *** DETACHED tensor with grad_fn! ***")
                        if not v_orig.requires_grad and _fixvar_total_count[0] <= 20:
                            print(f"    [FIXVAR] Input had requires_grad=False (val={v_orig.detach().cpu().numpy().flatten()[:4]})")
                return result
            lcLossBooleanMethods._fixVar = _debug_fixVar

            # --- Patch 3: VLM forward → log output values and grad status ---
            if args.peft:
                _orig_vlm_forward = InternVL.forward
                _vlm_step = [0]
                _vlm_log = []  # (step, mean_confidence, has_grad_fn)

                def _debug_vlm_forward(self_vlm, image, bounding_boxes, label=None):
                    result = _orig_vlm_forward(self_vlm, image, bounding_boxes, label=label)
                    has_fn = result.grad_fn is not None
                    max_prob = result.max(dim=-1).values.mean().item()
                    _vlm_log.append((_vlm_step[0], max_prob, has_fn))
                    if _vlm_step[0] < 10:
                        print(f"    [VLM] call #{_vlm_step[0]}: shape={list(result.shape)}, "
                              f"requires_grad={result.requires_grad}, has_grad_fn={has_fn}, "
                              f"mean_max_prob={max_prob:.4f}")
                        if result.numel() <= 10:
                            print(f"    [VLM] values={result.detach().cpu().numpy().round(4)}")
                    _vlm_step[0] += 1
                    return result
                InternVL.forward = _debug_vlm_forward

            # --- Patch 4: inferLocal → detect double softmax ---
            from domiknows.graph.dataNode import DataNode
            _orig_inferLocal = DataNode.inferLocal
            _inferlocal_step = [0]

            def _debug_inferLocal(self_dn, keys=("softmax",), **extra_kwargs):
                # Check values BEFORE softmax for double-softmax detection
                if _inferlocal_step[0] < 3 and "softmax" in keys:
                    # Check attributes on this datanode directly
                    for attr_key, attr_val in list(self_dn.attributes.items()):
                        if torch.is_tensor(attr_val) and attr_val.dim() >= 1:
                            vals = attr_val.detach().cpu().numpy().flatten()
                            if len(vals) == 2:
                                s = vals.sum()
                                if abs(s - 1.0) < 0.05:
                                    print(f"    [DOUBLE-SOFTMAX] pre-inferLocal: {attr_key} = {vals.round(4)} "
                                          f"(sums to {s:.4f} — ALREADY PROBABILITIES!)")
                                    post = np.exp(vals) / np.exp(vals).sum()
                                    print(f"    [DOUBLE-SOFTMAX] after 2nd softmax would be: {post.round(4)}")
                                    break
                    _inferlocal_step[0] += 1
                return _orig_inferLocal(self_dn, keys=keys, **extra_kwargs)
            DataNode.inferLocal = _debug_inferLocal

            print("=== DIAGNOSTIC PATCHES ACTIVE ===")
            print("Tracking: closs grad_fn, _fixVar detachments, VLM grad status, double-softmax")

            # ================================================================
            # END DIAGNOSTIC INSTRUMENTATION
            # ================================================================

            # Snapshot initial LoRA weights for tracking
            _lora_snapshot = {}
            for n, p in program.model.named_parameters():
                if p.requires_grad and 'lora' in n.lower():
                    _lora_snapshot[n] = p.data.clone()
                    break  # just track one

            # Monkey-patch train_epoch to track gradients at every step
            _orig_train_epoch = program.__class__.train_epoch
            _grad_step = [0]
            _grad_history = []  # (global_step, max_grad, grad_norm, n_params, n_lora)

            def _debug_train_epoch(self_prog, dataset, **kwargs):
                for result in _orig_train_epoch(self_prog, dataset, **kwargs):
                    # Extract loss value from result tuple
                    loss_val = result[0] if isinstance(result, tuple) else result
                    loss_info = ""
                    if torch.is_tensor(loss_val):
                        loss_info = f" loss={loss_val.item():.8f}"

                    # Collect gradient stats for all trainable parameters
                    lora_grads = []
                    non_lora_grads = []
                    for n, p in self_prog.model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            g_max = p.grad.abs().max().item()
                            g_norm = p.grad.norm().item()
                            if g_max > 0:
                                if 'lora' in n.lower():
                                    lora_grads.append((n, g_max, g_norm))
                                else:
                                    non_lora_grads.append((n, g_max, g_norm))

                    all_grads = lora_grads + non_lora_grads
                    overall_max = max((g for _, g, _ in all_grads), default=0.0)
                    overall_norm = sum(gn**2 for _, _, gn in all_grads)**0.5 if all_grads else 0.0

                    _grad_history.append((_grad_step[0], overall_max, overall_norm,
                                          len(all_grads), len(lora_grads)))

                    print(f"  step {_grad_step[0]}: params_w_grad={len(all_grads)} "
                          f"(lora={len(lora_grads)}), max_grad={overall_max:.8f}, "
                          f"grad_norm={overall_norm:.8f}{loss_info}")

                    # Print top gradient sources at every step
                    if lora_grads:
                        lora_grads.sort(key=lambda x: -x[1])
                        for n, g, gn in lora_grads[:3]:
                            print(f"    [lora] {n}: max={g:.8f} norm={gn:.8f}")
                    if non_lora_grads:
                        non_lora_grads.sort(key=lambda x: -x[1])
                        for n, g, gn in non_lora_grads[:2]:
                            print(f"    [other] {n}: max={g:.8f} norm={gn:.8f}")

                    if overall_max == 0.0:
                        no_grad_count = sum(1 for _, p in self_prog.model.named_parameters()
                                            if p.requires_grad and p.grad is None)
                        print(f"    WARNING: zero gradients! {no_grad_count} trainable params have grad=None")

                    # Periodic _fixVar summary
                    if _fixvar_detach_count[0] > 0 and _grad_step[0] % 10 == 0:
                        print(f"    [FIXVAR SUMMARY] {_fixvar_detach_count[0]}/{_fixvar_total_count[0]} "
                              f"tensors detached so far")

                    _grad_step[0] += 1
                    yield result
                    # Free accumulated computation graph memory between steps
                    gc.collect()
                    torch.cuda.empty_cache()
            program.train_epoch = types.MethodType(_debug_train_epoch, program)

            for i in range(args.epochs):
                print(f"Training epoch {i + 1}/{args.epochs}")
                save_file = ckpt_path(args.lr, i + 1, args.load_epoch, args.batch_size, args.tnorm, args.subset,
                                      args.question_type, **_ckpt_extra)
                epoch_start_step = _grad_step[0]
                program.train(dataset, Optim=torch.optim.Adam, train_epoch_num=1, c_lr=args.lr,
                              c_warmup_iters=0, batch_size=args.batch_size, device=device, print_loss=True)
                # Per-epoch gradient summary
                epoch_steps = [h for h in _grad_history if h[0] >= epoch_start_step]
                if epoch_steps:
                    non_zero = [(s, g) for s, g, _, _, _ in epoch_steps if g > 0]
                    zero = [(s, g) for s, g, _, _, _ in epoch_steps if g == 0.0]
                    print(f"  Epoch {i+1} gradient summary: {len(epoch_steps)} steps, "
                          f"{len(non_zero)} with gradients, {len(zero)} zero-gradient")
                    if non_zero:
                        print(f"    Grad range: min={min(g for _, g in non_zero):.8f} "
                              f"max={max(g for _, g in non_zero):.8f}")
                    if zero:
                        print(f"    First zero-gradient at step {zero[0][0]} "
                              f"(after {zero[0][0] - epoch_start_step} steps in this epoch)")
                # Check LoRA weight change
                for n, p in program.model.named_parameters():
                    if n in _lora_snapshot:
                        diff = (p.data - _lora_snapshot[n]).abs().max().item()
                        print(f"  LoRA weight change ({n}): {diff:.8f}")
                        break
                program.save(save_file)
                print(f"Saved to {save_file}")
                # Free graph memory between epochs to prevent OOM accumulation
                gc.collect()
                torch.cuda.empty_cache()

                # Per-epoch accuracy evaluation
                with torch.no_grad():
                    epoch_train_acc = program.evaluate_condition(dataset, device=device)
                print(f"  Epoch {i+1} train accuracy: {epoch_train_acc:.2f}%")
                if test_data is not None:
                    with torch.no_grad():
                        epoch_test_acc = program.evaluate_condition(test_data, device=device)
                    print(f"  Epoch {i+1} test accuracy: {epoch_test_acc:.2f}%")
                gc.collect()
                torch.cuda.empty_cache()

            # === DIAGNOSTIC SUMMARY ===
            print("\n" + "="*60)
            print("DIAGNOSTIC SUMMARY")
            print("="*60)
            # Constraint loss grad_fn analysis
            if _closs_log:
                with_fn = sum(1 for _, _, fn in _closs_log if fn)
                without_fn = sum(1 for _, _, fn in _closs_log if not fn)
                avg_loss = sum(v for _, v, _ in _closs_log) / len(_closs_log)
                print(f"Constraint loss: {len(_closs_log)} calls, "
                      f"{with_fn} WITH grad_fn, {without_fn} WITHOUT grad_fn")
                print(f"  Average closs: {avg_loss:.8f}")
                if without_fn > 0:
                    print(f"  *** {without_fn} constraint losses had NO grad_fn — "
                          f"backward cannot reach VLM for those steps! ***")
            # _fixVar detachment analysis
            print(f"_fixVar: {_fixvar_detach_count[0]}/{_fixvar_total_count[0]} "
                  f"tensors detached (gradient chain broken)")
            if _fixvar_detach_count[0] > 0:
                print(f"  *** _fixVar is breaking the gradient chain! ***")
            # VLM output analysis
            if args.peft and _vlm_log:
                with_fn = sum(1 for _, _, fn in _vlm_log if fn)
                without_fn = sum(1 for _, _, fn in _vlm_log if not fn)
                avg_conf = sum(c for _, c, _ in _vlm_log) / len(_vlm_log)
                print(f"VLM forward: {len(_vlm_log)} calls, "
                      f"{with_fn} WITH grad_fn, {without_fn} WITHOUT grad_fn")
                print(f"  Average max_prob (confidence): {avg_conf:.4f}")
                if without_fn > 0:
                    print(f"  *** {without_fn} VLM outputs had NO grad_fn — "
                          f"VLM is not part of computation graph! ***")
            print("="*60 + "\n")

            # Evaluate after training on train set
            with torch.no_grad():
                train_acc = program.evaluate_condition(dataset, device=device)
            print(f"Train accuracy after training: {train_acc:.2f}%")

            # Evaluate on test set (all examples) if available
            if test_data is not None:
                with torch.no_grad():
                    test_acc = program.evaluate_condition(test_data, device=device)
                print(f"Test accuracy (all examples): {test_acc:.2f}%")
            else:
                test_acc = None

            # Evaluate on filtered test set (max-objects) if available
            if test_data_filtered is not None:
                with torch.no_grad():
                    test_acc_filtered = program.evaluate_condition(test_data_filtered, device=device)
                print(f"Test accuracy (<={args.max_objects} objects): {test_acc_filtered:.2f}%")
            else:
                test_acc_filtered = None

            # Write results summary
            with open(_results_file, 'a') as f:
                print(f"=== {args.exp_tag or 'experiment'} ===", file=f)
                print(f"Question type: {args.question_type}, tnorm: {args.tnorm}, lr: {args.lr}", file=f)
                print(f"temp: {args.softmax_temp}, lora_r: {args.lora_r}, epochs: {args.epochs}", file=f)
                print(f"Train examples: {len(dataset)}, Test examples: {len(test_data) if test_data else 0}", file=f)
                print(f"Train accuracy: {train_acc:.2f}%", file=f)
                if test_acc is not None:
                    print(f"Test accuracy (all): {test_acc:.2f}%", file=f)
                if test_acc_filtered is not None:
                    print(f"Test accuracy (<={args.max_objects} obj): {test_acc_filtered:.2f}%", file=f)
                # Gradient fade summary
                if _grad_history:
                    first_zero = next((s for s, g, _, _, _ in _grad_history if g == 0.0), None)
                    if first_zero is not None:
                        print(f"Gradients: first zero at step {first_zero}/{len(_grad_history)} total", file=f)
                    else:
                        print(f"Gradients: never zero across {len(_grad_history)} steps", file=f)
                print("", file=f)
        else:
            epoch_to_eval = args.epochs if args.epochs > 0 else 1
            save_file = ckpt_path(args.lr, epoch_to_eval, args.load_epoch, args.batch_size, args.tnorm, args.subset,
                                  args.question_type, **_ckpt_extra)
            if save_file.exists():
                print(f"Loading from {save_file}")
                program.load(save_file)
                with torch.no_grad():
                    acc = program.evaluate_condition(dataset, device=device)
                print(f"Accuracy on Test: {acc:.2f}%")

                with open(_results_file, 'a') as f:
                    print(save_file, file=f)
                    print(f"Question type: {args.question_type}", file=f)
                    print(f"Epoch: {args.load_epoch}", file=f)
                    print(f"Learning rate: {args.lr}", file=f)
                    print(f"Accuracy: {acc:.2f}%", file=f)
            else:
                print(f"Checkpoint not found: {save_file}")
