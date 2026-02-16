import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')
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


def ckpt_path(lr, epoch_idx, load_epoch_tag, batch, tnorm, subset, q_type="relation"):
    return MODEL_DIR / f"program_{q_type}_{lr}_{epoch_idx}_{load_epoch_tag}__{batch}_6000_{tnorm}_{subset}.pth"


class OracleModule(torch.nn.Module):
    """Oracle module for object-level attributes. Returns ground truth from all_objects."""

    def __init__(self, attr_name, relation, device='cpu'):
        super().__init__()
        self.attr_name = attr_name
        self.device = device
        self.category = None
        for cat, values in g_attribute_concepts.items():
            if attr_name in values:
                self.category = cat
                break

    def forward(self, data, bounding_boxes):
        n = len(bounding_boxes)
        yes = torch.tensor([0.0, 1.0], device=self.device)
        no = torch.tensor([1.0, 0.0], device=self.device)
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
                        help="Use lightweight dummy mode with 20 instances for testing")

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
    parser.add_argument("--oracle-mode", action="store_true",
                        help="Use ground truth answers instead of VLM/ResNet for debugging")
    parser.add_argument("--infer-only", action="store_true", help="Skip training, only evaluate a the model as is")

    args = parser.parse_args()

    if args.use_vlm and not args.oracle_mode:
        from internVLvLLM import InternVLShared as InternVL

    CACHE_DIR = preprocess_folders_and_files(args.dummy)
    NUM_INSTANCES = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"


    def filter_relation(property, arg1, arg2):
        return arg1.getAttribute("image_id") == arg2.getAttribute("image_id")


    # Load dataset with question type filter
    dataset = preprocess_dataset(args, NUM_INSTANCES, CACHE_DIR, question_type=args.question_type)

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

    # Pre-compute oracle ground truth and inject into data dict
    if args.oracle_mode:
        spatial_list = g_relational_concepts.get("spatial_relation", [])
        for i in range(len(dataset)):
            all_objs = dataset[i].get('all_objects', [])
            n = len(all_objs)

            gt_spatial = dataset[i].get('relation_spatial_relation', None)

            # Diagnostic for first instance
            if i == 0:
                print(f"\n=== ORACLE DIAGNOSTIC (instance 0) ===")
                print(f"  n_objects (from all_objects) = {n}")
                objs_raw = dataset[i].get('objects_raw', None)
                print(f"  objects_raw type: {type(objs_raw).__name__}, len: {len(objs_raw) if objs_raw is not None else 'N/A'}")
                if objs_raw is not None and hasattr(objs_raw, '__len__'):
                    print(f"  objects_raw shape: {objs_raw.shape if hasattr(objs_raw, 'shape') else len(objs_raw)}")
                print(f"  all_objects keys: {list(all_objs[0].keys()) if all_objs else 'EMPTY'}")
                if all_objs:
                    for oi, obj in enumerate(all_objs):
                        print(f"    obj{oi}: color={obj.get('color')}, shape={obj.get('shape')}, size={obj.get('size')}, material={obj.get('material')}")
                print(f"  relation_spatial_relation present: {gt_spatial is not None}")
                if gt_spatial is not None:
                    import numpy as np
                    gt_arr = np.array(gt_spatial)
                    print(f"  gt_spatial shape: {gt_arr.shape}")
                    print(f"  gt_spatial dtype: {gt_arr.dtype}")
                    print(f"  gt_spatial non-zero count: {np.count_nonzero(gt_arr)}")
                    print(f"  gt_spatial sum per column: {gt_arr.sum(axis=0)}")
                    for s_idx, s_name in enumerate(spatial_list):
                        col = gt_arr[:, s_idx]
                        print(f"    {s_name}: {int(col.sum())} True pairs out of {n*n}")
                        for src in range(n):
                            row = []
                            for dst in range(n):
                                row.append('Y' if col[src*n+dst] > 0.5 else 'N')
                            print(f"      obj{src}: {row}")
                else:
                    print(f"  WARNING: relation_spatial_relation is None!")
                    all_keys = list(dataset[i].keys())
                    print(f"  Available data dict keys: {all_keys}")
                print(f"  question: {dataset[i].get('question_raw', 'N/A')}")
                print(f"  answer: {dataset[i].get('answer', 'N/A')}")
                print(f"=== END DIAGNOSTIC ===\n")

            if gt_spatial is not None:
                # gt_spatial[j*n+i, col] = 1.0 means "obj_j has relation col to obj_i"
                # But execution.py places SOURCE at obj1 and RESULT at obj2,
                # so left(pair) = True should mean "obj2 is left of obj1".
                # We must transpose: oracle[src*n+dst] = gt_spatial[dst*n+src]
                for s_idx, s_name in enumerate(spatial_list):
                    oracle_data = []
                    for src in range(n):
                        for dst in range(n):
                            if gt_spatial[dst * n + src][s_idx] > 0.5:
                                oracle_data.append([0, 100])
                            else:
                                oracle_data.append([100, 0])
                    dataset[i][f"oracle_is_{s_name}"] = oracle_data

            for attr in ['size', 'color', 'material', 'shape']:
                oracle_data = []
                for obj_i in range(n):
                    for obj_j in range(n):
                        if all_objs[obj_i].get(attr) == all_objs[obj_j].get(attr):
                            oracle_data.append([0, 100])
                        else:
                            oracle_data.append([100, 0])
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
                    module=OracleModule(attr_name, relation=1, device=device), device=device)
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
            MODEL_PATH = "OpenGVLab/InternVL3_5-8B"
            if attr_name in spatial_relations:
                relaton_2_obj[attr_variable] = ModuleLearner(image["pil_image"], object["bounding_boxes"],
                                                             module=InternVL(model_path=MODEL_PATH, device=device,
                                                                             relation=2, attr=attr_name), device=device)
            elif attr_name.startswith("same_"):
                relaton_2_obj[attr_variable] = ModuleLearner(image["pil_image"], object["bounding_boxes"],
                                                             module=InternVL(model_path=MODEL_PATH, device=device,
                                                                             relation=2, attr=attr_name), device=device)
            else:
                object[attr_variable] = ModuleLearner(image["pil_image"], object["bounding_boxes"],
                                                      module=InternVL(model_path=MODEL_PATH, device=device, relation=1,
                                                                      attr=attr_name), device=device)

    # Compile and create program
    dataset = graph.compile_executable(dataset, logic_keyword='logic_str', logic_label_keyword='logic_label')

    poi = [image, object, *attribute_names_dict.values(), graph.constraint, relaton_2_obj]
    program = InferenceProgram(graph, SolverModel, poi=poi, device=device, tnorm=args.tnorm)

    save_file = ckpt_path(args.lr, 1, args.load_epoch, args.batch_size, args.tnorm, args.subset, args.question_type)

    if args.infer_only:
        acc = program.evaluate_condition(dataset, device=device)
        print(f"Accuracy on Test: {acc * 100:.2f}%")
        with open('results.txt', 'a') as f:
            print(save_file, file=f)
            print(f"Question type: {args.question_type}", file=f)
            print(f"Epoch: {args.load_epoch}", file=f)
            print(f"Learning rate: {args.lr}", file=f)
            print(f"Accuracy: {acc * 100:.2f}%", file=f)
    else:
        if not args.eval_only:
            if args.load_previous_save and args.subset > 1:
                previous_save = ckpt_path(args.lr, 1, args.load_epoch, args.batch_size, args.tnorm, args.subset - 1,
                                          args.question_type)
                if previous_save.exists():
                    program.load(previous_save)
            elif args.load_previous_save and args.load_epoch > 0:
                previous_save = ckpt_path(args.lr, 1, args.load_epoch - 1, args.batch_size, args.tnorm, args.subset,
                                          args.question_type)
                if previous_save.exists():
                    program.load(previous_save)

            for i in range(args.epochs):
                print(f"Training epoch {i + 1}/{args.epochs}")
                save_file = ckpt_path(args.lr, i + 1, args.load_epoch, args.batch_size, args.tnorm, args.subset,
                                      args.question_type)
                program.train(dataset, Optim=torch.optim.Adam, train_epoch_num=1, c_lr=args.lr,
                              c_warmup_iters=0, batch_size=args.batch_size, device=device, print_loss=False)
                program.save(save_file)
                print(f"Saved to {save_file}")
        else:
            epoch_to_eval = args.epochs if args.epochs > 0 else 1
            save_file = ckpt_path(args.lr, epoch_to_eval, args.load_epoch, args.batch_size, args.tnorm, args.subset,
                                  args.question_type)
            if save_file.exists():
                print(f"Loading from {save_file}")
                program.load(save_file)
                acc = program.evaluate_condition(dataset, device=device)
                print(f"Accuracy on Test: {acc * 100:.2f}%")

                with open('results.txt', 'a') as f:
                    print(save_file, file=f)
                    print(f"Question type: {args.question_type}", file=f)
                    print(f"Epoch: {args.load_epoch}", file=f)
                    print(f"Learning rate: {args.lr}", file=f)
                    print(f"Accuracy: {acc * 100:.2f}%", file=f)
            else:
                print(f"Checkpoint not found: {save_file}")
