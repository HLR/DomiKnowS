import sys, os
# Note: CUDA_VISIBLE_DEVICES can be set via environment variable before running
# e.g., CUDA_VISIBLE_DEVICES=1 python main.py
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

from domiknows import setProductionLogMode
setProductionLogMode()

from domiknows.program import CallbackProgram
from domiknows.program.lossprogram import GumbelInferenceProgram
from domiknows.program.model.pytorch import SolverModel, PoiModel
import torch.nn as nn
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor, FunctionalReaderSensor, ModuleSensor
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor

from domiknows.program.plugins.callback_plugin_manager import create_standard_plugin_manager
from domiknows.program.plugins.bert_unfreezing_plugin import create_optimizer_factory, create_optimizer_with_differential_lr

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

_models = {}


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
    """Passthrough learner that applies softmax to pre-computed oracle logits."""

    def forward(self, input):
        return torch.softmax(input, dim=-1)


class MaskedCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss wrapper that accepts (logit, labels, mask)."""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logit, labels, mask=None):
        # logit: (N, 2)  labels: (N,) with values 0 or 1
        labels = labels.long()
        per_sample = self.ce(logit, labels)
        if mask is not None and mask.any():
            per_sample = per_sample * mask.float()
            return per_sample.sum() / mask.float().sum().clamp(min=1)
        return per_sample.mean()


class _LossFactory(dict):
    """dict subclass that auto-creates MaskedCrossEntropyLoss for any key.
    __bool__ returns True so `if not self.loss:` in PoiModel.poi_loss
    doesn't short-circuit when the dict is initially empty."""
    def __missing__(self, key):
        val = MaskedCrossEntropyLoss()
        self[key] = val
        return val
    def __bool__(self):
        return True  # always truthy — losses will be created on demand


class InferenceProgramWithCallbacks(CallbackProgram, GumbelInferenceProgram):
    """InferenceProgram with callback support."""
    
    def default_after_train_step(self, output=None):
        """Override to do nothing - GumbelInferenceProgram already handles backward."""
        pass
    
    def __init__(self, graph, Model, loss=None, **kwargs):
        """
        Initialize with proper handling of loss parameter.
        
        Args:
            graph: Knowledge graph
            Model: Model class (e.g., PoiModel, SolverModel)
            loss: Loss function (optional, primarily for PoiModel)
            **kwargs: Additional arguments
        """
        # Standard initialization
        super().__init__(graph, Model, loss=loss, **kwargs)
        
        # Initialize all callback hooks
        self.after_train_step = [self.default_after_train_step]
        self.before_train = []
        self.after_train = []
        self.before_train_epoch = []
        self.after_train_epoch = []
        self.before_train_step = []
        self.before_test = []
        self.after_test = []
        self.before_test_epoch = []
        self.after_test_epoch = []
        self.before_test_step = []
        self.after_test_step = []


def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, str):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Logic-guided VQA training / evaluation using DomiKnows framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dummy mode test with existsL (default)
  uv run main.py --dummy --epochs 4

  # Train with iotaL for query questions  
  uv run main.py --dummy --question-type query --epochs 4

  # Full training run
  uv run main.py --train-size 6000 --test-size 1000 --epochs 1 --lr 1e-5 --question-type query

  # Evaluation only
  uv run main.py --eval-only --question-type query --test-size 1000
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
                        help="Subset index 1-6 for memory-efficient training (default: -1)")
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
                        help="Type of questions to train on (default: relation)")
    parser.add_argument("--use-vlm", default=False, action="store_true", 
                        help="use InternVL for predictions")
    parser.add_argument("--oracle-mode", action="store_true",
                        help="Use ground truth answers instead of VLM/ResNet for debugging")
    parser.add_argument("--infer-only", action="store_true", 
                        help="Skip training, only evaluate a the model as is")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for computation (default: cuda if available)")
    
    # Gumbel-Softmax settings
    parser.add_argument("--use_gumbel", type=str2bool, nargs='?', const=True, default=False, 
                        help="Use Gumbel-Softmax for counting")
    parser.add_argument("--gumbel_temp_start", type=float, default=5.0, 
                        help="Initial Gumbel temperature")
    parser.add_argument("--gumbel_temp_end", type=float, default=0.5, 
                        help="Final Gumbel temperature")
    parser.add_argument("--gumbel_anneal_start", type=int, default=0, 
                        help="Epoch to start annealing temperature")
    parser.add_argument("--hard_gumbel", type=str2bool, nargs='?', const=True, default=True, 
                        help="Use hard Gumbel")

    # Register callback plugin arguments (exclude BERT unfreezing)
    from domiknows.program.plugins.callback_plugin_manager import CallbackPluginManager
    from domiknows.program.plugins.epoch_logging_plugin import EpochLoggingPlugin
    from domiknows.program.plugins.adaptive_tnorm_plugin import AdaptiveTNormPlugin
    from domiknows.program.plugins.gradient_flow_plugin import GradientFlowPlugin
    from domiknows.program.plugins.counting_schedule_plugin import CountingSchedulePlugin
    from domiknows.program.plugins.gumbel_monitoring_plugin import GumbelMonitoringPlugin
    
    plugin_manager = CallbackPluginManager()
    plugin_manager.register(EpochLoggingPlugin(), 'EpochLogging')
    plugin_manager.register(AdaptiveTNormPlugin(), 'AdaptiveTNorm')
    plugin_manager.register(GradientFlowPlugin(), 'GradientFlow')
    plugin_manager.register(CountingSchedulePlugin(), 'CountingSchedule')
    plugin_manager.register(GumbelMonitoringPlugin(), 'GumbelMonitoring')
    # NOTE: BERT unfreezing plugin excluded - no BERT model in VQA task
    
    plugin_manager.add_arguments_to_parser(parser)
    
    args = parser.parse_args()
    return args


def program_declaration(train, dev, args, device='cpu'):
    """Create and configure the DomiKnows program with sensors and learners."""
    global _models
    
    if args.use_vlm and not args.oracle_mode:
        from internVLvLLM import InternVLShared as InternVL

    def filter_relation(property, arg1, arg2):
        return arg1.getAttribute("image_id") == arg2.getAttribute("image_id")

    # Load dataset with question type filter
    dataset = train
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

    # Set up logic labels
    for i in range(len(dataset)):
        dataset[i]["logic_str"] = questions_executions[i]

        if query_types[i] is not None:
            answer = dataset[i].get('answer', '')
            if isinstance(answer, str):
                label_idx = ATTRIBUTE_TO_INDEX.get(answer.lower(), 0)
                dataset[i]["logic_label"] = torch.LongTensor([label_idx]).to(device)
            else:
                dataset[i]["logic_label"] = torch.LongTensor([0]).to(device)
            dataset[i]["query_type"] = query_types[i]
        else:
            dataset[i]["logic_label"] = torch.LongTensor([bool(dataset[i]['answer'])]).to(device)
            dataset[i]["query_type"] = None

    # Pre-compute oracle ground truth
    if args.oracle_mode:
        spatial_list = g_relational_concepts.get("spatial_relation", [])
        for i in range(len(dataset)):
            all_objs = dataset[i].get('all_objects', [])
            n = len(all_objs)
            gt_spatial = dataset[i].get('relation_spatial_relation', None)

            if gt_spatial is not None:
                for s_idx, s_name in enumerate(spatial_list):
                    oracle_data = []
                    for pair_idx in range(n * n):
                        if gt_spatial[pair_idx][s_idx] > 0.5:
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

    # Set up sensors - shared across all modes
    image["pil_image"] = FunctionalReaderSensor(keyword="pil_image", forward=lambda data: [data])
    image["image_id"] = FunctionalReaderSensor(keyword='image_index', forward=lambda data: [data])
    object["bounding_boxes"] = FunctionalReaderSensor(keyword="objects_raw",
                                                      forward=lambda data: torch.Tensor(data).to(device))
    object["properties"] = ReaderSensor(keyword="all_objects")
    object["image_id"] = FunctionalSensor(image["image_id"], "bounding_boxes",
                                          forward=lambda data, data2: data * len(data2))
    
    # Mode-specific embeddings
    if not args.use_vlm and not args.oracle_mode:
        resnet_model = ResnetLEFT(device=device)
        image["emb"] = ModuleSensor("image_id", "pil_image", module=resnet_model, device=device)
        object_feature_extraction_model = LEFTObjectEMB(device=device)
        object["feature_emb"] = ModuleLearner(image["emb"], "bounding_boxes", 
                                              module=object_feature_extraction_model, device=device)
        object_feature_fc = LinearLayer(128 * 32 * 32, 1024, device=device)
        object["emb"] = ModuleLearner("feature_emb", "bounding_boxes", 
                                      module=object_feature_fc, device=device)
        _models['resnet'] = resnet_model
        _models['object_emb'] = object_feature_extraction_model
        _models['object_fc'] = object_feature_fc

    object[image_object_contains] = EdgeSensor(object["bounding_boxes"], image["pil_image"],
                                               relation=image_object_contains,
                                               forward=lambda b, _: torch.ones(len(b)).unsqueeze(-1))

    relaton_2_obj[obj1.reversed, obj2.reversed] = CompositionCandidateSensor(
        object['image_id'], relations=(obj1.reversed, obj2.reversed), forward=filter_relation)
    
    if not args.use_vlm and not args.oracle_mode:
        object_relation_extraction = LEFTRelationEMB(input_size=256, output_size=1024, device=device)
        relaton_2_obj["emb"] = ModuleLearner(image["emb"], object["bounding_boxes"], 
                                             object["feature_emb"],
                                             module=object_relation_extraction, device=device)
        _models['relation_emb'] = object_relation_extraction

    # Set up learners for attributes and relations
    spatial_relations = g_relational_concepts.get("spatial_relation", [])
    classifiers = {}

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
                classifier = torch.nn.Linear(1024, 2).to(device)
                classifiers[attr_name] = classifier
                relaton_2_obj[attr_variable] = ModuleLearner("emb", module=classifier, device=device)
            elif attr_name.startswith("same_"):
                classifier = torch.nn.Linear(1024, 2).to(device)
                classifiers[attr_name] = classifier
                relaton_2_obj[attr_variable] = ModuleLearner("emb", module=classifier, device=device)
            else:
                classifier = torch.nn.Linear(1024, 2).to(device)
                classifiers[attr_name] = classifier
                object[attr_variable] = ModuleLearner("emb", module=classifier, device=device)
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
                                                      module=InternVL(model_path=MODEL_PATH, device=device, 
                                                                      relation=1, attr=attr_name), device=device)

    _models['classifiers'] = classifiers

    # Compile dataset
    graph.constraint['label'] = ReaderSensor(keyword='logic_label', label=True)
    train_dataset = graph.compile_executable(dataset, logic_keyword='logic_str', 
                                             logic_label_keyword='logic_label')

    poi = [image, object, *attribute_names_dict.values(), graph.constraint, relaton_2_obj]
    
    # Use BCELoss for constraint satisfaction
    import torch.nn as nn
    loss_func = nn.BCELoss
    
    program = InferenceProgramWithCallbacks(
        graph, SolverModel,
        loss=loss_func,
        poi=poi,
        device=device,
        tnorm=args.tnorm,
        use_gumbel=args.use_gumbel,
        initial_temp=args.gumbel_temp_start,
        final_temp=args.gumbel_temp_end,
        anneal_start_epoch=args.gumbel_anneal_start,
        anneal_epochs=args.epochs - args.gumbel_anneal_start,
        hard_gumbel=args.hard_gumbel,
    )

    dev_dataset = None
    if dev is not None and len(dev) > 0:
        dev_dataset = graph.compile_executable(dev, logic_keyword='logic_str', 
                                               logic_label_keyword='logic_label')

    return program, train_dataset, dev_dataset


def log_training_config(args, models=None, train=None, dev=None, test=None, plugin_manager=None):
    """Log all training configuration parameters."""
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    
    print("\n[Data]")
    print(f"  Question type:    {args.question_type}")
    print(f"  Train size:       {args.train_size if args.train_size else 'all'}")
    print(f"  Test size:        {args.test_size if args.test_size else 'all'}")
    if train is not None:
        print(f"  Train examples:   {len(train)}")
    if test is not None:
        print(f"  Test examples:    {len(test)}")
    
    print("\n[Training]")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  Device:           {args.device}")
    print(f"  Dummy mode:       {args.dummy}")
    
    print("\n[Model]")
    print(f"  Oracle mode:      {args.oracle_mode}")
    print(f"  Use VLM:          {args.use_vlm}")
    if models and not args.oracle_mode and not args.use_vlm:
        total_params = 0
        for name, model in models.items():
            if name == 'classifiers':
                # classifiers is a dict of individual classifiers
                for clf_name, clf in model.items():
                    total_params += sum(p.numel() for p in clf.parameters())
            else:
                # Other models are nn.Module instances
                total_params += sum(p.numel() for p in model.parameters())
        print(f"  Total params:     {total_params:,}")
    
    print("\n[Constraints]")
    print(f"  T-norm:           {args.tnorm}")
    
    print("\n[Gumbel-Softmax]")
    if args.use_gumbel:
        print(f"  Enabled:          Yes")
        print(f"  Initial temp:     {args.gumbel_temp_start}")
        print(f"  Final temp:       {args.gumbel_temp_end}")
        print(f"  Anneal start:     Epoch {args.gumbel_anneal_start}")
        print(f"  Hard Gumbel:      {args.hard_gumbel}")
    else:
        print(f"  Enabled:          No")
    
    if plugin_manager:
        plugin_manager.log_all_configs(args)
    
    print("\n[Mode]")
    print(f"  Evaluate only:    {args.eval_only}")
    print(f"  Infer only:       {args.infer_only}")
    print(f"  Load previous:    {args.load_previous_save}")
    
    print("\n" + "=" * 60 + "\n")


def main(args):
    global _models
    
    CACHE_DIR = preprocess_folders_and_files(args.dummy)
    NUM_INSTANCES = 20
    device = args.device

    # Load dataset
    dataset = preprocess_dataset(args, NUM_INSTANCES, CACHE_DIR, question_type=args.question_type)
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Question type: {args.question_type}")

    # Print samples
    if len(dataset) > 0:
        for i in range(min(3, len(dataset))):
            print(f"\nSample question: {dataset[i].get('question_raw', '')}")
            print(f"Sample answer: {dataset[i].get('answer', '')}")

    # Create plugin manager (without BERT unfreezing)
    from domiknows.program.plugins.callback_plugin_manager import CallbackPluginManager
    from domiknows.program.plugins.epoch_logging_plugin import EpochLoggingPlugin
    from domiknows.program.plugins.adaptive_tnorm_plugin import AdaptiveTNormPlugin
    from domiknows.program.plugins.gradient_flow_plugin import GradientFlowPlugin
    from domiknows.program.plugins.counting_schedule_plugin import CountingSchedulePlugin
    from domiknows.program.plugins.gumbel_monitoring_plugin import GumbelMonitoringPlugin
    
    plugin_manager = CallbackPluginManager()
    plugin_manager.register(EpochLoggingPlugin(), 'EpochLogging')
    plugin_manager.register(AdaptiveTNormPlugin(), 'AdaptiveTNorm')
    plugin_manager.register(GradientFlowPlugin(), 'GradientFlow')
    plugin_manager.register(CountingSchedulePlugin(), 'CountingSchedule')
    plugin_manager.register(GumbelMonitoringPlugin(), 'GumbelMonitoring')
    
    # Create program
    program, train_dataset, dev_dataset = program_declaration(
        dataset if not args.eval_only else dataset, 
        None,
        args, 
        device=device
    )
    
    # Log configuration
    log_training_config(args, _models, train=dataset, dev=None, test=dataset,
                       plugin_manager=plugin_manager)
    
    save_file = ckpt_path(args.lr, 1, args.load_epoch, args.batch_size, args.tnorm, 
                         args.subset, args.question_type)

    if args.infer_only:
        acc = program.evaluate_condition(train_dataset, device=device)
        print(f"Accuracy on Test: {acc * 100:.2f}%")
        with open('results.txt', 'a') as f:
            print(save_file, file=f)
            print(f"Question type: {args.question_type}", file=f)
            print(f"Accuracy: {acc * 100:.2f}%", file=f)
    else:
        if not args.eval_only:
            # Configure plugins (no BERT-specific optimizer factory needed)
            if not args.oracle_mode and not args.use_vlm:
                plugin_manager.configure_all(
                    program=program,
                    models=_models,
                    args=args,
                    dataset=train_dataset
                )
                
                Optim = torch.optim.Adam
            else:
                Optim = torch.optim.Adam
            
            # Load previous checkpoint if needed
            if args.load_previous_save and args.subset > 1:
                previous_save = ckpt_path(args.lr, 1, args.load_epoch, args.batch_size, 
                                         args.tnorm, args.subset - 1, args.question_type)
                if previous_save.exists():
                    program.load(previous_save)
            elif args.load_previous_save and args.load_epoch > 0:
                previous_save = ckpt_path(args.lr, 1, args.load_epoch - 1, args.batch_size, 
                                         args.tnorm, args.subset, args.question_type)
                if previous_save.exists():
                    program.load(previous_save)

            # Training loop
            for i in range(args.epochs):
                print(f"Training epoch {i + 1}/{args.epochs}")
                save_file = ckpt_path(args.lr, i + 1, args.load_epoch, args.batch_size, 
                                     args.tnorm, args.subset, args.question_type)
                program.train(train_dataset, Optim=Optim, train_epoch_num=1, c_lr=args.lr,
                              c_warmup_iters=0, batch_size=args.batch_size, device=device, 
                              print_loss=False)
                program.save(save_file)
                print(f"Saved to {save_file}")
            
            # Final evaluation
            final_eval = program.evaluate_condition(train_dataset, device=device, 
                                                   threshold=0.5, return_dict=True)
            
            # Display plugin summaries
            plugin_manager.final_display_all(final_eval=final_eval)
            
        else:
            # Evaluation only
            epoch_to_eval = args.epochs if args.epochs > 0 else 1
            save_file = ckpt_path(args.lr, epoch_to_eval, args.load_epoch, args.batch_size, 
                                 args.tnorm, args.subset, args.question_type)
            if save_file.exists():
                print(f"Loading from {save_file}")
                program.load(save_file)
                acc = program.evaluate_condition(train_dataset, device=device)
                print(f"Accuracy on Test: {acc * 100:.2f}%")

                with open('results.txt', 'a') as f:
                    print(save_file, file=f)
                    print(f"Question type: {args.question_type}", file=f)
                    print(f"Accuracy: {acc * 100:.2f}%", file=f)
            else:
                print(f"Checkpoint not found: {save_file}")

    return 0


if __name__ == '__main__':
    args = parse_arguments()
    main(args)