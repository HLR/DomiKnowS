import sys, os
# Note: CUDA_VISIBLE_DEVICES can be set via environment variable before running
# e.g., CUDA_VISIBLE_DEVICES=1 python main.py
# Parse --gpu early (before torch import) so CUDA device is selected correctly.
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
        break
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')
sys.path.append('./')
from pathlib import Path
import argparse, torch, logging
from datetime import datetime


class _TeeWriter:
    """Write to terminal and a log file at the same time."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def isatty(self):
        return self._stream.isatty() if hasattr(self._stream, "isatty") else False


def setup_console_log():
    """Mirror stdout/stderr to logs/console.log for the current run only."""
    run_dir = Path(__file__).resolve().parent
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "console.log"

    # Always start a fresh console log for each run.
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write(f"Session started: {datetime.now().isoformat(timespec='seconds')}\n")
    log_file.write("=" * 80 + "\n")
    log_file.flush()

    sys.stdout = _TeeWriter(sys.stdout, log_file)
    sys.stderr = _TeeWriter(sys.stderr, log_file)

try:
    from monitor.constraint_monitor import (# type: ignore
        next_step,enable_monitoring, start_new_epoch, finish_experiment, disable_monitoring
    )
    MONITORING_AVAILABLE = True

    # Monitoring is OFF by default. Set CLEVR_ENABLE_MONITOR=1 to opt in.
    if (MONITORING_AVAILABLE
            and '--help' not in sys.argv
            and '-h' not in sys.argv
            and os.environ.get("CLEVR_ENABLE_MONITOR") == "1"):
        enable_monitoring(slave_mode=True, master_url="http://localhost:8080")
        #enable_monitoring(port=8080, slave_mode=False)  # Master mode with web server
except ImportError:
    MONITORING_AVAILABLE = False

from domiknows import setProductionLogMode, setup_step_notebook, StepNotebook

from domiknows.program import CallbackProgram
from domiknows.program.lossprogram import InferenceProgram
from domiknows.program.model.pytorch import SolverModel
import torch.nn as nn
from domiknows.sensor.pytorch import EdgeSensor, ModuleLearner
from domiknows.sensor.pytorch.sensors import ReaderSensor, FunctionalSensor, FunctionalReaderSensor, ModuleSensor
from domiknows.sensor.pytorch.relation_sensors import CompositionCandidateSensor

from domiknows.program.plugins.grad_chain_diagnostic import GradChainDiagnostic

try:
    from .preprocess import preprocess_dataset, preprocess_folders_and_files, load_full_dataset
    from .graph import create_graph
    from .modules import LEFTObjectEMB, LEFTRelationEMB, ResnetLEFT, LinearLayer, boxes_in_backbone_frame
    from .dataset import g_relational_concepts, g_attribute_concepts
except ImportError:
    from preprocess import preprocess_dataset, preprocess_folders_and_files, load_full_dataset
    from graph import create_graph
    from modules import LEFTObjectEMB, LEFTRelationEMB, ResnetLEFT, LinearLayer, boxes_in_backbone_frame
    from dataset import g_relational_concepts, g_attribute_concepts

RUN_DIR = Path(__file__).parent.resolve()
MODEL_DIR = RUN_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

_models = {}


def ckpt_path(lr, epoch_idx, load_epoch_tag, batch, tnorm, subset, q_type="relation",
              lora_r=4, softmax_temp=1.0, max_objects=None, exp_tag=None):
    if exp_tag:
        return MODEL_DIR / f"program_{q_type}_{exp_tag}_e{epoch_idx}.pth"
    mo = f"_mo{max_objects}" if max_objects else ""
    return MODEL_DIR / (
        f"program_{q_type}_{lr}_{epoch_idx}_{load_epoch_tag}__{batch}_6000_{tnorm}_{subset}"
        f"_r{lora_r}_t{softmax_temp}{mo}.pth"
    )


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


class _CallbacksMixin:
    """Shared callback hook setup for callback-enabled programs."""
    
    def _init_callback_hooks(self):
        # Initialize all callback hooks
        self.after_train_step = [self.default_after_train_step]
        self.before_train = []
        self.after_train = []
        self.before_train_epoch = []
        if MONITORING_AVAILABLE:
            self.before_train_epoch.append(start_new_epoch)
        self.after_train_epoch = []
        self.before_train_step = []
        if MONITORING_AVAILABLE:
            self.before_train_step.append(next_step)
        self.before_test = []
        self.after_test = []
        self.before_test_epoch = []
        self.after_test_epoch = []
        self.before_test_step = []
        self.after_test_step = []


from domiknows.program.lossprogram import SemanticLossProgram


class SemanticLossProgramWithCallbacks(_CallbacksMixin, CallbackProgram, SemanticLossProgram):
    """Exact semantic loss (-log P(satisfied) via circuits) with the Clever callbacks."""
    # The constraint-accuracy evaluator is defined on InferenceProgram but only
    # needs populate() and the graph; reuse it unchanged.
    evaluate_condition = InferenceProgram.evaluate_condition

    def default_after_train_step(self, output=None):
        """No-op: the loss program's epoch already runs backward and step."""
        pass

    def __init__(self, graph, Model, **kwargs):
        super().__init__(graph, Model, **kwargs)
        self._init_callback_hooks()


class InferenceProgramWithCallbacks(_CallbacksMixin, CallbackProgram, InferenceProgram):
    """InferenceProgram with callback support."""
    
    def default_after_train_step(self, output=None):
        """Override to do nothing - InferenceProgram already handles backward."""
        pass
    
    def __init__(self, graph, Model, loss=None, **kwargs):
        """Initialize callback-enabled InferenceProgram."""
        super().__init__(graph, Model, loss=loss, **kwargs)
        self._init_callback_hooks()


class _RunningTrainAccTracker:
    """Accumulate constraint-verify accuracy on each training step's datanode.
    Eliminates the separate per-epoch train-eval pass — training already forwarded
    the data, we just verify the result post-step instead of rerunning forward.
    Also logs per-step loss and per-epoch train acc to a TensorBoard writer if set."""

    def __init__(self, tb_writer=None):
        self.correct = 0
        self.total = 0
        self.tb_writer = tb_writer
        self.global_step = 0
        self.epoch = 0

    def reset(self, *_args, **_kwargs):
        self.correct = 0
        self.total = 0

    def after_step(self, output=None, **_kwargs):
        if not output or len(output) < 3:
            return
        # Log step loss to TB regardless of verify success.
        loss = output[0]
        if self.tb_writer is not None and torch.is_tensor(loss):
            try:
                self.tb_writer.add_scalar("train/step_loss", float(loss.detach()), self.global_step)
            except Exception:
                pass
            self.global_step += 1
        datanode = output[2]
        if datanode is None:
            return
        try:
            active = datanode.getActiveExecutableConstraintNames()
        except Exception:
            return
        for lc_name in active:
            try:
                label = datanode.getExecutableConstraintLabel(lc_name)
                if label is None:
                    continue
                result = datanode.verifySingleConstraint(lc_name, key="/local/argmax")
                if result is None:
                    continue
                is_satisfied = result["satisfied"] == 100.0
                expected = int(label.item() if torch.is_tensor(label) else label) == 1
                if is_satisfied == expected:
                    self.correct += 1
                self.total += 1
            except Exception:
                continue

    def report_and_reset(self, *_args, **_kwargs):
        if self.total > 0:
            acc = 100.0 * self.correct / self.total
            print(f"[running-train-acc] {self.correct}/{self.total} = {acc:.2f}%")
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("train/acc", acc, self.epoch)
        self.epoch += 1
        self.reset()


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


QUESTION_TYPE_CHOICES = (
    "relation",
    "query",
    "query_relation",
    "exist",
    "complex_relation",
    "counting",
)

QUESTION_TYPE_QUERY_MODES = {"query", "query_relation"}


def _parse_question_type_arg(value):
    raw = str(value).replace(",", "+")
    parts = []
    seen = set()
    for token in raw.split("+"):
        normalized = token.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        parts.append(normalized)

    if not parts:
        raise argparse.ArgumentTypeError("question-type must contain at least one value")

    invalid = [q for q in parts if q not in QUESTION_TYPE_CHOICES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unsupported question type(s): {invalid}. Supported: {QUESTION_TYPE_CHOICES}"
        )

    return "+".join(parts)


def _split_question_types(question_type_value):
    return [q for q in str(question_type_value).split("+") if q]


def _includes_query_type(question_type_value):
    return any(q in QUESTION_TYPE_QUERY_MODES for q in _split_question_types(question_type_value))


# LEFT-style curriculum schedule: (start_epoch, max_scene_size, max_program_size)
# Active bucket is selected by: start < epoch <= next_start
CURRICULUM_STRATEGY = [
    (0, 3, 3),    # scene≤3 AND program≤3 — simplest scene, simplest existL
    (5, 3, 4),
    (10, 3, 6),
    (15, 4, 8),
    (25, 4, 12),
    (40, 5, 16),
    (60, 7, 20),
    (80, 9, 25),
    (10**9, None, None),
]


def _scene_size(sample):
    objs = sample.get("all_objects")
    if isinstance(objs, (list, tuple)):
        return len(objs)
    objs_raw = sample.get("objects_raw")
    if isinstance(objs_raw, (list, tuple)):
        return len(objs_raw)
    return 0


def _program_size(sample):
    prog = sample.get("program")
    if isinstance(prog, (list, tuple)):
        return len(prog)
    return 0


def _num_variables(sample):
    """Logical variables of a 3D-FORCE question (lambda binders of its program)."""
    import re
    prog = sample.get("program_str")
    return len(re.findall(r"lambda\s+\w+\s*:", prog)) if isinstance(prog, str) else 0


def _subset_logic_dataset(logic_dataset, raw_items):
    """Build a LogicDataset restricted to ``raw_items`` without recompiling.

    The program's models snapshot the graph's properties/sensors and the loss
    model's constraint registry at construction time, so constraints compiled
    afterwards (via ``graph.compile_executable``) never receive labels and are
    silently ignored during both training and evaluation. Re-using the
    already-compiled executable constraints avoids that and keeps the graph
    from growing on every curriculum stage.
    """
    from domiknows.graph.executable import LogicDataset

    index_by_id = {id(item): idx for idx, item in enumerate(logic_dataset.data)}
    indices = [index_by_id[id(item)] for item in raw_items if id(item) in index_by_id]
    if len(indices) != len(raw_items):
        raise ValueError(
            f"{len(raw_items) - len(indices)} curriculum items are not part of the "
            "compiled dataset; compile the subset before creating the program."
        )
    return LogicDataset(
        [logic_dataset.data[i] for i in indices],
        [logic_dataset.lc_name_list[i] for i in indices],
        logic_keyword=logic_dataset.logic_keyword,
        logic_label_keyword=logic_dataset.logic_label_keyword,
        vector_label_names=logic_dataset.vector_label_names,
        deduplicated=logic_dataset.deduplicated,
        concept_bindings=[logic_dataset.concept_bindings[i] for i in indices],
        parameterized=logic_dataset.parameterized,
    )


def _print_force3d_soft_acc(args, program, dataset, device, tag):
    """Puzzle-only: P(satisfied) > 0.5 accuracy plus mean P per class."""
    if getattr(args, "dataset", "clevr") != "force3d" or args.force3d_split != "puzzle" or dataset is None:
        return
    from force3d_dataset import soft_constraint_accuracy
    acc, correct, total, p_pos, p_neg = soft_constraint_accuracy(
        program, dataset, device=device, tnorm=("P" if args.tnorm in ("default", "auto") else args.tnorm))
    print(f"{tag} soft accuracy: {acc:.2f}% ({correct}/{total})  mean P | yes={p_pos:.3f} no={p_neg:.3f}")


def _print_force3d_ref_top1(args, program, dataset, device, tag):
    """REF-only: argmax-of-selection accuracy next to the built-in miotaL score."""
    if getattr(args, "dataset", "clevr") != "force3d" or args.force3d_split != "ref" or dataset is None:
        return
    from force3d_dataset import force3d_ref_top1
    with torch.no_grad():
        acc, correct, total = force3d_ref_top1(program, dataset, device=device)
    print(f"{tag} REF top-1 accuracy: {acc:.2f}% ({correct}/{total})")


def _build_optimizer_factory(args, optim_cls):
    """Optimizer factory for LearningBasedProgram.train, which calls Optim(model.parameters()).

    Binds --lr, which used to be ignored for the model (it trained at Adam's
    default 1e-3).  Optionally freezes the ROI feature extractors or gives them
    their own learning rate: at 1e-3 the 134M-parameter object projection lost
    the information it carries at initialisation (color linear probe 69% at
    init, 18% after one epoch, majority 25%).
    """
    feature_modules = [_models.get(k) for k in ("object_emb", "object_fc", "relation_emb")]
    feature_params = {id(p): p for m in feature_modules if m is not None for p in m.parameters()}
    freeze = bool(getattr(args, "freeze_features", False))
    feature_lr = getattr(args, "feature_lr", None)
    if freeze:
        for p in feature_params.values():
            p.requires_grad_(False)
    n_feat = sum(p.numel() for p in feature_params.values())
    print(f"[optim] lr={args.lr} | feature extractors: {n_feat / 1e6:.1f}M params, "
          f"{'frozen' if freeze else ('lr=' + str(feature_lr) if feature_lr is not None else 'lr=' + str(args.lr))}")

    def factory(params):
        params = [p for p in params if p.requires_grad]
        if freeze or feature_lr is None:
            return optim_cls(params, lr=args.lr)
        rest = [p for p in params if id(p) not in feature_params]
        feats = [p for p in params if id(p) in feature_params]
        groups = [{"params": rest, "lr": args.lr}]
        if feats:
            groups.append({"params": feats, "lr": feature_lr})
        return optim_cls(groups)

    return factory


def _init_head_prior(classifier, prior, weight_scale=0.05):
    """Start a 2-way linear head at P(true) = prior.

    Random heads output ~0.5 per predicate; an existsL over a joint table of
    thousands of rows is then saturated at 1.0 for every question and the
    label carries no signal (see the 3D-FORCE diagnosis).  Setting the bias to
    the class log-odds and shrinking the weights makes the formula start
    unsatisfied for most questions, so gradients discriminate.
    """
    import math
    prior = min(max(float(prior), 1e-4), 1 - 1e-4)
    with torch.no_grad():
        classifier.weight.mul_(weight_scale)
        classifier.bias.zero_()
        classifier.bias[1] = math.log(prior / (1.0 - prior))
    return classifier


def _prior_for_name(attr_name, args):
    """Class prior for a predicate head from the vocabulary structure."""
    for group, values in g_attribute_concepts.items():
        if attr_name in values:
            return 1.0 / max(len(values), 2)
    return getattr(args, "prior_relation", 0.25)


FORCE3D_CURRICULUM_STRATEGY = [
    # (start_epoch_exclusive, max_scene_size, max_program_size): scene-size only,
    # 3D-FORCE has no short programs.  Small scenes keep the joint tables small
    # while the predicate heads move away from their priors.
    (0, 8, None),
    (3, 10, None),
    (6, None, None),
    (10**9, None, None),
]


# --curriculum vars (3D-FORCE): stages by the number of logical variables per
# question, as (first_epoch, max_variables) with None meaning no limit.  Scene
# size does not separate difficulty here (8-12 objects everywhere); variable
# count does: a 1-variable question sends its whole gradient to that object's
# attribute heads, a 5-variable one spreads it over a joint table.
FORCE3D_VARS_SCHEDULE = [(1, 1), (3, 2), (5, 3), (7, None)]
# Items per stage (set from --train-size); None uses every eligible item.
FORCE3D_STAGE_SIZE = None


def _parse_vars_schedule(text):
    """"1:1,3:2,5:3,7:all" -> [(1, 1), (3, 2), (5, 3), (7, None)]."""
    stages = []
    for part in text.split(","):
        epoch, limit = part.split(":")
        stages.append((int(epoch), None if limit.strip() in ("all", "0") else int(limit)))
    stages.sort()
    if not stages or stages[0][0] != 1:
        raise ValueError("--curriculum-vars-schedule must start at epoch 1")
    return stages


def _vars_limit(epoch):
    limit = None
    for first_epoch, max_vars in FORCE3D_VARS_SCHEDULE:
        if first_epoch <= epoch:
            limit = max_vars
    return limit


def _vars_stage_items(items, max_vars, size):
    eligible = [s for s in items if max_vars is None or _num_variables(s) <= max_vars]
    return eligible if size is None else eligible[:size]


def _load_force3d_dataset(args, CACHE_DIR):
    """Load 3D-FORCE, swap the vocabulary, split by scene and apply size caps.

    Returns train + test samples tagged with ``force3d_role`` so the split
    block in ``main`` can separate them after the min/max-object filters.
    """
    import dataset as clevr_dataset
    from force3d_dataset import (FORCE3D_ATTRIBUTE_CONCEPTS, FORCE3D_RELATIONAL_CONCEPTS,
                                 split_by_scene)
    from preprocess import preprocess_force3d

    clevr_dataset.set_vocabulary(FORCE3D_ATTRIBUTE_CONCEPTS, FORCE3D_RELATIONAL_CONCEPTS)
    global FORCE3D_VARS_SCHEDULE, FORCE3D_STAGE_SIZE
    if args.curriculum == "vars":
        FORCE3D_VARS_SCHEDULE = _parse_vars_schedule(args.curriculum_vars_schedule)
        FORCE3D_STAGE_SIZE = args.train_size
        print("[force3d] variable-count curriculum (first epoch, max variables):",
              FORCE3D_VARS_SCHEDULE, f"| items per stage: {FORCE3D_STAGE_SIZE or 'all eligible'}")
    elif args.curriculum != "none":
        CURRICULUM_STRATEGY[:] = FORCE3D_CURRICULUM_STRATEGY
        print("[force3d] using scene-size curriculum:", FORCE3D_CURRICULUM_STRATEGY[:-1])
    samples = preprocess_force3d(args, CACHE_DIR)
    train, test = split_by_scene(samples, args.force3d_test_scenes, seed=args.force3d_split_seed)
    if args.curriculum == "vars":
        # --train-size caps each stage, not the split: keep the union of every
        # stage's items (in split order) and attach images only to those.
        pool, chosen = train[args.train_start:], {}
        for _, max_vars in FORCE3D_VARS_SCHEDULE:
            stage = _vars_stage_items(pool, max_vars, FORCE3D_STAGE_SIZE)
            chosen.update((id(s), s) for s in stage)
            print(f"[force3d] curriculum stage vars<={max_vars if max_vars is not None else 'all'}: "
                  f"{len(stage)} items")
        train = [s for s in pool if id(s) in chosen]
    elif args.train_size is not None:
        train = train[args.train_start: args.train_start + args.train_size]
    if args.test_size is not None:
        test = test[: args.test_size]
    if args.force3d_split == "puzzle" and train and test:
        from force3d_dataset import structure_only_baseline
        struct_acc, majority_acc, n_eval = structure_only_baseline(train, test)
        print(f"[force3d] structure-only baseline (question structure, no image): {struct_acc:.1f}% "
              f"held-out, majority class {majority_acc:.1f}%, n={n_eval}; a model must beat "
              f"the structure baseline, not 50%")
    for d in train:
        d["force3d_role"] = "train"
    for d in test:
        d["force3d_role"] = "test"
    from force3d_dataset import attach_images
    attach_images(train + test)
    print(f"[force3d] images attached for {len(train)} train + {len(test)} test samples")
    return train + test


def _get_curriculum_limits(epoch):
    for i in range(len(CURRICULUM_STRATEGY) - 1):
        start, max_scene_size, max_program_size = CURRICULUM_STRATEGY[i]
        next_start = CURRICULUM_STRATEGY[i + 1][0]
        if start < epoch <= next_start:
            return max_scene_size, max_program_size
    return None, None


def _select_curriculum_train_raw(train_raw, epoch, mode):
    if mode == "vars":
        max_vars = _vars_limit(epoch)
        filtered = _vars_stage_items(train_raw, max_vars, FORCE3D_STAGE_SIZE)
        limit = f"vars<={max_vars if max_vars is not None else 'all'}"
        kind = "none" if filtered else "full"
        info = {"mode": mode, "scene_limit": None, "program_limit": limit,
                "selected": len(filtered), "total": len(train_raw), "fallback": kind != "none",
                "fallback_kind": kind, "min_violation": None}
        return (filtered or train_raw), (mode, None, limit, kind), info

    max_scene_size, max_program_size = _get_curriculum_limits(epoch)

    filtered = train_raw
    scene_limit = max_scene_size if mode in ("scene", "all") else None
    program_limit = max_program_size if mode in ("program", "all") else None

    if scene_limit is not None:
        filtered = [s for s in filtered if _scene_size(s) <= scene_limit]
    if program_limit is not None:
        filtered = [s for s in filtered if _program_size(s) <= program_limit]

    fallback_kind = "none"
    min_violation = None

    if len(filtered) == 0 and mode == "all" and scene_limit is not None and program_limit is not None:
        # Relax all-mode from strict AND to OR before giving up.
        relaxed = [
            s for s in train_raw
            if _scene_size(s) <= scene_limit or _program_size(s) <= program_limit
        ]
        if len(relaxed) > 0:
            filtered = relaxed
            fallback_kind = "relaxed_or"

    if len(filtered) == 0 and len(train_raw) > 0:
        # Pick the nearest bucket by minimum limit violation.
        def _violation(sample):
            v = 0
            if scene_limit is not None:
                v += max(0, _scene_size(sample) - scene_limit)
            if program_limit is not None:
                v += max(0, _program_size(sample) - program_limit)
            return v

        min_violation = min(_violation(s) for s in train_raw)
        filtered = [s for s in train_raw if _violation(s) == min_violation]
        if len(filtered) > 0:
            fallback_kind = "nearest"

    if len(filtered) == 0:
        filtered = train_raw
        fallback_kind = "full"

    fallback = fallback_kind != "none"

    info = {
        "mode": mode,
        "scene_limit": scene_limit,
        "program_limit": program_limit,
        "selected": len(filtered),
        "total": len(train_raw),
        "fallback": fallback,
        "fallback_kind": fallback_kind,
        "min_violation": min_violation,
    }
    key = (mode, scene_limit, program_limit, fallback_kind)
    return filtered, key, info


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Logic-guided VQA training / evaluation using DomiKnows framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    parser.add_argument("--train-start", type=int, default=0,
                        help="Start index within the full dataset for the training slice "
                             "(default: 0). The training slice is "
                             "dataset[train_start : train_start + train_size].")
    parser.add_argument("--test-size", type=int, default=None,
                        help="Number of test examples to use (default: use all available)")
    parser.add_argument("--test-start", type=int, default=None,
                        help="Start index within the full dataset for the test slice "
                             "(default: None — fall back to the legacy hold-out-from-end "
                             "behavior). When set, the test slice is "
                             "dataset[test_start : test_start + test_size] drawn from the "
                             "full cached dataset, so it can be disjoint from --train-start.")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of training epochs (default: 4)")
    parser.add_argument("--lr", "--learning-rate", type=float, default=1e-3,
                        help="Learning rate for optimizer (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Mini-batch size for training (default: 1)")
    parser.add_argument("--curriculum", type=str, default="all",
                        choices=["none", "scene", "program", "all", "vars"],
                        help="Curriculum learning mode using LEFT staged limits "
                             "on scene size and/or program size; 'vars' (3D-FORCE) stages "
                             "by logical variables per question (--curriculum-vars-schedule), "
                             "with --train-size items per stage")
    parser.add_argument("--curriculum-vars-schedule", type=str, default="1:1,3:2,5:3,7:all",
                        help="--curriculum vars stages as first_epoch:max_variables "
                             "('all' = no limit)")
    parser.add_argument("--subset", type=int, default=-1,
                        help="Subset index 1-6 for memory-efficient training (default: -1)")
    parser.add_argument("--load-epoch", type=int, default=0,
                        help="Starting epoch when resuming training (default: 0)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, only evaluate a saved checkpoint")
    parser.add_argument("--dummy", action="store_true",
                        help="Use lightweight dummy mode with 20 instances for testing")
    parser.add_argument("--num-instances", type=int, default=20,
                        help="Number of instances to cache/load in dummy mode (default: 20)")
    parser.add_argument("--test-split", type=int, default=0,
                        help="Hold out last N examples as test set (default: 0 = no split)")
    parser.add_argument("--max-objects", type=int, default=None,
                        help="Filter training set to images with at most N objects")
    parser.add_argument("--min-objects", type=int, default=None,
                        help="Filter dataset to images with at least N objects")
    parser.add_argument("--load_previous_save", action="store_true",
                        help="Load checkpoint from previous subset/epoch before training")
    parser.add_argument("--question-type",
                        type=_parse_question_type_arg,
                        default="relation",
                        help="Type of questions to train on. Supports one type, or merge two "
                             "types with '+' (example: relation+exist).")
    parser.add_argument("--relation-syntax",
                        choices=["legacy", "binary"],
                        default="binary",
                        help="Relation syntax emitted by translator")
    parser.add_argument("--use-vlm", default=False, action="store_true", 
                        help="use InternVL for predictions")
    parser.add_argument("--peft", action="store_true",
                        help="Use PEFT (LoRA) fine-tuning with HuggingFace InternVL")
    parser.add_argument("--load-4bit", action="store_true",
                        help="Use QLoRA 4-bit quantization for VLM")
    parser.add_argument("--softmax-temp", type=float, default=2.0,
                        help="Temperature for VLM softmax output. >1 softens output; helps escape "
                             "cold-start saturation where baseline P(Yes)≈0.001 on CLEVR yes/no atoms. "
                             "Too high over-saturates to the opposite pole (both kill BCE gradient).")
    parser.add_argument("--yes-bias", type=float, default=0.0,
                        help="Additive bias on the Yes logit before logsumexp in _score_batch. "
                             "Use a positive value (e.g. 3.0) to counter the VLM's strong No-prior "
                             "so existsL product t-norm doesn't start fully saturated.")
    parser.add_argument("--pos-weight", type=float, default=1.0,
                        help="BCE pos_weight on the Yes (logic_label=1) class in InferenceModel. "
                             "Use >1 to rebalance against majority-class collapse.")
    parser.add_argument("--tensorboard", type=str2bool, nargs='?', const=True, default=True,
                        help="Write per-step train loss and per-epoch acc to TensorBoard. "
                             "Log dir: logs/tb/<exp_tag or timestamp>")
    parser.add_argument("--running-train-acc", type=str2bool, nargs='?', const=True, default=True,
                        help="Accumulate per-step constraint-verify accuracy during training "
                             "and print at end of each epoch. Free signal (training already "
                             "did the forward pass); replaces the costly separate train-eval.")
    parser.add_argument("--skip-train-eval", action="store_true",
                        help="Skip the per-epoch evaluation pass over the training set "
                             "(and the pre-training baseline on train). Use test set as the "
                             "only signal each epoch. Massive speedup at scale where train-eval "
                             "dominates pickle-deserialize time.")
    parser.add_argument("--simple-only", action="store_true",
                        help="Pre-filter dataset to samples with scene_size <= 3 AND "
                             "program_size <= 3 before --train-size slicing. Use to "
                             "guarantee the training set consists of the simplest existL "
                             "questions on the smallest scenes.")
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="Sanity check: randomly permute train-set answers before "
                             "logic_label is computed. If the model still overfits, "
                             "labels are leaking through the constraint loss.")
    parser.add_argument("--shuffle-seed", type=int, default=0xdeadbeef,
                        help="Seed for --shuffle-labels permutation")
    parser.add_argument("--oracle-mode", action="store_true",
                        help="Use ground truth answers instead of VLM/ResNet for debugging")
    parser.add_argument("--oracle-confidence", type=float, default=1.0,
                        help="Oracle confidence level (0.5=random, 1.0=perfect)")
    parser.add_argument("--infer-only", action="store_true", 
                        help="Skip training, only evaluate a the model as is")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU index to use (parsed early before torch import)")
    parser.add_argument("--max-num-patches", type=int, default=1,
                        help="Maximum number of image patches/tiles for InternVL")
    parser.add_argument("--lora-r", type=int, default=4,
                        help="LoRA rank for PEFT fine-tuning")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha for PEFT fine-tuning (default: 2 * lora_r)")
    parser.add_argument("--no-lora", action="store_true",
                        help="Skip LoRA application for eval-only runs")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Override model path")
    parser.add_argument("--exp-tag", type=str, default=None,
                        help="Experiment tag for checkpoint/results filenames")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for computation (default: cuda if available)")
    
    # Production logging settings
    parser.add_argument("--production-log-mode", type=str2bool, nargs='?', const=True, default=True,
                        help="Enable production logging mode (default: true)")
    parser.add_argument("--no-time-log", type=str2bool, nargs='?', const=True, default=False,
                        help="Disable regression timer log output when production mode is enabled")
    parser.add_argument("--reuse-model", type=str2bool, nargs='?', const=True, default=True,
                        help="Reuse compiled models in production mode (default: true)")
    
    # t-norm settings
    parser.add_argument("--tnorm", choices=["G", "P", "L", "SP", "default", "auto"],
                        default="G",
                        help="T-norm mode: G/P/L/SP = fixed t-norm, 'default' = per-type defaults, "
                             "'auto' = adaptive during training. Gödel ('G', default) uses min/max — "
                             "exact at any arity, focused gradient on the argmin/argmax atom. "
                             "Product ('P') stays differentiable as conversionSigmoid → 1 but its OR "
                             "shortcut is only correct for n=2. Lukasiewicz ('L') hard-clips at "
                             "min(1, Σp_i) and produces exact-zero gradient past the clip — bad for "
                             "existsL over many atoms.")
    
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
    
    # Constraint printing settings
    parser.add_argument("--print-constraints", type=str, default=None, metavar="JSON_FILE",
                        help="Load a CLEVR questions JSON (with programs), print each question "
                             "and its compiled constraint expression, then exit. "
                             "Supports standard CLEVR format or [{question, program, answer}, ...]")
    parser.add_argument("--print-limit", type=int, default=70,
                        help="Max number of questions to print with --print-constraints")
    parser.add_argument("--disable-plugins", action="store_true",
                        help="Skip all callback plugins (AdaptiveTNorm, GradientFlow, "
                             "EpochLogging, GumbelMonitoring). Useful in tests where "
                             "plugin diagnostics are noise and their extra forward "
                             "passes add memory pressure.")
    parser.add_argument("--step-notebook", type=str2bool, nargs='?', const=True, default=True,
                        help="Write a per-step JSONL notebook (logs/step_notebook.jsonl) "
                             "with the question, GT answer, predicted answer and per-concept "
                             "softmax/argmax tensors for each evaluated example.")
    parser.add_argument("--step-notebook-file", type=str, default=None,
                        help="Override the step-notebook output filename "
                             "(default: step_notebook[_<exp_tag>].jsonl in logs/).")

    # Register callback plugin arguments (exclude BERT unfreezing)
    from domiknows.program.plugins.callback_plugin_manager import CallbackPluginManager
    from domiknows.program.plugins.epoch_logging_plugin import EpochLoggingPlugin
    from domiknows.program.plugins.adaptive_tnorm_plugin import AdaptiveTNormPlugin
    from domiknows.program.plugins.gradient_flow_plugin import GradientFlowPlugin
    from domiknows.program.plugins.gumbel_monitoring_plugin import GumbelMonitoringPlugin
    
    plugin_manager = CallbackPluginManager()
    plugin_manager.register(EpochLoggingPlugin(), 'EpochLogging')
    plugin_manager.register(AdaptiveTNormPlugin(), 'AdaptiveTNorm')
    plugin_manager.register(GradientFlowPlugin(), 'GradientFlow')
    plugin_manager.register(GumbelMonitoringPlugin(), 'GumbelMonitoring')
    
    # ---- 3D-FORCE adapter (see force3d_dataset.py) ----
    parser.add_argument("--dataset", choices=["clevr", "force3d"], default="clevr",
                        help="Dataset adapter: CLEVR (default) or 3D-FORCE via force3d_dataset.py")
    parser.add_argument("--force3d-root", type=str,
                        default="/localscratch/kamalida/projects/SaPy/datasets/3D-FORCE",
                        help="3D-FORCE root containing 3DForcePuzzle.json/3DForceRef.json and multiview/")
    parser.add_argument("--force3d-split", choices=["puzzle", "ref"], default="puzzle",
                        help="3D-FORCE split: puzzle (yes/no existsL) or ref (object selection via miotaL)")
    parser.add_argument("--force3d-json", type=str, default=None,
                        help="Question file to load instead of the released split (e.g. output of "
                             "gen_force3d_free.py); relative paths resolve under --force3d-root.")
    parser.add_argument("--force3d-test-scenes", type=int, default=20,
                        help="Number of whole scenes held out as the 3D-FORCE test set")
    parser.add_argument("--force3d-split-seed", type=int, default=0,
                        help="Seed for the 3D-FORCE scene split")
    parser.add_argument("--force3d-view", choices=["first"], default="first",
                        help="Which view to feed the model (only 'first' = camera 0 for now)")
    parser.add_argument("--program", choices=["inference", "semantic"], default="inference",
                        help="Training objective: t-norm constraint loss (inference, default) or exact "
                             "circuit semantic loss -log P(satisfied) (semantic).")
    parser.add_argument("--circuit-backend", default="bdd", help="Semantic-loss circuit backend")
    parser.add_argument("--circuit-max-nodes", type=int, default=200000)
    parser.add_argument("--circuit-size-limit-action", default="raise")
    parser.add_argument("--freeze-features", action="store_true",
                        help="Freeze the ROI object/relation embedding layers and train only the "
                             "predicate heads (the pretrained backbone is never trained).")
    parser.add_argument("--feature-lr", type=float, default=None,
                        help="Learning rate for the ROI object/relation embedding layers "
                             "(default: --lr). Their 134M-parameter projections lose their "
                             "information at 1e-3.")
    parser.add_argument("--backbone-size", type=int, default=224,
                        help="ResNet input resolution (square). 448 doubles the ROI feature map; "
                             "on 3D-FORCE it lifts shape and object-heading probes substantially.")
    parser.add_argument("--init-prior", action="store_true",
                        help="Initialise each predicate head at its class prior (1/#values for "
                             "attributes, --prior-relation for relations) instead of ~0.5, so "
                             "existsL over large joint tables is not saturated at start.")
    parser.add_argument("--prior-relation", type=float, default=0.25,
                        help="Prior P(true) used by --init-prior for relation heads.")
    parser.add_argument("--infer-type", choices=["ILP", "local"], default="ILP",
                        help="Inference run by the model on every populate: 'ILP' (default, Gurobi) or "
                             "'local' (local/argmax only; ~2-10x faster evaluation, identical accuracy "
                             "numbers since the metric reads local argmax)")
    plugin_manager.add_arguments_to_parser(parser)
    
    args = parser.parse_args()
    args.question_type = _parse_question_type_arg(args.question_type)
    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r
    if args.peft:
        args.use_vlm = True
    return args


def program_declaration(train, dev, args, device='cpu'):
    """Create and configure the DomiKnows program with sensors and learners."""
    global _models
    
    if args.use_vlm and not args.oracle_mode:
        if args.peft:
            from peftvllm import InternVLSharedHF as InternVL
        else:
            from internVLvLLM import InternVLShared as InternVL

    def filter_relation(property=None, arg1=None, arg2=None, **kwargs):
        """Filter function for relation sensors. Flexible parameter handling for forward/reverse pairs."""
        # Try explicit parameters first (forward pair: arg1, arg2)
        if arg1 is not None and arg2 is not None:
            return arg1.getAttribute("image_id") == arg2.getAttribute("image_id")
        
        # Fall back to extracting from kwargs (reverse pair variants)
        remaining = [v for v in kwargs.values() if v is not None and hasattr(v, 'getAttribute')]
        if len(remaining) >= 2:
            return remaining[0].getAttribute("image_id") == remaining[1].getAttribute("image_id")
        
        return True

    # Build graph/logic over the full set used in this run so both train/dev can compile.
    dataset = train + (dev if dev is not None else [])
    include_query = _includes_query_type(args.question_type)
    results = create_graph(
        dataset,
        include_query_questions=include_query,
        relation_syntax=args.relation_syntax,
    )

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
    obj1_rev = results[10] if len(results) > 10 else None
    obj2_rev = results[11] if len(results) > 11 else None
    relation_2_obj_rev = results[12] if len(results) > 12 else None

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
        elif torch.is_tensor(dataset[i].get("logic_label")):
            # Precomputed vector label (3D-FORCE REF one-hot for miotaL).
            dataset[i]["logic_label"] = dataset[i]["logic_label"].to(device)
            dataset[i]["query_type"] = None
        else:
            dataset[i]["logic_label"] = torch.LongTensor([bool(dataset[i]['answer'])]).to(device)
            dataset[i]["query_type"] = None

    # Boxes for the ResNet path, in the backbone's input frame.  ResnetLEFT
    # resizes every image to BACKBONE_INPUT_SIZE squared, so raw pixel boxes
    # (objects_raw) must be rescaled per axis.  Feeding them unscaled pooled
    # features from the wrong region: 60% of 3D-FORCE objects had zero overlap
    # with their box and 59% of CLEVR object centres fell outside the map.
    # objects_raw itself stays in pixels for the VLM and oracle paths.
    for i in range(len(dataset)):
        dataset[i]["objects_backbone"] = boxes_in_backbone_frame(
            dataset[i].get("objects_raw"), dataset[i].get("pil_image"))

    # Pre-compute oracle ground truth
    if args.oracle_mode:
        import math
        oracle_conf = args.oracle_confidence
        if oracle_conf >= 0.999:
            oracle_logit = 100
        elif oracle_conf <= 0.001:
            oracle_logit = -100
        else:
            oracle_logit = math.log(oracle_conf / (1.0 - oracle_conf))

        spatial_list = g_relational_concepts.get("spatial_relation", [])
        _base_opposite = {"left": "right", "right": "left", "front": "behind", "behind": "front"}
        inverse_for_reverse = {}
        for _rel in spatial_list:
            if _rel in _base_opposite:
                inverse_for_reverse[_rel] = _base_opposite[_rel]
            else:
                from force3d_dataset import opposite_relation
                inverse_for_reverse[_rel] = opposite_relation(_rel)
        for i in range(len(dataset)):
            all_objs = dataset[i].get('all_objects', [])
            # Use bounding-box count (objects_raw) as the authoritative object
            # count.  Relation datanodes are created from bounding boxes, so the
            # oracle tensors must have the same n*n length.  all_objects comes
            # from scene['objects'] (GT) while objects_raw may come from
            # scene['objects_detection'] — they can differ.
            objects_raw = dataset[i].get('objects_raw', None)
            if objects_raw is not None:
                n = len(objects_raw)
            else:
                n = len(all_objs)
            gt_spatial = dataset[i].get('relation_spatial_relation', None)

            if gt_spatial is not None:
                for s_idx, s_name in enumerate(spatial_list):
                    oracle_data = []
                    for pair_idx in range(n * n):
                        if pair_idx < len(gt_spatial) and gt_spatial[pair_idx][s_idx] > 0.5:
                            oracle_data.append([0, oracle_logit])
                        else:
                            oracle_data.append([oracle_logit, 0])
                    dataset[i][f"oracle_is_{s_name}"] = oracle_data

                # Reverse labels are derived from inverse forward relations.
                # Example: left_rev(a,b) uses right(a,b) truth values.
                for rev_name, src_name in inverse_for_reverse.items():
                    src_key = f"oracle_is_{src_name}"
                    if src_key in dataset[i]:
                        dataset[i][f"oracle_is_{rev_name}_rev"] = list(dataset[i][src_key])

            for attr in list(g_attribute_concepts.keys()):
                oracle_data = []
                for obj_i in range(n):
                    for obj_j in range(n):
                        if obj_i < len(all_objs) and obj_j < len(all_objs) and \
                                all_objs[obj_i].get(attr) == all_objs[obj_j].get(attr):
                            oracle_data.append([0, oracle_logit])
                        else:
                            oracle_data.append([oracle_logit, 0])
                dataset[i][f"oracle_is_same_{attr}"] = oracle_data

    # Set up sensors - shared across all modes
    image["pil_image"] = FunctionalReaderSensor(keyword="pil_image", forward=lambda data: [data])
    # image_filename is used by VLM modules as a fallback: if pil_image is None
    # (stale dataset cache built before images were downloaded), the module can
    # load the image directly from train/images/<filename> on demand.
    image["image_filename"] = FunctionalReaderSensor(keyword="image_filename", forward=lambda data: [data])
    image["image_id"] = FunctionalReaderSensor(keyword='image_index', forward=lambda data: [data])
    object["bounding_boxes"] = FunctionalReaderSensor(keyword="objects_raw",
                                                      forward=lambda data: torch.Tensor(data).to(device))
    # Same boxes rescaled into the ResNet backbone frame (used for ROI pooling).
    object["backbone_boxes"] = FunctionalReaderSensor(keyword="objects_backbone",
                                                      forward=lambda data: torch.Tensor(data).to(device))
    object["properties"] = ReaderSensor(keyword="all_objects")
    object["image_id"] = FunctionalSensor(image["image_id"], "bounding_boxes",
                                          forward=lambda data, data2: data * len(data2))
    
    # Mode-specific embeddings
    if not args.use_vlm and not args.oracle_mode:
        resnet_model = ResnetLEFT(device=device)
        image["emb"] = ModuleSensor("image_id", "pil_image", module=resnet_model, device=device)
        object_feature_extraction_model = LEFTObjectEMB(device=device)
        object["feature_emb"] = ModuleLearner(image["emb"], "backbone_boxes", 
                                              module=object_feature_extraction_model, device=device)
        object_feature_fc = LinearLayer(128 * 32 * 32, 1024, device=device)
        object["emb"] = ModuleLearner("feature_emb", "backbone_boxes", 
                                      module=object_feature_fc, device=device)
        _models['resnet'] = resnet_model
        _models['object_emb'] = object_feature_extraction_model
        _models['object_fc'] = object_feature_fc

    object[image_object_contains] = EdgeSensor(object["bounding_boxes"], image["pil_image"],
                                               relation=image_object_contains,
                                               forward=lambda b, _: torch.ones(len(b)).unsqueeze(-1))

    relaton_2_obj[obj1.reversed, obj2.reversed] = CompositionCandidateSensor(
        object['image_id'], relations=(obj1.reversed, obj2.reversed), forward=filter_relation)
    if relation_2_obj_rev is not None and obj1_rev is not None and obj2_rev is not None:
        relation_2_obj_rev[obj1_rev.reversed, obj2_rev.reversed] = CompositionCandidateSensor(
            object['image_id'], relations=(obj1_rev.reversed, obj2_rev.reversed), forward=filter_relation)
    
    if not args.use_vlm and not args.oracle_mode:
        object_relation_extraction = LEFTRelationEMB(input_size=256, output_size=1024, device=device)
        relaton_2_obj["emb"] = ModuleLearner(image["emb"], object["backbone_boxes"], 
                                             object["feature_emb"],
                                             module=object_relation_extraction, device=device)
        if relation_2_obj_rev is not None:
            relation_2_obj_rev["emb"] = ModuleLearner(
                image["emb"],
                object["backbone_boxes"],
                object["feature_emb"],
                module=object_relation_extraction,
                device=device,
            )
        _models['relation_emb'] = object_relation_extraction

    # Set up learners for attributes and relations
    spatial_relations = g_relational_concepts.get("spatial_relation", [])
    spatial_relations_rev = [f"{name}_rev" for name in spatial_relations]
    spatial_relation_names = set(spatial_relations + spatial_relations_rev)
    classifiers = {}

    for attr_name, attr_variable in attribute_names_dict.items():
        relation_target = relaton_2_obj
        if attr_name in spatial_relations_rev and relation_2_obj_rev is not None:
            relation_target = relation_2_obj_rev

        if attr_name in ("distinct", "distinct_rev"):
            # Fixed identity relation (3D-FORCE adapter): distinct(i, j) iff i != j.
            # Never learned; the loader provides the logits in every mode.
            relation_target[f"{attr_variable}_label"] = FunctionalReaderSensor(
                keyword=f"oracle_is_{attr_name}",
                forward=lambda data: torch.Tensor(data).to(device))
            relation_target[attr_variable] = ModuleLearner(
                f"{attr_name}_label", module=OracleDummyLearner(), device=device)
            continue

        if args.oracle_mode:
            if attr_name in spatial_relation_names:
                relation_target[f"{attr_variable}_label"] = FunctionalReaderSensor(
                    keyword=f"oracle_is_{attr_name}",
                    forward=lambda data: torch.Tensor(data).to(device))
                relation_target[attr_variable] = ModuleLearner(
                    f"{attr_name}_label", module=OracleDummyLearner(), device=device)
            elif attr_name.startswith("same_"):
                relation_target[f"{attr_variable}_label"] = FunctionalReaderSensor(
                    keyword=f"oracle_is_{attr_name}",
                    forward=lambda data: torch.Tensor(data).to(device))
                relation_target[attr_variable] = ModuleLearner(
                    f"{attr_name}_label", module=OracleDummyLearner(), device=device)
            else:
                object[attr_variable] = ModuleLearner(
                    object["properties"], object["bounding_boxes"],
                    module=OracleModule(attr_name, relation=1, device=device,
                                        confidence=args.oracle_confidence), device=device)
        elif not args.use_vlm:
            if attr_name in spatial_relation_names:
                classifier = torch.nn.Linear(1024, 2).to(device)
                if getattr(args, "init_prior", False):
                    _init_head_prior(classifier, _prior_for_name(attr_name, args))
                classifiers[attr_name] = classifier
                relation_target[attr_variable] = ModuleLearner("emb", module=classifier, device=device)
            elif attr_name.startswith("same_"):
                classifier = torch.nn.Linear(1024, 2).to(device)
                if getattr(args, "init_prior", False):
                    _init_head_prior(classifier, _prior_for_name(attr_name, args))
                classifiers[attr_name] = classifier
                relation_target[attr_variable] = ModuleLearner("emb", module=classifier, device=device)
            else:
                classifier = torch.nn.Linear(1024, 2).to(device)
                if getattr(args, "init_prior", False):
                    _init_head_prior(classifier, _prior_for_name(attr_name, args))
                classifiers[attr_name] = classifier
                object[attr_variable] = ModuleLearner("emb", module=classifier, device=device)
        else:
            MODEL_PATH = args.model_path or ("OpenGVLab/InternVL3_5-1B" if args.peft else "OpenGVLab/InternVL3_5-8B")
            vlm_extra = dict(
                use_llm_lora=not args.no_lora,
                use_vision_lora=False,
                load_4bit=args.load_4bit,
                softmax_temperature=args.softmax_temp,
                yes_bias=args.yes_bias,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                max_num=args.max_num_patches,
            ) if args.peft else {}
            if attr_name in spatial_relation_names:
                relation_target[attr_variable] = ModuleLearner(
                    image["pil_image"], image["image_filename"], object["bounding_boxes"],
                    module=InternVL(model_path=MODEL_PATH, device=device,
                                    relation=2, attr=attr_name,
                                    **vlm_extra), device=device)
            elif attr_name.startswith("same_"):
                relation_target[attr_variable] = ModuleLearner(
                    image["pil_image"], image["image_filename"], object["bounding_boxes"],
                    module=InternVL(model_path=MODEL_PATH, device=device,
                                    relation=2, attr=attr_name,
                                    **vlm_extra), device=device)
            else:
                object[attr_variable] = ModuleLearner(
                    image["pil_image"], image["image_filename"], object["bounding_boxes"],
                    module=InternVL(model_path=MODEL_PATH, device=device,
                                    relation=1, attr=attr_name,
                                    **vlm_extra), device=device)

    _models['classifiers'] = classifiers

    # Compile dataset — both train AND dev must be compiled BEFORE the
    # program is created, 
    graph.constraint['label'] = ReaderSensor(keyword='logic_label', label=True)
    train_dataset = graph.compile_executable(train, logic_keyword='logic_str',
                                             logic_label_keyword='logic_label',
                                             extra_namespace_values=attribute_names_dict)
    dev_dataset = None
    if dev is not None and len(dev) > 0:
        dev_dataset = graph.compile_executable(dev, logic_keyword='logic_str',
                                               logic_label_keyword='logic_label',
                                               extra_namespace_values=attribute_names_dict)

    # Diagnostic: verify executable constraints were registered
    print(f"[graph] Total dataset size (train + dev): {len(dataset)}")
    n_elc = len(getattr(graph, 'executableLCs', {}))
    n_glc = len(getattr(graph, 'logicalConstrains', {}))
    print(f"[graph] Executable constraints (per-sample): {n_elc}")
    print(f"[graph] Global constraints (graph-level):    {n_glc}")
    if n_elc == 0:
        # Check if logic_str is actually present in the data
        sample = train[0] if train else {}
        has_key = 'logic_str' in sample
        val = sample.get('logic_str', '<MISSING>')
        print(f"[graph] WARNING: 0 executable constraints registered!")
        print(f"[graph]   train[0] has 'logic_str': {has_key}")
        print(f"[graph]   train[0]['logic_str'] = {val!r}")
        has_label = 'logic_label' in sample
        print(f"[graph]   train[0] has 'logic_label': {has_label}")

    poi = [image, object, *attribute_names_dict.values(), graph.constraint, relaton_2_obj]
    if relation_2_obj_rev is not None:
        poi.append(relation_2_obj_rev)
    
    # Use BCELoss for constraint satisfaction
    import torch.nn as nn
    loss_func = nn.BCELoss
    
    program_kwargs = {
        'loss': loss_func,
        'poi': poi,
        'device': device,
        'tnorm': args.tnorm,
        'pos_weight': args.pos_weight,
    }
    if getattr(args, "infer_type", "ILP") == "local":
        program_kwargs['inferTypes'] = ['local/argmax']
    if args.use_gumbel:
        program_kwargs.update({
            'use_gumbel': args.use_gumbel,
            'initial_temp': args.gumbel_temp_start,
            'final_temp': args.gumbel_temp_end,
            'anneal_start_epoch': args.gumbel_anneal_start,
            'anneal_epochs': args.epochs - args.gumbel_anneal_start,
            'hard_gumbel': args.hard_gumbel,
        })

    if getattr(args, "program", "inference") == "semantic":
        # Exact -log P(formula) in log space: no vanishing product and no
        # single-element Gödel gradient over large joint tables.
        program_kwargs = {k: v for k, v in program_kwargs.items()
                          if k not in ("tnorm", "pos_weight", "use_gumbel", "initial_temp", "final_temp",
                                       "anneal_start_epoch", "anneal_epochs", "hard_gumbel")}
        program_kwargs.update(circuit_backend=args.circuit_backend,
                              circuit_max_nodes=args.circuit_max_nodes,
                              circuit_size_limit_action=args.circuit_size_limit_action)
        program = SemanticLossProgramWithCallbacks(graph, SolverModel, **program_kwargs)
    else:
        program = InferenceProgramWithCallbacks(graph, SolverModel, **program_kwargs)

    return program, train_dataset, dev_dataset, attribute_names_dict


def _print_constraints_and_exit(args):
    """
    Load questions from a JSON file, compile constraint expressions,
    and print each question with its constraint string.
    """
    import json as json_io

    json_path = args.print_constraints
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json_io.load(f)

    # Accept CLEVR format {"questions": [...]} or flat list [...]
    if isinstance(raw, dict):
        entries = raw.get("questions", raw.get("data", []))
    elif isinstance(raw, list):
        entries = raw
    else:
        print(f"Unsupported JSON structure in {json_path}")
        return

    if args.print_limit is not None:
        entries = entries[: args.print_limit]

    # Build minimal dataset dicts expected by create_graph
    dataset = []
    for i, entry in enumerate(entries):
        dataset.append({
            "program": entry.get("program", []),
            "question_raw": entry.get("question", f"<question {i}>"),
            "answer": entry.get("answer", "?"),
        })

    print(f"Loaded {len(dataset)} questions from {json_path}")
    print(f"Relation syntax: {args.relation_syntax}")
    print()

    include_query = _includes_query_type(args.question_type)

    # Compile one at a time so a single unsupported op doesn't kill the run
    print("=" * 70)
    print("COMPILED CONSTRAINTS")
    print("=" * 70)

    ok_count = 0
    err_count = 0

    for i, item in enumerate(dataset):
        q = item["question_raw"]
        a = item["answer"]
        program = item.get("program", [])

        try:
            single = [item]
            results = create_graph(
                single,
                include_query_questions=include_query,
                apply_constraints=False,  # skip constraints for speed
                relation_syntax=args.relation_syntax,
            )
            exc = results[0][0]
            qt = results[9][0] if len(results) > 9 else None
            ok_count += 1
        except Exception as e:
            exc = f"<ERROR: {e}>"
            qt = None
            err_count += 1

        print(f"[{i}] Q: {q}")
        print(f"     A: {a}")
        if qt is not None:
            print(f"     query_type: {qt}")
        print(f"     Constraint: {exc}")
        print()

    print("=" * 70)
    print(f"Total: {ok_count + err_count}  |  OK: {ok_count}  |  Errors: {err_count}")
    print("=" * 70)
    
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
    print(f"  Curriculum:       {args.curriculum}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  Device:           {args.device}")
    print(f"  Dummy mode:       {args.dummy}")
    print(f"  Num instances:    {args.num_instances}")
    print(f"  Test split:       {args.test_split}")
    print(f"  Max objects:      {args.max_objects if args.max_objects is not None else 'none'}")
    print(f"  Min objects:      {args.min_objects if args.min_objects is not None else 'none'}")
    
    print("\n[Model]")
    print(f"  Oracle mode:      {args.oracle_mode}")
    print(f"  Oracle conf:      {args.oracle_confidence}")
    print(f"  Use VLM:          {args.use_vlm}")
    print(f"  PEFT mode:        {args.peft}")
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
    print(f"  Relation syntax:  {args.relation_syntax}")
    print(f"  Softmax temp:     {args.softmax_temp}")
    print(f"  Yes-bias:         {args.yes_bias}")
    print(f"  Pos-weight:       {args.pos_weight}")

    print("\n[Logging]")
    print(f"  Production mode:  {args.production_log_mode}")
    print(f"  No time log:      {args.no_time_log}")
    print(f"  Reuse model:      {args.reuse_model}")
    
    print("\n[Gumbel-Softmax]")
    if args.use_gumbel:
        print(f"  Enabled:          Yes")
        print(f"  Initial temp:     {args.gumbel_temp_start}")
        print(f"  Final temp:       {args.gumbel_temp_end}")
        print(f"  Anneal start:     Epoch {args.gumbel_anneal_start}")
        print(f"  Hard Gumbel:      {args.hard_gumbel}")
    else:
        print(f"  Enabled:          No")
    
    if plugin_manager and not getattr(args, 'disable_plugins', False):
        plugin_manager.log_all_configs(args)
    
    print("\n[Mode]")
    print(f"  Evaluate only:    {args.eval_only}")
    print(f"  Infer only:       {args.infer_only}")
    print(f"  Load previous:    {args.load_previous_save}")
    print(f"  Exp tag:          {args.exp_tag if args.exp_tag else 'none'}")
    
    print("\n" + "=" * 60 + "\n")


def main(args):
    global _models
    # Before any sample is prepared or ResnetLEFT is built: both read the size.
    sys.modules[boxes_in_backbone_frame.__module__].set_backbone_input_size(
        getattr(args, "backbone_size", 224))

    CACHE_DIR = preprocess_folders_and_files(args.dummy, skip_extract=(args.dataset == "force3d"))
    NUM_INSTANCES = args.num_instances
    device = args.device

    # Load dataset
    if getattr(args, 'simple_only', False):
        # Filter to scene<=3 AND program<=3 BEFORE train-size slice
        full_ds = load_full_dataset(args, NUM_INSTANCES, CACHE_DIR,
                                     question_type=args.question_type)
        before = len(full_ds)
        filtered = [d for d in full_ds
                    if _scene_size(d) <= 3 and _program_size(d) <= 3]
        print(f"[simple-only] Filtered {before} → {len(filtered)} samples "
              f"(scene<=3 AND program<=3)")
        if args.train_size is not None:
            dataset = filtered[: args.train_size]
        else:
            dataset = filtered
    elif args.dataset == "force3d":
        dataset = _load_force3d_dataset(args, CACHE_DIR)
    else:
        dataset = preprocess_dataset(args, NUM_INSTANCES, CACHE_DIR, question_type=args.question_type)

    if args.print_constraints is not None:
        _print_constraints_and_exit(args)
        return 0

    if args.min_objects is not None:
        before = len(dataset)
        dataset = [d for d in dataset if len(d.get('all_objects', [])) >= args.min_objects]
        print(f"Filtered dataset: {before} -> {len(dataset)} images (min {args.min_objects} objects)")

    if args.max_objects is not None:
        n_within = sum(1 for d in dataset if len(d.get('all_objects', [])) <= args.max_objects)
        print(f"Dataset: {len(dataset)} total, {n_within} with <={args.max_objects} objects")

    # Train/test split on raw examples before graph compilation.
    # Priority:
    #   1. If --test-start is set, draw an independent test slice
    #      dataset[test_start : test_start + test_size] from the FULL cached
    #      dataset so it can be disjoint from --train-start.
    #   2. Else if --test-split is set, hold out the tail of the train slice.
    #   3. Else use the full train slice for training only.
    if args.dataset == "force3d":
        train_raw = [d for d in dataset if d.get("force3d_role") == "train"]
        test_raw = [d for d in dataset if d.get("force3d_role") == "test"] or None
        print(f"[force3d] scene split: {len(train_raw)} train, {len(test_raw or [])} test questions "
              f"({args.force3d_test_scenes} held-out scenes, seed {args.force3d_split_seed})")
    elif args.test_start is not None and args.test_size is not None and not args.eval_only:
        full_dataset = load_full_dataset(args, NUM_INSTANCES, CACHE_DIR,
                                         question_type=args.question_type)
        t_start = max(0, int(args.test_start))
        test_raw = full_dataset[t_start : t_start + args.test_size]
        train_raw = dataset
        print(f"Train/test slices: train[{args.train_start}:{args.train_start + (args.train_size or len(dataset))}] "
              f"({len(train_raw)} train), test[{t_start}:{t_start + args.test_size}] "
              f"({len(test_raw)} test) — drawn from full dataset size {len(full_dataset)}")
    elif args.test_split > 0 and args.test_split < len(dataset):
        test_raw = dataset[-args.test_split:]
        train_raw = dataset[:-args.test_split]
        print(f"Train/test split: {len(train_raw)} train, {len(test_raw)} test")
    else:
        if args.test_split >= len(dataset) and len(dataset) > 0:
            print(f"Requested test-split={args.test_split} is too large for dataset size {len(dataset)}; using full dataset for training")
        train_raw = dataset
        test_raw = None

    # Apply max-objects filtering to training set only (test set remains unfiltered).
    if args.max_objects is not None:
        before_train = len(train_raw)
        train_raw = [d for d in train_raw if len(d.get('all_objects', [])) <= args.max_objects]
        print(f"Training set after max-objects filter: {before_train} -> {len(train_raw)}")
        if test_raw is not None:
            test_within = sum(1 for d in test_raw if len(d.get('all_objects', [])) <= args.max_objects)
            print(f"Held-out test set (unfiltered): {len(test_raw)} total, {test_within} with <={args.max_objects} objects")
            test_raw_filtered = [d for d in test_raw if len(d.get('all_objects', [])) <= args.max_objects]
        else:
            test_raw_filtered = None
    else:
        test_raw_filtered = None
    
    print(f"Dataset length: {len(train_raw)}")
    print(f"Question type: {args.question_type}")

    if getattr(args, 'shuffle_labels', False) and len(train_raw) > 1:
        import random
        rng = random.Random(args.shuffle_seed)
        answers = [d.get('answer') for d in train_raw]
        permuted = list(answers)
        rng.shuffle(permuted)
        n_changed = sum(1 for a, b in zip(answers, permuted) if a != b)
        for d, a in zip(train_raw, permuted):
            d['answer'] = a
        print(f"[shuffle-labels] Permuted {len(train_raw)} train answers "
              f"(seed={args.shuffle_seed}, {n_changed} changed)")

    # Print samples
    if len(train_raw) > 0:
        for i in range(min(3, len(train_raw))):
            print(f"\nSample question: {train_raw[i].get('question_raw', '')}")
            print(f"Sample answer: {train_raw[i].get('answer', '')}")

    # Create plugin manager (without BERT unfreezing)
    from domiknows.program.plugins.callback_plugin_manager import CallbackPluginManager
    from domiknows.program.plugins.epoch_logging_plugin import EpochLoggingPlugin
    from domiknows.program.plugins.adaptive_tnorm_plugin import AdaptiveTNormPlugin
    from domiknows.program.plugins.gradient_flow_plugin import GradientFlowPlugin
    from domiknows.program.plugins.gumbel_monitoring_plugin import GumbelMonitoringPlugin
    
    plugin_manager = CallbackPluginManager()
    plugin_manager.register(EpochLoggingPlugin(), 'EpochLogging')
    plugin_manager.register(AdaptiveTNormPlugin(), 'AdaptiveTNorm')
    plugin_manager.register(GradientFlowPlugin(), 'GradientFlow')
    plugin_manager.register(GumbelMonitoringPlugin(), 'GumbelMonitoring')
    
    # Create program
    program, train_dataset, test_dataset, attribute_names_dict = program_declaration(
        train_raw,
        test_raw,
        args, 
        device=device
    )

    test_dataset_filtered = None
    if test_raw_filtered is not None and test_raw is not None and len(test_raw_filtered) < len(test_raw):
        test_dataset_filtered = _subset_logic_dataset(test_dataset, test_raw_filtered)

    eval_dataset = test_dataset if test_dataset is not None else train_dataset
    eval_dataset_name = "test" if test_dataset is not None else "train"

    _ckpt_extra = dict(
        lora_r=args.lora_r,
        softmax_temp=args.softmax_temp,
        max_objects=args.max_objects,
        exp_tag=args.exp_tag,
    )
    _results_file = f"results_{args.exp_tag}.txt" if args.exp_tag else "results.txt"
    
    # Log configuration
    log_training_config(args, _models, train=train_raw, dev=None,
                       test=test_raw if test_raw is not None else train_raw,
                       plugin_manager=plugin_manager)

    if getattr(args, 'step_notebook', False):
        notebook_dir = RUN_DIR / "logs"
        notebook_file = args.step_notebook_file or (
            f"step_notebook_{args.exp_tag}.jsonl" if args.exp_tag else "step_notebook.jsonl"
        )
        setup_step_notebook(
            log_dir=str(notebook_dir),
            filename=notebook_file,
            run_tag=args.exp_tag,
            metadata={
                'question_type': args.question_type,
                'use_vlm': args.use_vlm,
                'peft': args.peft,
                'oracle_mode': args.oracle_mode,
                'oracle_confidence': args.oracle_confidence if args.oracle_mode else None,
                'infer_only': args.infer_only,
                'tnorm': args.tnorm,
                'train_size': len(train_raw),
                'test_size': len(test_raw) if test_raw is not None else 0,
                'epochs': args.epochs,
            },
        )
    
    save_file = ckpt_path(args.lr, 1, args.load_epoch, args.batch_size, args.tnorm,
                         args.subset, args.question_type, **_ckpt_extra)

    if args.infer_only:
        with torch.no_grad():
            acc = program.evaluate_condition(eval_dataset, device=device)
        print(f"Accuracy on {eval_dataset_name.capitalize()}: {acc :.2f}%")
        with open(_results_file, 'a') as f:
            print(save_file, file=f)
            print(f"Question type: {args.question_type}", file=f)
            print(f"Epoch: {args.load_epoch}", file=f)
            print(f"Learning rate: {args.lr}", file=f)
            print(f"Train examples: {len(train_raw)}", file=f)
            print(f"Test examples: {len(test_raw) if test_raw is not None else 0}", file=f)
            print(f"Accuracy: {acc :.2f}%", file=f)
    else:
        if not args.eval_only:
            # Configure plugins (no BERT-specific optimizer factory needed)
            if not args.oracle_mode and not args.use_vlm and not args.disable_plugins:
                plugin_manager.configure_all(
                    program=program,
                    models=_models,
                    args=args,
                    dataset=train_dataset
                )
                
                Optim = torch.optim.Adam
            else:
                Optim = torch.optim.Adam
            # LearningBasedProgram.train builds the model optimizer as
            # Optim(model.parameters()) without a learning rate, so --lr used to
            # reach only the constraint optimizer (c_lr) and the model always
            # trained at Adam's default 1e-3.  Bind it explicitly.
            Optim = _build_optimizer_factory(args, Optim)
            
            # Load previous checkpoint if needed
            if args.load_previous_save and args.subset > 1:
                previous_save = ckpt_path(args.lr, 1, args.load_epoch, args.batch_size,
                                         args.tnorm, args.subset - 1, args.question_type,
                                         **_ckpt_extra)
                if previous_save.exists():
                    program.load(previous_save)
            elif args.load_previous_save and args.load_epoch > 0:
                previous_save = ckpt_path(args.lr, 1, args.load_epoch - 1, args.batch_size,
                                         args.tnorm, args.subset, args.question_type,
                                         **_ckpt_extra)
                if previous_save.exists():
                    program.load(previous_save)

            # TensorBoard writer
            _tb_writer = None
            if getattr(args, 'tensorboard', False):
                from torch.utils.tensorboard import SummaryWriter
                _tb_tag = args.exp_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
                _tb_dir = RUN_DIR / "logs" / "tb" / _tb_tag
                _tb_dir.mkdir(parents=True, exist_ok=True)
                _tb_writer = SummaryWriter(log_dir=str(_tb_dir))
                print(f"[tb] writing to {_tb_dir}")

            # Baseline evaluation before training for quick sanity-check.
            if getattr(args, 'skip_train_eval', False):
                baseline_acc = float('nan')
                print("Accuracy before training: <skipped>")
                if test_dataset is not None:
                    with torch.no_grad():
                        test_baseline_acc = program.evaluate_condition(test_dataset, device=device)
                    print(f"Test accuracy before training: {test_baseline_acc :.2f}%")
                    if _tb_writer is not None:
                        _tb_writer.add_scalar("test/acc", test_baseline_acc, 0)
            else:
                with torch.no_grad():
                    baseline_acc = program.evaluate_condition(train_dataset, device=device)
                print(f"Accuracy before training: {baseline_acc :.2f}%")
                if _tb_writer is not None:
                    _tb_writer.add_scalar("train/acc", baseline_acc, 0)

            # Install gradient chain diagnostic
            diagnostic = GradChainDiagnostic(program, _models['classifiers'])
            diagnostic.install()

            # Running per-step train-acc: replaces the costly separate train-eval
            _train_acc_tracker = None
            if getattr(args, 'running_train_acc', False):
                _train_acc_tracker = _RunningTrainAccTracker(tb_writer=_tb_writer)
                program.before_train_epoch.append(_train_acc_tracker.reset)
                program.after_train_step.append(_train_acc_tracker.after_step)
                program.after_train_epoch.append(_train_acc_tracker.report_and_reset)

            cached_curriculum_key = None
            cached_curriculum_dataset = train_dataset
            active_train_dataset = train_dataset
            active_train_raw = train_raw
            active_train_label = "train"

            # Training loop
            for i in range(args.epochs):
                print(f"Training epoch {i + 1}/{args.epochs}")
                # Expose the outer epoch number to program/callbacks so
                # tqdm descriptions and plugin logs show the true epoch
                # instead of the reset inner-1 from train_epoch_num=1.
                program.global_epoch = i + 1

                epoch_train_dataset = train_dataset
                epoch_train_raw = train_raw
                if args.curriculum != "none":
                    epoch_train_raw, curriculum_key, curriculum_info = _select_curriculum_train_raw(
                        train_raw, i + 1, args.curriculum
                    )
                    if curriculum_key != cached_curriculum_key:
                        if cached_curriculum_key is None:
                            print(
                                f"[curriculum] CURRICULUM_STRATEGY initial stage at epoch {i + 1}: "
                                f"mode={curriculum_info['mode']}, "
                                f"scene<={curriculum_info['scene_limit']}, "
                                f"program<={curriculum_info['program_limit']}"
                            )
                        else:
                            _, prev_scene_limit, prev_program_limit, _ = cached_curriculum_key
                            print(
                                f"[curriculum] CURRICULUM_STRATEGY updated at epoch {i + 1}: "
                                f"scene<={prev_scene_limit}, program<={prev_program_limit} "
                                f"-> scene<={curriculum_info['scene_limit']}, "
                                f"program<={curriculum_info['program_limit']}"
                            )
                        if curriculum_info["fallback_kind"] == "full":
                            cached_curriculum_dataset = train_dataset
                            print(
                                f"[curriculum] Epoch {i + 1}: mode={curriculum_info['mode']}, "
                                f"scene<={curriculum_info['scene_limit']}, "
                                f"program<={curriculum_info['program_limit']} yielded 0 samples "
                                "even after relaxation; using full training set"
                            )
                        else:
                            cached_curriculum_dataset = _subset_logic_dataset(
                                train_dataset, epoch_train_raw
                            )
                            if curriculum_info["fallback_kind"] == "relaxed_or":
                                print(
                                    f"[curriculum] Epoch {i + 1}: strict bucket empty; using relaxed "
                                    f"scene/program OR filter, samples={curriculum_info['selected']}/"
                                    f"{curriculum_info['total']}"
                                )
                            elif curriculum_info["fallback_kind"] == "nearest":
                                print(
                                    f"[curriculum] Epoch {i + 1}: strict bucket empty; using nearest "
                                    f"bucket (violation={curriculum_info['min_violation']}), "
                                    f"samples={curriculum_info['selected']}/{curriculum_info['total']}"
                                )
                            else:
                                print(
                                    f"[curriculum] Epoch {i + 1}: mode={curriculum_info['mode']}, "
                                    f"scene<={curriculum_info['scene_limit']}, "
                                    f"program<={curriculum_info['program_limit']}, "
                                    f"samples={curriculum_info['selected']}/{curriculum_info['total']}"
                                )
                        cached_curriculum_key = curriculum_key
                    epoch_train_dataset = cached_curriculum_dataset
                    print(
                        f"[curriculum] Epoch {i + 1}: training on "
                        f"{len(epoch_train_raw)}/{len(train_raw)} examples"
                    )

                active_train_dataset = epoch_train_dataset
                active_train_raw = epoch_train_raw
                active_train_label = "curriculum train" if args.curriculum != "none" else "train"
                epoch_logging_plugin = plugin_manager.get_plugin('EpochLogging')
                if (
                    epoch_logging_plugin is not None
                    and hasattr(epoch_logging_plugin, "_create_eval_subset")
                    and hasattr(epoch_logging_plugin, "args")
                ):
                    epoch_logging_plugin.dataset = active_train_dataset
                    epoch_logging_plugin._create_eval_subset(
                        active_train_dataset,
                        eval_fraction=epoch_logging_plugin.args.eval_fraction,
                        min_samples=epoch_logging_plugin.args.eval_min_samples,
                        seed=epoch_logging_plugin.args.eval_seed,
                    )

                save_file = ckpt_path(args.lr, i + 1, args.load_epoch, args.batch_size,
                                     args.tnorm, args.subset, args.question_type,
                                     **_ckpt_extra)
                program.train(epoch_train_dataset, Optim=Optim, train_epoch_num=1, c_lr=args.lr,
                              c_warmup_iters=0, batch_size=args.batch_size, device=device,
                              print_loss=False)
                program.save(save_file)
                print(f"Saved to {save_file}")

                if getattr(args, 'skip_train_eval', False):
                    print(f"Epoch {i + 1} {active_train_label} accuracy: <skipped>")
                else:
                    with torch.no_grad():
                        epoch_train_acc = program.evaluate_condition(active_train_dataset, device=device)
                    print(f"Epoch {i + 1} {active_train_label} accuracy: {epoch_train_acc :.2f}%")
                    _print_force3d_soft_acc(args, program, active_train_dataset, device, f"Epoch {i + 1} {active_train_label}")
                if test_dataset is not None:
                    with torch.no_grad():
                        epoch_test_acc = program.evaluate_condition(test_dataset, device=device)
                    print(f"Epoch {i + 1} test accuracy: {epoch_test_acc :.2f}%")
                    _print_force3d_ref_top1(args, program, test_dataset, device, f"Epoch {i + 1}")
                    _print_force3d_soft_acc(args, program, test_dataset, device, f"Epoch {i + 1} test")
                    if _tb_writer is not None:
                        _tb_writer.add_scalar("test/acc", epoch_test_acc, i + 1)

            # Final evaluation
            _skip_train_eval = getattr(args, 'skip_train_eval', False)
            with torch.no_grad():
                if _skip_train_eval:
                    final_train_acc = float('nan')
                    final_eval = {}
                else:
                    final_train_acc = program.evaluate_condition(active_train_dataset, device=device)
                    final_eval = program.evaluate_condition(train_dataset, device=device,
                                                            threshold=0.5, return_dict=True)
                final_test_acc = program.evaluate_condition(test_dataset, device=device) if test_dataset is not None else None
                final_test_acc_filtered = (
                    program.evaluate_condition(test_dataset_filtered, device=device)
                    if test_dataset_filtered is not None else None
                )
            if _skip_train_eval:
                print(f"{active_train_label.capitalize()} accuracy after training: <skipped>")
            else:
                print(f"{active_train_label.capitalize()} accuracy after training: {final_train_acc:.2f}%")
            if final_test_acc is not None:
                print(f"Test accuracy after training: {final_test_acc:.2f}%")
                _print_force3d_ref_top1(args, program, test_dataset, device, "Final")
            if final_test_acc_filtered is not None:
                print(f"Test accuracy (<={args.max_objects} objects): {final_test_acc_filtered:.2f}%")
            
            # Display plugin summaries
            if not args.disable_plugins:
                plugin_manager.final_display_all(final_eval=final_eval)

            with open(_results_file, 'a') as f:
                print(f"=== {args.exp_tag or 'training_run'} ===", file=f)
                print(save_file, file=f)
                print(f"Question type: {args.question_type}", file=f)
                print(f"Epochs: {args.epochs}", file=f)
                print(f"Learning rate: {args.lr}", file=f)
                print(f"Train examples: {len(train_raw)}", file=f)
                print(f"Active train examples (final epoch): {len(active_train_raw)}", file=f)
                print(f"Test examples: {len(test_raw) if test_raw is not None else 0}", file=f)
                print(f"Baseline accuracy: {baseline_acc:.2f}%", file=f)
                print(f"{active_train_label.capitalize()} accuracy: {final_train_acc:.2f}%", file=f)
                if final_test_acc is not None:
                    print(f"Test accuracy: {final_test_acc:.2f}%", file=f)
                if final_test_acc_filtered is not None:
                    print(f"Test accuracy (<={args.max_objects} obj): {final_test_acc_filtered:.2f}%", file=f)
                print("", file=f)
            
        else:
            # Evaluation only
            epoch_to_eval = args.epochs if args.epochs > 0 else 1
            save_file = ckpt_path(args.lr, epoch_to_eval, args.load_epoch, args.batch_size,
                                 args.tnorm, args.subset, args.question_type,
                                 **_ckpt_extra)
            if save_file.exists():
                print(f"Loading from {save_file}")
                program.load(save_file)
                acc = program.evaluate_condition(eval_dataset, device=device)
                print(f"Accuracy on {eval_dataset_name.capitalize()}: {acc :.2f}%")

                with open(_results_file, 'a') as f:
                    print(save_file, file=f)
                    print(f"Question type: {args.question_type}", file=f)
                    print(f"Epoch: {args.load_epoch}", file=f)
                    print(f"Learning rate: {args.lr}", file=f)
                    print(f"Train examples: {len(train_raw)}", file=f)
                    print(f"Test examples: {len(test_raw) if test_raw is not None else 0}", file=f)
                    print(f"Accuracy: {acc :.2f}%", file=f)
            else:
                print(f"Checkpoint not found: {save_file}")
    
    if MONITORING_AVAILABLE:
        finish_experiment(label="run_1")
        disable_monitoring()

    _nb = StepNotebook.active()
    if _nb is not None:
        _nb.close()

    return 0


if __name__ == '__main__':
    setup_console_log()
    args = parse_arguments()
    if args.production_log_mode:
        setProductionLogMode(no_UseTimeLog=args.no_time_log, reuse_model=args.reuse_model)
    main(args)
