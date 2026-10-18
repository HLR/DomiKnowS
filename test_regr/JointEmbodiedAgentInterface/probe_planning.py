"""Run input-sensitivity probes on validation data with one model load."""
import json
from pathlib import Path
import torch
from .eval_planning import build_eval_parser, _prepare_eval
from .models import JointQwenVLPlanner
from .checkpoint import load_joint_checkpoint
from .program import _runtime_adapter
from test_regr.VLABenchAgentInterface.main import _controller
from test_regr.VLABenchAgentInterface.training import evaluate_planner
from test_regr.VLABenchAgentInterface.planning_augmentation import diagnostic_examples


def main(argv=None):
    args = build_eval_parser().parse_args(argv)
    if args.split != "validation" or args.output is None:
        raise ValueError("probes require --split validation and an output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    args.vlabench_planning_dir = str(args.planning_dir)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    runtime, splits = _prepare_eval(args, device)
    examples = [e for e in splits["validation"] if e.dimension == "PhysLaw"]
    if args.limit:
        examples = examples[:args.limit]
    planner = JointQwenVLPlanner.from_pretrained(eai_vocabulary=runtime.eai_vocabulary, vlabench_vocabulary=runtime.vlabench_vocabulary,
        model_id=args.planner_model, load_in_4bit=args.load_in_4bit, decoder_hidden_size=args.planner_decoder_hidden_dim)
    controller = _controller(args, device)
    load_joint_checkpoint(args.checkpoint, runtime=runtime, planner=planner, controller=controller, map_location=device,
                          allow_planning_schema_migration=args.allow_planning_schema_migration)
    planner.eval()
    results = {}
    for mode in ("original", "permuted-entities", "no-images", "shuffled-images", "shuffled-instructions"):
        print(f"[probe] {mode} ({len(examples)} validation PhysLaw episodes)", flush=True)
        with (args.output / f"{mode}.jsonl").open("w") as trace, runtime.domain_scope("vlabench"):
            results[mode] = evaluate_planner(planner.for_domain("vlabench"), diagnostic_examples(examples, mode, args.seed),
                _runtime_adapter(runtime, "vlabench"), decision_trace=trace)
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({mode: {"examples": result["examples"], "generation_failures": result["generation_failures"],
                              "scores": result["planning_metrics"]["overall"]} for mode, result in results.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
