import argparse
import json
from pathlib import Path

from .dataset import DEFAULT_TEMPORAL_DATA_ROOT, discover_temporal_datasets, load_temporal_instances
from .llm_inference import SmallCausalLMChoiceBackend, run_llm_inference


def default_matres_file(root):
    discovered = discover_temporal_datasets(root)
    for preferred in ("platinum.txt", "aquaint.txt", "timebank.txt"):
        for path in discovered["matres"]:
            path = Path(path)
            if path.name == preferred:
                return path
    for path in discovered["matres"]:
        path = Path(path)
        if path.is_file():
            return path
    raise FileNotFoundError(f"No MATRES file found under {root}")


def main():
    parser = argparse.ArgumentParser(description="Run small-LLM multiple-choice inference for TemporalRelation examples.")
    parser.add_argument("--root", type=Path, default=DEFAULT_TEMPORAL_DATA_ROOT)
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    path = args.path or default_matres_file(args.root)
    instances = load_temporal_instances(path, limit=args.limit, group_by_document=True)
    backend = SmallCausalLMChoiceBackend(
        model_path=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    for index, instance in enumerate(instances):
        result = run_llm_inference(instance, backend)
        print(json.dumps({"index": index, "doc_id": instance.get("doc_id"), **result}, indent=2))


if __name__ == "__main__":
    main()
