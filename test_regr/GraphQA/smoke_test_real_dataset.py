import argparse
from collections import Counter

from .dataset import (
    GraphQADatasetNotFound,
    discover_vqar_dataset,
    load_vqar_tasks,
    vqar_task_to_graphqa_instance,
)
from .execution import create_query_logic, validate_dataset_convertible
from .graph import create_graphqa_graph
from .oracle import answer_object


def smoke_test(root=None, limit=25):
    discovered = discover_vqar_dataset(root)
    task_paths = discovered["task_paths"]
    if not task_paths:
        raise GraphQADatasetNotFound(
            f"No task pickle files found under {discovered['data_dir'] / 'dataset/task_list'}"
        )

    task_path = task_paths[0]
    tasks = load_vqar_tasks(task_path, limit=limit)
    instances = []
    failures = []
    functions = Counter()

    for index, task in enumerate(tasks):
        for clause in task.get("question", {}).get("clauses", []):
            functions[clause.get("function")] += 1
        try:
            instance = vqar_task_to_graphqa_instance(task)
            create_query_logic(instance)
            instances.append(instance)
        except Exception as exc:
            failures.append((index, type(exc).__name__, str(exc)))

    compiled_count = 0
    if instances:
        context = create_graphqa_graph(instances)
        convertible_failures = validate_dataset_convertible(instances)
        if not convertible_failures:
            compiled_count = len(context.graph.compile_executable(
                [dict(instance, logic_str=create_query_logic(instance), logic_label=0) for instance in instances],
                logic_keyword="logic_str",
                logic_label_keyword="logic_label",
                extra_namespace_values=context.namespace,
            ))

    print(f"root={discovered['root']}")
    print(f"task_path={task_path}")
    print(f"sampled_tasks={len(tasks)}")
    print(f"converted_instances={len(instances)}")
    print(f"compile_checked={compiled_count}")
    print(f"conversion_failures={len(failures)}")
    print(f"clause_functions={dict(functions)}")

    if instances:
        example = instances[0]
        print("example_source_question_id=" + str(example.get("source_question_id")))
        print("example_objects=" + str(example.get("objects")[:10]))
        print("example_query=" + str(example.get("query")))
        print("example_expected_answer=" + str(example.get("expected_answer")))
        print("example_oracle_answer=" + str(answer_object(example)))
        print("example_logic=" + create_query_logic(example))

    if failures[:5]:
        print("first_failures=" + str(failures[:5]))

    return len(instances), failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    smoke_test(root=args.root, limit=args.limit)


if __name__ == "__main__":
    main()
