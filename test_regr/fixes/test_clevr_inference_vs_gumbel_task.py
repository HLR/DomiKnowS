from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

TASK_DIR = Path(__file__).resolve().parents[2] / "Tasks" / "clevr_inference_vs_gumbel"
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from clevr_constraints import answer_to_query_label, prepare_logic_fields, translate_program_to_constraint
from graph import create_graph
import main as clevr_main


def test_query_program_translates_to_query_iota_constraint():
    data = json.load(open(TASK_DIR / "data" / "clevr_20_programs.json", encoding="utf-8"))
    program = data["items"][0]["program"]

    constraint = translate_program_to_constraint(program)

    assert "queryL(material" in constraint
    assert "iotaL(" in constraint
    assert "right(" in constraint
    assert "left(" in constraint


def test_answer_to_query_label_maps_all_attribute_groups():
    assert answer_to_query_label("gray", "color") == 0
    assert answer_to_query_label("metal", "material") == 1
    assert answer_to_query_label("sphere", "shape") == 1
    assert answer_to_query_label("large", "size") == 1


def test_compact_dataset_preserves_question_answer_order():
    copied = json.load(open(TASK_DIR / "data" / "20_examples_string_CLEVR.json", encoding="utf-8"))
    compact = json.load(open(TASK_DIR / "data" / "clevr_20_programs.json", encoding="utf-8"))

    compact_questions = [item["question"] for item in compact["items"]]
    copied_questions = [item["question"] for item in copied]

    assert len(compact_questions) == 40
    assert compact_questions == copied_questions


def test_expanded_compact_dataset_covers_templates_and_answer_types():
    compact = json.load(open(TASK_DIR / "data" / "clevr_20_programs.json", encoding="utf-8"))
    generated = [item for item in compact["items"] if item.get("generated_source")]

    assert len(compact["items"]) == 40
    assert len(generated) == 20
    assert {item["template_filename"] for item in generated} == {
        "zero_hop.json",
        "one_hop.json",
        "two_hop.json",
        "three_hop.json",
        "same_relate.json",
        "single_and.json",
        "single_or.json",
        "comparison.json",
        "compare_integer.json",
    }
    final_functions = {
        item["program"][-1]["function"]
        for item in generated
    }
    assert any(fn.startswith("query_") for fn in final_functions)
    assert "count" in final_functions
    assert any(fn in {"exist", "equal_integer", "less_than", "greater_than"} or fn.startswith("equal_") for fn in final_functions)


def test_expanded_compact_dataset_compiles_mixed_answer_types():
    items = clevr_main.load_items()
    results = create_graph(items, include_query_questions=True, relation_syntax="legacy")
    executions = results[0]
    query_types = results[9]

    prepare_logic_fields(items, device="cpu", executions=executions, query_types=query_types)

    assert len(items) == 40
    assert any(item["query_type"] is not None for item in items)
    assert any(item["query_type"] is None and isinstance(item["answer"], bool) for item in items)
    assert any(item["query_type"] is None and isinstance(item["answer"], int) for item in items)


def test_task_defaults_keep_global_constraints_downweighted(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py"])

    args = clevr_main.parse_args()

    assert args.train_items is None
    assert args.eval_items is None
    assert args.global_constraint_loss_weight == 0.1
    assert args.executable_constraint_loss_weight == 1.0
    assert args.ilp_benchmark_warmup == 1
    assert args.ilp_benchmark_repeats == 3
    assert args.ilp_benchmark_items == 0
    assert args.ilp_benchmark_only is False


def test_ilp_benchmark_selects_relation_free_count_samples():
    items = clevr_main.load_items()

    selected = clevr_main._select_relation_free_count_samples(items, items)

    assert [sample["question"] for sample in selected] == [
        "What number of tiny cyan cylinders are there?",
        "What number of objects are either cubes or big matte things?",
        "How many things are either cyan spheres or tiny brown shiny blocks?",
        "How many red rubber objects are there?",
    ]
    for sample in selected:
        assert sample["program"][-1]["function"] == "count"
        assert all(
            step["function"] in {
                "scene",
                "count",
                "unique",
                "union",
                "intersect",
            }
            or step["function"].startswith("filter_")
            for step in sample["program"]
        )

    assert clevr_main._select_simple_count_sample(items, items) is selected[0]


def test_main_benchmark_only_skips_comparison_and_ad_hoc(monkeypatch):
    args = SimpleNamespace(
        device="cpu",
        disable_global_constraint_loss=False,
        global_constraint_loss_weight=0.1,
        executable_constraint_loss_weight=1.0,
        ilp_benchmark_only=True,
        ilp_benchmark_warmup=2,
        ilp_benchmark_repeats=4,
        ilp_benchmark_items=3,
    )
    built = object()
    calls = []
    monkeypatch.setattr(clevr_main, "parse_args", lambda: args)
    monkeypatch.setattr(clevr_main, "load_items", lambda: ["sample"])

    def fake_build(name, program_cls, items, parsed_args, device, **kwargs):
        calls.append(("build", name, program_cls, items, parsed_args, device, kwargs))
        return built

    monkeypatch.setattr(clevr_main, "build_program", fake_build)
    monkeypatch.setattr(
        clevr_main,
        "train_program",
        lambda program, parsed_args, device: calls.append(
            ("train", program, parsed_args, device)
        ),
    )
    monkeypatch.setattr(
        clevr_main,
        "print_post_training_ilp_benchmark",
        lambda program, device, **kwargs: calls.append(
            ("benchmark", program, device, kwargs)
        ),
    )
    monkeypatch.setattr(
        clevr_main,
        "evaluate",
        lambda *_args, **_kwargs: pytest.fail("benchmark-only evaluated models"),
    )
    monkeypatch.setattr(
        clevr_main,
        "print_post_training_ad_hoc_results",
        lambda *_args, **_kwargs: pytest.fail("benchmark-only ran ad hoc inference"),
    )

    clevr_main.main()

    assert calls == [
        (
            "build",
            "InferenceProgram",
            clevr_main.InferenceProgram,
            ["sample"],
            args,
            "cpu",
            {},
        ),
        ("train", built, args, "cpu"),
        (
            "benchmark",
            built,
            "cpu",
            {"warmup": 2, "repeats": 4, "items": 3},
        ),
    ]


def test_post_training_example_uses_ephemeral_ad_hoc_query():
    calls = {}

    class FakeDataNode:
        def inferExecutableResults(self, **kwargs):
            calls.setdefault("inference", []).append(kwargs)
            return {
                "ADHOC0": {
                    "type": "count",
                    "answer": 2,
                    "probability": 0.75,
                    "distribution": None,
                    "mode": "tnorm",
                    "exact": False,
                }
            }

    class FakeBuilder:
        def getDataNode(self, *, device):
            calls["device"] = device
            return FakeDataNode()

    class FakeModel:
        def eval(self):
            calls["model_eval"] = True

        def __call__(self, sample):
            calls["sample"] = sample
            return None, None, None, FakeBuilder()

    class FakeConstraintModel:
        tnorm = "P"

        def eval(self):
            calls["constraint_model_eval"] = True

    sample = {
        "question": "How many red objects are there?",
        "answer": 2,
        "logic_str": 'sumL(red("x"))',
    }
    red = object()
    built = clevr_main.BuiltProgram(
        name="learned",
        program=SimpleNamespace(
            model=FakeModel(),
            cmodel=FakeConstraintModel(),
        ),
        train_dataset=[],
        eval_dataset=[sample],
        query_namespace={"red": red},
    )

    returned_sample, comparison = clevr_main.infer_ad_hoc_example(built, "cpu")

    assert returned_sample is sample
    assert list(comparison.results) == ["tnorm", "circuit", "ilp"]
    assert all(result["answer"] == 2 for result in comparison.results.values())
    assert comparison.answers_agree is True
    assert comparison.types_agree is True
    assert [call["mode"] for call in calls["inference"]] == [
        "tnorm",
        "circuit",
        "ilp",
    ]
    assert all(
        call["queries"] == 'sumL(red("x"))'
        for call in calls["inference"]
    )
    assert all(
        call["queryNamespace"] == {"red": red}
        for call in calls["inference"]
    )
    assert all(call["tnorm"] == "P" for call in calls["inference"])
    assert all(call["populate"] is False for call in calls["inference"])
    assert all(call["compiled"] is False for call in calls["inference"])
    assert calls["device"] == "cpu"
    assert calls["model_eval"] is True
    assert calls["constraint_model_eval"] is True


def _mock_ilp_benchmark(*, fail_dynamic=False):
    events = []
    datanodes = []

    class FakeConcept:
        def __init__(self, name):
            self.name = name

    all_concepts = tuple(
        FakeConcept(name)
        for name in ("constraint", "obj", "red", "rubber", "blue")
    )

    class FakeInnerConstraint:
        @staticmethod
        def getLcConcepts():
            return {"red", "rubber"}

    class FakeGraph:
        def __init__(self):
            self._active = all_concepts
            self._active_concepts = None
            self.executableLCs = {
                "ELC7": SimpleNamespace(innerLC=FakeInnerConstraint())
            }

        def set_active_concepts(self, requested):
            if requested is None:
                self._active = all_concepts
                self._active_concepts = None
                events.append(("activate", "full"))
            else:
                names = ("constraint", "obj", *requested)
                self._active = tuple(FakeConcept(name) for name in names)
                self._active_concepts = frozenset(names)
                events.append(("activate", tuple(requested)))
            return self._active

        def get_active_concepts(self):
            return self._active

    graph = FakeGraph()

    class FakeConstraintDataNode:
        def __init__(self):
            self.attributes = {
                "label/label": 1,
                "ELC7/label": 1,
            }

    class FakeDataNode:
        def __init__(self, active_names):
            self.active_names = active_names
            self.collectedConceptsAndRelations = None
            self.current_device = "cpu"
            self.constraint_dn = FakeConstraintDataNode()

        def _getExecutableConstraintDataNode(self):
            return self.constraint_dn

        def inferILPResults(self, *concepts):
            events.append((
                "infer_ilp",
                self.active_names,
                tuple(concept.name for concept in concepts),
                dict(self.constraint_dn.attributes),
            ))
            if fail_dynamic and len(self.active_names) < len(all_concepts):
                raise RuntimeError("dynamic inference failed")
            predicate_count = 5 if len(self.active_names) == len(all_concepts) else 2
            self.collectedConceptsAndRelations = [object()] * predicate_count
            self.constraint_dn.attributes["ELC7/answer"] = 2

    class FakeBuilder:
        def __init__(self, datanode):
            self.datanode = datanode

        def getDataNode(self, *, device):
            assert device == "cpu"
            return self.datanode

    class FakeModel:
        def eval(self):
            events.append(("eval", "model"))

        def __call__(self, sample):
            active_names = tuple(concept.name for concept in graph.get_active_concepts())
            datanode = FakeDataNode(active_names)
            datanodes.append(datanode)
            events.append(("forward", active_names))
            return None, None, None, FakeBuilder(datanode)

    class FakeConstraintModel:
        tnorm = "P"

        def eval(self):
            events.append(("eval", "constraint"))

    sample = {
        "question": "How many red rubber objects are there?",
        "answer": 2,
        "logic_str": 'sumL(andL(red("x"), rubber(path="x")))',
        "_constraint_ELC7": 1,
    }
    built = clevr_main.BuiltProgram(
        name="learned",
        program=SimpleNamespace(
            graph=graph,
            model=FakeModel(),
            cmodel=FakeConstraintModel(),
        ),
        train_dataset=[],
        eval_dataset=[],
        query_namespace={"red": object(), "rubber": object()},
        ilp_benchmark_sample=sample,
    )
    return built, graph, events, datanodes


def test_ilp_benchmark_uses_fresh_full_then_dynamic_datanodes(monkeypatch):
    built, graph, events, datanodes = _mock_ilp_benchmark()
    clock = iter((0.0, 0.5, 1.0, 1.1, 2.0, 2.4, 3.0, 3.2))
    monkeypatch.setattr(clevr_main, "perf_counter", lambda: next(clock))

    comparison = clevr_main.benchmark_ilp_graph_activation(
        built, "cpu", warmup=0, repeats=2
    )

    assert len(datanodes) == 4
    assert len({id(datanode) for datanode in datanodes}) == 4
    assert [event[:2] for event in events if event[0] == "activate"] == [
        ("activate", "full"),
        ("activate", ("red", "rubber")),
        ("activate", "full"),
        ("activate", ("red", "rubber")),
        ("activate", "full"),
    ]
    inference_events = [event for event in events if event[0] == "infer_ilp"]
    assert len(inference_events) == 4
    assert all(event[2] == () for event in inference_events)
    assert all(event[3] == {"ELC7/label": 1} for event in inference_events)
    assert comparison.requested_concepts == ("red", "rubber")
    assert comparison.full.durations_seconds == pytest.approx((0.5, 0.4))
    assert comparison.dynamic.durations_seconds == pytest.approx((0.1, 0.2))
    assert comparison.full.median_seconds == pytest.approx(0.45)
    assert comparison.dynamic.median_seconds == pytest.approx(0.15)
    assert comparison.milliseconds_saved == pytest.approx(300.0)
    assert comparison.reduction_percent == pytest.approx(100.0 * 0.3 / 0.45)
    assert comparison.speedup == pytest.approx(3.0)
    assert comparison.answers_agree is True
    assert comparison.full.predicate_count == 5
    assert comparison.dynamic.predicate_count == 2
    assert graph._active_concepts is None


def test_ilp_benchmark_suite_runs_multiple_questions_and_aggregates(monkeypatch):
    samples = tuple(
        {"question": f"question {index}", "answer": index}
        for index in range(1, 4)
    )
    built = SimpleNamespace(
        name="learned",
        ilp_benchmark_sample=samples[0],
        ilp_benchmark_samples=samples,
    )
    calls = []

    def fake_benchmark(
        actual_built,
        device,
        *,
        warmup,
        repeats,
        sample,
    ):
        calls.append((actual_built, device, warmup, repeats, sample))
        index = samples.index(sample) + 1
        full = clevr_main.ILPTiming(
            durations_seconds=(float(index * 2),),
            median_seconds=float(index * 2),
            answer=index,
            result_type="count",
            active_concepts=("all",),
            predicate_count=5,
        )
        dynamic = clevr_main.ILPTiming(
            durations_seconds=(float(index) / 2.0,),
            median_seconds=float(index) / 2.0,
            answer=index,
            result_type="count",
            active_concepts=("query",),
            predicate_count=2,
        )
        return clevr_main.ILPGraphPerformance(
            sample=sample,
            requested_concepts=("query",),
            full=full,
            dynamic=dynamic,
            milliseconds_saved=float(index) * 1500.0,
            reduction_percent=75.0,
            speedup=4.0,
            answers_agree=True,
        )

    monkeypatch.setattr(
        clevr_main,
        "benchmark_ilp_graph_activation",
        fake_benchmark,
    )

    report = clevr_main.benchmark_ilp_graph_activations(
        built,
        "cpu",
        warmup=1,
        repeats=3,
        items=2,
    )

    assert [call[-1] for call in calls] == list(samples[:2])
    assert all(call[2:4] == (1, 3) for call in calls)
    assert report.comparisons[0].sample is samples[0]
    assert report.comparisons[1].sample is samples[1]
    assert report.attempted == 2
    assert report.failures == ()
    assert len(report.question_types) == 1
    assert report.question_types[0].question_type == "unknown"
    assert report.question_types[0].attempted == 2
    assert report.question_types[0].succeeded == 2
    assert report.question_types[0].failed == 0
    assert report.question_types[0].full_average_seconds == pytest.approx(3.0)
    assert report.question_types[0].dynamic_average_seconds == pytest.approx(0.75)
    assert report.full_workload_seconds == pytest.approx(6.0)
    assert report.dynamic_workload_seconds == pytest.approx(1.5)
    assert report.milliseconds_saved == pytest.approx(4500.0)
    assert report.reduction_percent == pytest.approx(75.0)
    assert report.speedup == pytest.approx(4.0)
    assert report.answers_agree is True
    table = clevr_main._question_type_table(report)
    assert all(
        heading in table[0]
        for heading in (
            "Question type",
            "Success/total",
            "Full avg.",
            "Dynamic avg.",
            "Speedup",
        )
    )
    assert table[2].split() == [
        "unknown",
        "2/2",
        "3000.00",
        "ms",
        "750.00",
        "ms",
        "4.00\N{MULTIPLICATION SIGN}",
    ]
    assert table[-1].split() == [
        "Successful-question",
        "aggregate",
        "2/2",
        "3000.00",
        "ms",
        "750.00",
        "ms",
        "4.00\N{MULTIPLICATION SIGN}",
    ]


def test_question_type_table_ignores_only_gurobi_license_limits():
    summary = clevr_main.ILPQuestionTypePerformance(
        question_type="query_color",
        attempted=4,
        succeeded=2,
        failed=2,
        full_average_seconds=1.0,
        dynamic_average_seconds=0.25,
        milliseconds_saved=750.0,
        reduction_percent=75.0,
        speedup=4.0,
        answers_agree=True,
        full_average_predicates=27.0,
        dynamic_average_predicates=8.0,
    )
    report = clevr_main.ILPBenchmarkReport(
        comparisons=(object(), object()),
        failures=(
            clevr_main.ILPBenchmarkFailure(
                sample={},
                question_type="query_color",
                error_type="GurobiError",
                error=(
                    "Model too large for size-limited license; "
                    "visit gurobi.com/unrestricted"
                ),
            ),
            clevr_main.ILPBenchmarkFailure(
                sample={},
                question_type="query_color",
                error_type="RuntimeError",
                error="all hypotheses infeasible",
            ),
        ),
        question_types=(summary,),
        attempted=4,
        full_workload_seconds=2.0,
        dynamic_workload_seconds=0.5,
        milliseconds_saved=1500.0,
        reduction_percent=75.0,
        speedup=4.0,
        answers_agree=True,
    )

    table = clevr_main._question_type_table(report)

    # The license-limited sample is omitted from the denominator, while the
    # genuine inference failure remains: two successes out of three runs.
    assert table[2].split()[:2] == ["query_color", "2/3"]
    assert table[-1].split()[:3] == [
        "Successful-question",
        "aggregate",
        "2/2",
    ]


def test_ilp_benchmark_zero_item_limit_runs_all_compatible_questions(monkeypatch):
    samples = ({"question": "one"}, {"question": "two"})
    built = SimpleNamespace(
        name="learned",
        ilp_benchmark_sample=samples[0],
        ilp_benchmark_samples=samples,
    )
    visited = []
    timing = clevr_main.ILPTiming(
        durations_seconds=(1.0,),
        median_seconds=1.0,
        answer=1,
        result_type="count",
        active_concepts=("obj",),
        predicate_count=1,
    )

    def fake_benchmark(_built, _device, **kwargs):
        visited.append(kwargs["sample"])
        return clevr_main.ILPGraphPerformance(
            sample=kwargs["sample"],
            requested_concepts=("obj",),
            full=timing,
            dynamic=timing,
            milliseconds_saved=0.0,
            reduction_percent=0.0,
            speedup=1.0,
            answers_agree=True,
        )

    monkeypatch.setattr(
        clevr_main,
        "benchmark_ilp_graph_activation",
        fake_benchmark,
    )

    report = clevr_main.benchmark_ilp_graph_activations(
        built, "cpu", warmup=0, repeats=1, items=0
    )

    assert visited == list(samples)
    assert len(report.comparisons) == 2


def test_ilp_benchmark_suite_reports_failure_and_continues(monkeypatch):
    samples = (
        {"question": "fails", "program": [{"function": "query_color"}]},
        {"question": "works", "program": [{"function": "count"}]},
    )
    built = SimpleNamespace(
        name="learned",
        ilp_benchmark_sample=samples[0],
        ilp_benchmark_samples=samples,
    )
    timing = clevr_main.ILPTiming(
        durations_seconds=(1.0,),
        median_seconds=1.0,
        answer=1,
        result_type="count",
        active_concepts=("obj",),
        predicate_count=1,
    )

    def fake_benchmark(_built, _device, **kwargs):
        if kwargs["sample"] is samples[0]:
            raise RuntimeError("all hypotheses infeasible")
        return clevr_main.ILPGraphPerformance(
            sample=kwargs["sample"],
            requested_concepts=("obj",),
            full=timing,
            dynamic=timing,
            milliseconds_saved=0.0,
            reduction_percent=0.0,
            speedup=1.0,
            answers_agree=True,
        )

    monkeypatch.setattr(
        clevr_main,
        "benchmark_ilp_graph_activation",
        fake_benchmark,
    )

    report = clevr_main.benchmark_ilp_graph_activations(
        built, "cpu", warmup=0, repeats=1, items=0
    )

    assert report.attempted == 2
    assert len(report.comparisons) == 1
    assert len(report.failures) == 1
    assert report.failures[0].question_type == "query_color"
    assert report.failures[0].error_type == "RuntimeError"
    assert report.answers_agree is True
    query_summary, count_summary = report.question_types
    assert (query_summary.attempted, query_summary.succeeded, query_summary.failed) == (1, 0, 1)
    assert query_summary.full_average_seconds is None
    assert (count_summary.attempted, count_summary.succeeded, count_summary.failed) == (1, 1, 0)


def test_ilp_benchmark_restores_full_graph_after_failure(monkeypatch):
    built, graph, _events, _datanodes = _mock_ilp_benchmark(fail_dynamic=True)
    clock = iter((0.0, 0.5, 1.0, 1.1))
    monkeypatch.setattr(clevr_main, "perf_counter", lambda: next(clock))

    with pytest.raises(RuntimeError, match="dynamic inference failed"):
        clevr_main.benchmark_ilp_graph_activation(
            built, "cpu", warmup=0, repeats=1
        )

    assert graph._active_concepts is None


@pytest.mark.gurobi
def test_post_training_ad_hoc_query_supports_circuit_and_ilp_modes():
    args = SimpleNamespace(
        epochs=1,
        train_items=1,
        eval_items=1,
        device="cpu",
        lr=1e-2,
        tnorm="P",
        seed=0,
        global_constraint_loss_weight=0.1,
        executable_constraint_loss_weight=1.0,
        disable_global_constraint_loss=True,
        gumbel_temp_start=1.0,
        gumbel_temp_end=0.3,
        hard_gumbel=False,
    )
    all_items = clevr_main.load_items()
    # Start with two count questions for the three-mode check, then include a
    # relational attribute query as a direct-ILP grounding regression.
    built = clevr_main.build_program(
        "backend-smoke",
        clevr_main.InferenceProgram,
        [all_items[38], all_items[20], all_items[0]],
        args,
        "cpu",
    )
    # Use the smaller four-object count scene to keep the real ILP smoke fast.
    built.ilp_benchmark_sample = built.eval_dataset[0]

    _sample, comparison = clevr_main.infer_ad_hoc_example(built, "cpu")
    assert list(comparison.results) == ["tnorm", "circuit", "ilp"]
    assert comparison.types_agree is True

    tnorm = comparison.results["tnorm"]
    assert tnorm["type"] == "count"
    assert isinstance(tnorm["answer"], int)
    assert tnorm["mode"] == "tnorm"
    assert tnorm["exact"] is False
    assert isinstance(tnorm["probability"], float)
    assert tnorm["distribution"] is not None

    circuit = comparison.results["circuit"]
    assert circuit["type"] == "count"
    assert isinstance(circuit["answer"], int)
    assert circuit["mode"] == "circuit"
    assert circuit["exact"] is True
    assert isinstance(circuit["probability"], float)
    assert circuit["distribution"] is not None

    ilp = comparison.results["ilp"]
    assert ilp["type"] == "count"
    assert isinstance(ilp["answer"], int)
    assert ilp["mode"] == "ilp"
    assert ilp["exact"] is None
    assert ilp["probability"] is None
    assert ilp["distribution"] is None

    performance = clevr_main.benchmark_ilp_graph_activation(
        built,
        "cpu",
        warmup=0,
        repeats=1,
    )
    assert performance.full.result_type == "count"
    assert performance.dynamic.result_type == "count"
    assert isinstance(performance.full.answer, int)
    assert performance.full.answer == performance.dynamic.answer
    assert performance.answers_agree is True
    assert performance.full.median_seconds > 0
    assert performance.dynamic.median_seconds > 0
    assert len(performance.dynamic.active_concepts) < len(
        performance.full.active_concepts
    )
    assert performance.dynamic.predicate_count < performance.full.predicate_count
    assert built.program.graph._active_concepts is None

    relational_query = built.ilp_benchmark_samples[2]
    relational_performance = clevr_main.benchmark_ilp_graph_activation(
        built,
        "cpu",
        warmup=0,
        repeats=1,
        sample=relational_query,
    )
    assert relational_performance.full.result_type == "query"
    assert relational_performance.dynamic.result_type == "query"
    assert isinstance(relational_performance.full.answer, str)
    assert (
        relational_performance.full.answer
        == relational_performance.dynamic.answer
    )
    assert relational_performance.answers_agree is True
    assert relational_performance.dynamic.predicate_count < (
        relational_performance.full.predicate_count
    )
    assert built.program.graph._active_concepts is None
