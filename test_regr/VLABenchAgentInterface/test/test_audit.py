from test_regr.VLABenchAgentInterface.main import _print_task_audit_table
from test_regr.VLABenchAgentInterface.world_graph import PRIMITIVE_TASK_PATTERNS


def test_audit_table_covers_every_adapter_task(capsys):
    metrics = {
        "per_task": {
            task: {"episodes": 1, "successes": 1, "progress": 1.0, "ik_failures": 0}
            for task in PRIMITIVE_TASK_PATTERNS
        }
    }

    _print_task_audit_table(metrics)
    rows = capsys.readouterr().out.splitlines()

    assert rows[:2] == [
        "| Task | Success | Progress | IK |",
        "| --- | --- | --- | --- |",
    ]
    assert [row.split("|")[1].strip() for row in rows[2:]] == sorted(PRIMITIVE_TASK_PATTERNS)
    assert all("| 1/1 | 1.000 | 0 |" in row for row in rows[2:])
