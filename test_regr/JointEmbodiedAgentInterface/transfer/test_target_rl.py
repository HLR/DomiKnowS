import pytest

from .data import allocation
from .protocol import freeze, read, sha
from .target_rl import advantages, paired_report, predict, score, select_adaptation, target_splits, update
from .target_rl_rollout import executor_proof


def trace(episode, sample, *, invalid=False, input_hash="same"):
    return ({"rollout_id": f"{episode}#{sample}", "episode_id": episode,
             "labels": [3], "prediction": [] if invalid else [{"name": "pick", "params": {}}],
             "input_hash": input_hash, "generation_error": "invalid plan" if invalid else None},
            None, None)


def test_mixed_rewards_count_invalid_emissions_and_match_native_jobs(tmp_path, monkeypatch):
    traces = [trace("reset", 0), trace("reset", 1, invalid=True), trace("other", 0)]

    def fake_run(argv, **kwargs):
        from pathlib import Path
        request = Path(argv[argv.index("--request") + 1])
        result = Path(argv[argv.index("--result") + 1])
        assert [job["rollout_id"] for job in read(request)["jobs"]] == ["reset#0", "other#0"]
        freeze(result, {"request_sha256": sha(request), "rollouts": [
            {"rollout_id": "other#0", "success": False, "termination_reason": "grasp failed"},
            {"rollout_id": "reset#0", "success": True, "termination_reason": "completed"},
        ]})
    monkeypatch.setattr("subprocess.run", fake_run)
    scores = score(traces, tmp_path / "batch", {"split": "training"}, ["simulator"])
    assert [row["success"] for row in scores] == [True, False, False]
    assert advantages(scores) == [.5, -.5, 0.]


def test_pairing_rejects_input_drift(tmp_path):
    baseline = score([trace("reset", 0, invalid=True)], tmp_path / "source", {}, ["unused"])
    changed = score([trace("reset", 0, invalid=True, input_hash="changed")],
                    tmp_path / "adapted", {}, ["unused"])
    assert paired_report(baseline, baseline)["success_rate_delta"] == 0.
    with pytest.raises(ValueError, match="input or reset"):
        paired_report(baseline, changed)


def test_adaptation_requires_native_validation_improvement_and_retention():
    def candidate(epoch, successes, eligible):
        return {"epoch": epoch, "eligible": eligible,
                "target": {"baseline": {"successes": 10}, "adapted": {"successes": successes}}}

    rejected = [candidate(1, 9, True), candidate(2, 10, True), candidate(3, 20, False)]
    assert select_adaptation(rejected) is None
    eligible = candidate(4, 11, True)
    assert select_adaptation([*rejected, eligible, candidate(5, 11, True)]) == eligible


def test_confirmation_indices_stay_out_of_libero_adaptation(tmp_path):
    target = tmp_path / "target.json"
    evaluation = tmp_path / "evaluation.json"
    reservations = tmp_path / "reservations"
    reservations.mkdir()
    freeze(evaluation, {"episodes": []})
    freeze(target, {"dataset": "libero", "integrity": {"passed": True},
                    "reserved_manifest_sha256": sha(evaluation), "episodes": [
                        {"episode_id": "case/state10", "state_index": 10, "split": "training"},
                        {"episode_id": "case/state35", "state_index": 35, "split": "validation"},
                    ]})
    with pytest.raises(ValueError, match="confirmation states"):
        target_splits("libero", target, evaluation, [], reservations)
    rows = allocation("libero", ["case"], exclude_libero_confirmation=True)
    assert [row["index"] for row in rows if row["split"] == "validation"] == list(range(30, 35))


def test_executor_qualification_refuses_changed_code(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("libero_campaign.py", "fixed_executor.py", "libero_protocol.py"):
        (source / name).write_text(name)
    evaluation = tmp_path / "evaluation.json"
    freeze(evaluation, {"episodes": []})
    calibration = tmp_path / "calibration.json"
    freeze(calibration, {"episodes": 100, "success_rate": .91, "manifest_sha256": sha(evaluation),
                         "executor_source_sha256": sha(source / "fixed_executor.py"),
                         "bridge_source_sha256": sha(source / "libero_campaign.py")})
    assert "libero_protocol.py" in executor_proof("libero", source, calibration, evaluation)
    (source / "fixed_executor.py").write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        executor_proof("libero", source, calibration, evaluation)


def test_target_update_reinforces_native_success_and_freezes_eai_head(tmp_path, monkeypatch):
    import torch
    from ..test.test_joint_pipeline import joint_fixture, make_planner
    from ..utils.policy_reference import FrozenPolicyReference
    from .repair import common, objective

    examples, runtime = joint_fixture.__wrapped__()
    planner = make_planner(runtime)
    frozen_head = planner.label_heads["eai"]
    frozen_head.requires_grad_(False)
    head_before = {name: value.clone() for name, value in frozen_head.state_dict().items()}
    parameters = [p for p in planner.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=.1)
    reference = FrozenPolicyReference(planner, optimizer)
    vocab = runtime.vlabench_vocabulary
    good = [vocab.label_for_token(token) for token in
            ("skill:pick", "arg:target_entity_name", "obj:0", "<eos>")]
    bad = [*good[:2], vocab.label_for_token("obj:1"), vocab.eos_label]
    labels = iter((good, bad))
    monkeypatch.setattr(planner, "sample_labels", lambda *args, **kwargs: (next(labels), None))
    public = tmp_path / "public.json"
    freeze(public, {"instruction": "pick the object", "images": [], "entities": ["entity_000"]})
    rows = [{"episode_id": "reset", "generation_input": str(public), "files": {str(public): sha(public)}}]
    traces = predict(planner, runtime, rows, "none", 2, False)
    assert traces[0][0]["prediction"][0]["params"] == {"target_entity_name": 0}
    assert traces[1][0]["generation_error"]
    bound, policy = traces[0][1:]
    records = [{**traces[0][0], "success": True}, {**traces[1][0], "success": False}]
    seen = []

    def anchor(domain):
        def loss(*args):
            seen.append(domain)
            return parameters[0].sum() * 0., {}
        return loss

    monkeypatch.setattr(common, "source_context", lambda *args: bound)
    monkeypatch.setattr(objective, "planning_loss", anchor("vlabench"))
    monkeypatch.setattr(objective, "eai_loss", anchor("eai"))
    def successful_plan_margin():
        planner.eval()
        with torch.no_grad():
            good_logp, bad_logp = [planner.replay_labels_logprob("vlabench", bound, plan, policy, max_steps=61)
                                   for plan in (good, bad)]
        return (good_logp - bad_logp).item()

    before = successful_plan_margin()
    source = ({"item": {"target_plan_labels": torch.tensor(good)}}, None)
    assert update(planner, runtime, optimizer, reference, traces, records, source, examples[0], (.25, .5))
    assert successful_plan_margin() > before
    assert seen == ["vlabench", "eai"]
    assert all(torch.equal(value, head_before[name]) for name, value in frozen_head.state_dict().items())
