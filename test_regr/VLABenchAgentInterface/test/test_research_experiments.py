"""Independent budget, split and warm-up resume contracts for GPU2 orchestration."""
import hashlib,json
from types import SimpleNamespace
import pytest
import torch
from ..research.experiments import command_matrix,evaluation_matrix,family_episodes


def manifest(tmp_path):
    rows={name:[{"task":"select_fruit","research_family":family,"scene_group":f"{name}:{family}:{i}",
        "planning_facts":{"research":{"evidence":{"episode":str(i),"step":0}}}}
        for family in ("ambiguity","freshness") for i in range(12)] for name in ("train","validation","calibration","test")}
    path=tmp_path/"rollouts.json";path.write_text(json.dumps(rows))
    return {"planning_dir":"planning_train","control_source":"measured_control","rollout_manifest":str(path),"output":str(tmp_path/"runs")},rows


def test_exact_family_budgets_and_variant_groups(tmp_path):
    _,rows=manifest(tmp_path)
    selected=family_episodes(rows["test"],10,42)
    assert len(selected)==20 and len({r["scene_group"] for r in selected})==20
    assert selected==family_episodes(rows["test"],10,42)
    with pytest.raises(ValueError,match="need 13"):family_episodes(rows["test"],13)
    duplicate=[{**row,"variant":"other"} for row in rows["test"]]
    assert len(family_episodes([*rows["test"],*duplicate],10))==20


def test_matched_commands_parse_and_keep_budgets_and_fixed_checkpoints(tmp_path):
    from ..main import build_parser
    spec,_=manifest(tmp_path);matrix=command_matrix(spec)
    assert len(matrix)==6 and {row["seed"] for row in matrix}=={42,43,44}
    parser=build_parser()
    for row in matrix:
        train=parser.parse_args(row["arguments"][3:])
        assert train.sft_epochs==3 and train.controller_warmup_steps==20000 and train.rl_epochs==3
        assert train.controller_warmup_units=="updates" and train.research_eval_episodes_per_family==10
        assert train.eval_rollouts_per_task==1 and train.rl_preflight_min_successful_tasks==1
        evaluations=evaluation_matrix(row,spec)
        assert len(evaluations)==(18 if row["profile"]=="research_v1" else 1)
        paths=set()
        for evaluation in evaluations:
            args=parser.parse_args(evaluation["arguments"][3:]);paths.add(args.agent_checkpoint)
            assert args.research_rollout_split=="test" and args.research_eval_episodes_per_family==10
        assert len(paths)==1
    pilot=command_matrix(spec,pilot=True)
    assert len(pilot)==2
    args=parser.parse_args(pilot[0]["arguments"][3:])
    assert args.sft_epochs==1 and args.controller_warmup_steps==200 and args.research_eval_episodes_per_family==2


def test_warmup_counts_optimizer_updates_and_resumes_without_replaying_committed_updates(tmp_path,monkeypatch):
    from .. import training
    from ..research.control_warmup import warm_controller
    calls=[];fail=[True]
    def train(model,loader,optimizer,*,steps,grad_accumulation,**kwargs):
        calls.append(steps)
        if fail[0] and len(calls)==2:raise RuntimeError("interrupted")
        for _ in range(steps//grad_accumulation):
            optimizer.zero_grad()
            for parameter in model.parameters():parameter.grad=torch.ones_like(parameter)
            optimizer.step()
        return {"loss":1.,"steps":steps}
    monkeypatch.setattr(training,"train_controller_steps",train)
    model=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():model.weight.zero_()
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    kwargs=dict(phases=[("official",[],3)],output=tmp_path,device="cpu",grad_accumulation=2,chunk_updates=1)
    with pytest.raises(RuntimeError,match="interrupted"):warm_controller(model,optimizer,**kwargs)
    fail[0]=False
    resumed=torch.nn.Linear(1,1,bias=False);other=torch.optim.SGD(resumed.parameters(),lr=.1)
    result=warm_controller(resumed,other,**kwargs)
    assert result["official"]["optimizer_updates"]==3
    assert resumed.weight.item()==pytest.approx(-.3)
    assert calls==[2,2,2,2]
    used=len(calls);warm_controller(resumed,other,**kwargs)
    assert len(calls)==used


def test_measured_corpus_hashes_and_completion_are_enforced(tmp_path):
    from ..research.data import validate_control_hashes
    source=tmp_path/"trace.npz";source.write_bytes(b"measured trace fixture")
    (tmp_path/"data_hashes.json").write_text(json.dumps({"trace.npz":hashlib.sha256(source.read_bytes()).hexdigest()}))
    status=tmp_path/"run.json";status.write_text(json.dumps({"status":"running"}))
    with pytest.raises(ValueError,match="not completed"):validate_control_hashes(tmp_path)
    status.write_text(json.dumps({"status":"coverage_ready"}))
    assert validate_control_hashes(tmp_path)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError,match="changed"):validate_control_hashes(tmp_path)
