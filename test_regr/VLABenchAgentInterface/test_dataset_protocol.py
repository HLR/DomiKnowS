"""Independent counterexamples for GPU2 dataset separation and composite state loading."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from test_regr.VLABenchAgentInterface.dataset_protocol import assert_training_dataset

def episode(root, name, seed=1, instruction="Put fruit in the plate"):
    p=root/name
    for s in ["env_config","input","output"]:(p/s).mkdir(parents=True,exist_ok=True)
    (p/"env_config/env_config.json").write_text(json.dumps({"task":{"seed":seed,"components":[{"name":"fruit","position":[seed*0.01,0,0]},{"name":"plate"}]}}))
    (p/"input/instruction.txt").write_text(instruction)
    Image.new("RGB",(3,3)).save(p/"input/view.png")
    (p/"output/operation_sequence.json").write_text(json.dumps({"skill_sequence":[{"name":"pick","params":{"target_entity_name":"fruit"}}]}))
    return p

def role(root, value="training"):
    root.mkdir(parents=True,exist_ok=True)
    (root/"dataset_role.json").write_text(json.dumps({"role":value,"ready":True,"provenance":{"kind":"independent simulator reset","seed":1000}}))

def test_official_root_is_never_training(tmp_path):
    official=tmp_path/"official";episode(official,"ep")
    role(official,"training") # Marker alone must not override actual identity.
    with pytest.raises(ValueError,match="Official VLABench"):assert_training_dataset(official,protected_root=official)

def test_renaming_and_reformatting_do_not_hide_evaluation_overlap(tmp_path):
    official=tmp_path/"official";source=episode(official,"ep")
    train=tmp_path/"training";shutil.copytree(source,train/"renamed")
    p=train/"renamed/env_config/env_config.json";p.write_text(json.dumps(json.loads(p.read_text()),indent=4,sort_keys=True))
    role(train)
    with pytest.raises(ValueError,match="overlap"):assert_training_dataset(train,protected_root=official)

def test_new_scene_is_accepted(tmp_path):
    official=tmp_path/"official";episode(official,"ep",seed=1)
    train=tmp_path/"training";episode(train,"different",seed=1000);role(train)
    assert assert_training_dataset(train,protected_root=official)["official_evaluation_overlap"]==0

def test_evaluation_role_rejected_even_for_a_different_scene(tmp_path):
    train=tmp_path/"training";episode(train,"ep",seed=1000);role(train,"evaluation")
    with pytest.raises(ValueError,match="role"):assert_training_dataset(train,protected_root=tmp_path/"unused")

def test_unclassified_data_rejected(tmp_path):
    train=tmp_path/"training";episode(train,"ep")
    with pytest.raises(ValueError,match="Unclassified"):assert_training_dataset(train,protected_root=tmp_path/"unused")

def test_duplicate_scene_rejected(tmp_path):
    train=tmp_path/"training";episode(train,"a");episode(train,"b");role(train)
    with pytest.raises(ValueError,match="Duplicate"):assert_training_dataset(train,protected_root=tmp_path/"unused")

def test_joint_training_rejects_official_data_before_models(tmp_path):
    from test_regr.JointEmbodiedAgentInterface.main import command_train_agent
    with pytest.raises(ValueError,match="Official VLABench"):
        command_train_agent(SimpleNamespace(vlabench_planning_dir="/workspace/test_regr/VLABenchAgentInterface/data/planning",output=str(tmp_path)))

@pytest.mark.parametrize("name",["command_train_agent","command_train_planner","command_fit_task_classifier"])
def test_standalone_training_rejects_official_data_before_models(tmp_path,name):
    from test_regr.VLABenchAgentInterface import main
    with pytest.raises(ValueError,match="Official VLABench"):
        getattr(main,name)(SimpleNamespace(planning_dir="/workspace/test_regr/VLABenchAgentInterface/data/planning",output=str(tmp_path)))

def test_composite_uses_end_effector_state_and_preserves_language_id():
    from test_regr.VLABenchAgentInterface.dataset import LeRobotWindowDataset
    record={"observation.state.eef":[.1,.2,.3,.4,.5,.6,1.],"observation.state.joint":[9.]*7,"actions":[0.]*7,"images":np.zeros((3,8,8,3),dtype=np.uint8),"episode_index":17,"frame_index":0,"task_index":166}
    ds=LeRobotWindowDataset([record],observation_horizon=1,action_horizon=1)
    row=ds[0]
    assert np.allclose(row["state"][0].numpy(),record["observation.state.eef"])
    assert row["task_index"].item()==166
    assert row["plan_context"].tolist()==[0,0,0] # Unknown boundaries are not fabricated.

def test_composite_loader_excludes_metadata_parquet(tmp_path,monkeypatch):
    import datasets
    from test_regr.VLABenchAgentInterface.dataset import load_hf_control_records
    (tmp_path/"meta").mkdir();(tmp_path/"data").mkdir()
    (tmp_path/"meta/info.json").write_text(json.dumps({"features":{"observation.state.eef":{}}}))
    (tmp_path/"data/actions.parquet").touch();(tmp_path/"meta/tasks.parquet").touch()
    calls=[]
    def load(*args,**kw):calls.append((args,kw));return "loaded"
    monkeypatch.setattr(datasets,"load_dataset",load)
    assert load_hf_control_records(tmp_path,task="all")=="loaded"
    assert calls[0][1]["data_files"]=={"train":[str(tmp_path/"data/actions.parquet")]}

def test_composite_controller_sizes_language_embedding_from_metadata(tmp_path,monkeypatch):
    from test_regr.VLABenchAgentInterface import main
    monkeypatch.setattr(main,"load_control_task_instructions",lambda _: {0:"a",166:"b"})
    args=SimpleNamespace(tiny_vision=True,vision_dim=8,action_horizon=1,hidden_dim=8,max_views=3,control_source=str(tmp_path))
    model=main._controller(args,"cpu")
    assert model.task_embedding.num_embeddings==167


def test_untracked_resume_rejected(tmp_path):
    from test_regr.VLABenchAgentInterface.dataset_protocol import guard_training_dataset
    train=tmp_path/"training";episode(train,"ep",seed=98765);role(train)
    with pytest.raises(ValueError,match="no verified"):
        guard_training_dataset(train,resume=tmp_path/"old/model.pt")

def test_historical_output_is_not_relabelled_clean(tmp_path):
    from test_regr.VLABenchAgentInterface.dataset_protocol import guard_training_dataset
    train=tmp_path/"training";episode(train,"ep",seed=98765);role(train)
    output=tmp_path/"old_output";output.mkdir()
    marker=output/"dataset_protocol.json";marker.write_text(json.dumps({"official_evaluation_training_seen":True}))
    with pytest.raises(ValueError,match="new output"):
        guard_training_dataset(train,output)
    assert json.loads(marker.read_text())["official_evaluation_training_seen"] is True

def test_incomplete_training_corpus_rejected(tmp_path):
    train=tmp_path/"training";episode(train,"ep");role(train)
    marker=train/"dataset_role.json";v=json.loads(marker.read_text());v["ready"]=False;marker.write_text(json.dumps(v))
    with pytest.raises(ValueError,match="incomplete"):
        assert_training_dataset(train,protected_root=tmp_path/"unused")


def test_highest_composite_instruction_id_runs_controller(monkeypatch):
    import torch
    from test_regr.VLABenchAgentInterface import main
    monkeypatch.setattr(main,"load_control_task_instructions",lambda _: {0:"a",166:"b"})
    args=SimpleNamespace(tiny_vision=True,vision_dim=8,action_horizon=1,hidden_dim=8,max_views=3,control_source="composite")
    model=main._controller(args,"cpu").eval()
    with torch.no_grad():
        out=model(torch.zeros(1,2,3,3,32,32),torch.zeros(1,2,7),torch.tensor([166]))
    assert list(out.shape)==[1,1,7]
    assert torch.isfinite(out).all()

def test_composite_embedding_does_not_silently_load_primitive_weights(monkeypatch):
    from test_regr.VLABenchAgentInterface import main
    args=SimpleNamespace(tiny_vision=True,vision_dim=8,action_horizon=1,hidden_dim=8,max_views=3,control_source="primitive")
    monkeypatch.setattr(main,"load_control_task_instructions",lambda _: {0:"a",127:"b"})
    old=main._controller(args,"cpu")
    monkeypatch.setattr(main,"load_control_task_instructions",lambda _: {0:"a",166:"b"})
    new=main._controller(args,"cpu")
    with pytest.raises(RuntimeError,match="task_embedding.weight"):
        new.load_state_dict(old.state_dict())


def test_rewriting_instruction_does_not_make_an_official_scene_clean(tmp_path):
    official=tmp_path/"official";source=episode(official,"ep")
    train=tmp_path/"training";shutil.copytree(source,train/"renamed");role(train)
    (train/"renamed/input/instruction.txt").write_text("A newly written instruction for the same test scene")
    with pytest.raises(ValueError,match="overlap"):
        assert_training_dataset(train,protected_root=official)


def test_fixed_simulator_evaluation_scene_is_reserved(tmp_path):
    train=tmp_path/"training";e=episode(train,"ep",seed=300);role(train)
    tracks=tmp_path/"tracks";tracks.mkdir()
    config=json.loads((e/"env_config/env_config.json").read_text())
    (tracks/"track_1.json").write_text(json.dumps({"select_fruit":[config]}))
    with pytest.raises(ValueError,match="overlap"):
        assert_training_dataset(train,protected_root=tmp_path/"unused",simulator_root=tracks)


def test_clean_checkpoint_lineage_can_resume_from_a_dotted_output_directory(tmp_path):
    from test_regr.VLABenchAgentInterface.dataset_protocol import guard_training_dataset
    train=tmp_path/"training";episode(train,"ep",seed=98765);role(train)
    output=tmp_path/"clean.run"
    guard_training_dataset(train,output)
    assert (output/"dataset_protocol.json").is_file()
    checkpoint=output/"epoch_001.pt";checkpoint.touch()
    assert guard_training_dataset(train,resume=checkpoint)["official_evaluation_training_seen"] is False

def test_reference_plan_tampering_invalidates_a_frozen_corpus(tmp_path):
    import hashlib
    train=tmp_path/"training";ep=episode(train,"ep",seed=300);role(train)
    p=ep/"output/operation_sequence.json"
    manifest=train/"dataset_manifest.json";manifest.write_text(json.dumps({"files":{str(p.relative_to(train)):hashlib.sha256(p.read_bytes()).hexdigest()}}))
    marker=train/"dataset_role.json";data=json.loads(marker.read_text());data["manifest_sha256"]=hashlib.sha256(manifest.read_bytes()).hexdigest();marker.write_text(json.dumps(data))
    p.write_text(json.dumps({"skill_sequence":[]}))
    with pytest.raises(ValueError,match="file changed"):
        assert_training_dataset(train,protected_root=tmp_path/"unused")
