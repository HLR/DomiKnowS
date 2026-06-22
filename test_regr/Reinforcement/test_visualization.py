"""End-to-end test of the Flask reinforcement visualizer.

Builds a tiny ReinforcementProgram, attaches a ReinforcementVisualizer, runs
training in a background thread, and drives the HTTP API (state/next) from the
main thread — verifying that (a) training is gated by the UI, and (b) each step
payload contains the decoding, sampled decodings, generated sample, reward, and
loss.
"""
import json
import socket
import threading
import time
import urllib.request
from pathlib import Path
import sys

import pytest
import torch

pytest.importorskip("flask")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domiknows.graph import Graph, Concept, Relation, EnumConcept
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.program import ReinforcementProgram
from domiknows.reinforcement import ReinforcementVisualizer


class _Net(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.l = torch.nn.Linear(n, 2)

    def forward(self, rel, x):
        return self.l(x)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _build_program():
    Graph.clear(); Concept.clear(); Relation.clear()
    N, M = 5, 6
    with Graph("viz_test") as graph:
        a = Concept(name="a")
        b = Concept(name="b")
        (a_contain_b,) = a.contains(b)
        b_answer = b(name="answer_b", ConceptClass=EnumConcept, values=["zero", "one"])

    a["index"] = ReaderSensor(keyword="a")
    b["index"] = ReaderSensor(keyword="b")
    b[a_contain_b] = EdgeSensor(b["index"], a["index"], relation=a_contain_b,
                               forward=lambda bb, _: torch.ones(len(bb)).unsqueeze(-1))
    b[b_answer] = ModuleLearner(a_contain_b, "index", module=_Net(N))

    def reward_function(generator_output):
        zeros = sum(1 for x in generator_output if int(x) == 0)
        return torch.tensor([1.0 if zeros >= 1 else 0.0])

    program = ReinforcementProgram(
        graph, targets=[b_answer], reward_function=reward_function,
        num_samples=4, poi=[a, b, b_answer], device="cpu",
    )
    dataset = [{"a": [0], "b": [[float((i + j) % 3 - 1) for j in range(N)] for i in range(M)]}]
    return program, dataset


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode())


def _post(url):
    req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())


def test_visualizer_gates_steps_and_reports_details():
    torch.manual_seed(0)
    program, dataset = _build_program()

    port = _free_port()
    viz = ReinforcementVisualizer(port=port, auto_open=False)
    viz.attach(program)
    viz.start()
    base = f"http://127.0.0.1:{port}"

    def _train():
        program.train(dataset, train_epoch_num=3,
                      Optim=lambda p: torch.optim.Adam(p, lr=5e-3), device="cpu")

    t = threading.Thread(target=_train, daemon=True)
    t.start()

    seen = []
    deadline = time.time() + 30
    while time.time() < deadline:
        st = _get(base + "/api/state")
        if st["status"] == "waiting" and st["state"] is not None:
            seen.append(st["state"])
            _post(base + "/api/next")          # advance one gated step
        elif st["status"] == "done":
            break
        time.sleep(0.05)

    t.join(timeout=10)

    # We gated and observed at least the 3 training steps.
    assert len(seen) >= 3, f"expected >=3 gated steps, saw {len(seen)}"
    final = _get(base + "/api/state")
    assert final["status"] == "done"

    # Each payload carries the full per-step detail.
    p = seen[0]
    assert isinstance(p["loss"], (int, float))
    assert 0.0 <= p["mean_reward"] <= 1.0
    assert p["estimator"] in ("importance_weighted", "reinforce")
    assert p["targets"] and p["targets"][0]["concept"] == "answer_b"
    assert p["targets"][0]["probabilities"]            # decoding distribution
    assert len(p["samples"]) == 4
    s0 = p["samples"][0]
    assert "answer_b" in s0["assignment_labels"]       # sampled decoding
    assert s0["generator_output"] is not None          # generated sample
    assert "reward" in s0 and "logprob" in s0          # applied reward + logp


def test_stop_aborts_training():
    torch.manual_seed(0)
    program, dataset = _build_program()
    port = _free_port()
    # exit_on_stop=False so the test process is not terminated; training still aborts.
    viz = ReinforcementVisualizer(port=port, auto_open=False, exit_on_stop=False)
    viz.attach(program)
    viz.start()
    base = f"http://127.0.0.1:{port}"

    finished = {"ok": False}

    def _train():
        program.train(dataset, train_epoch_num=1000,   # would run a long time
                      Optim=lambda p: torch.optim.Adam(p, lr=5e-3), device="cpu")
        finished["ok"] = True

    t = threading.Thread(target=_train, daemon=True)
    t.start()

    # Wait for the first gated step, then stop.
    deadline = time.time() + 15
    while time.time() < deadline:
        if _get(base + "/api/state")["status"] == "waiting":
            break
        time.sleep(0.05)
    _post(base + "/api/stop")

    t.join(timeout=10)
    assert finished["ok"], "training thread did not exit after Stop"
    assert _get(base + "/api/state")["status"] == "stopped"


def test_play_mode_runs_without_gating():
    torch.manual_seed(0)
    program, dataset = _build_program()
    port = _free_port()
    viz = ReinforcementVisualizer(port=port, auto_open=False, default_mode="play", delay=0.0)
    viz.attach(program)
    viz.start()
    base = f"http://127.0.0.1:{port}"

    done = {"ok": False}

    def _train():
        program.train(dataset, train_epoch_num=2,
                      Optim=lambda p: torch.optim.Adam(p, lr=5e-3), device="cpu")
        done["ok"] = True

    t = threading.Thread(target=_train, daemon=True)
    t.start()
    t.join(timeout=20)

    assert done["ok"], "play mode should not block training"
    assert _get(base + "/api/state")["status"] == "done"
