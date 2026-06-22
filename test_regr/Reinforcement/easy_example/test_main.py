from pathlib import Path
import sys

import numpy as np
import torch

RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from graph import get_graph
from main import main, build_program
from utils import create_dataset


class _Args:
    device = 'cpu'
    atLeastL = False
    atMostL = False
    epoch = 50
    expected_atLeastL = 3
    expected_atMostL = 3
    expected_value = 0
    N = 10
    M = 8
    model = 'sampling'
    num_samples = 32
    lr = 5e-3


def test_create_dataset_shape():
    dataset = create_dataset(10, 8)
    assert len(dataset) == 1
    assert len(dataset[0]['a']) == 1
    assert len(dataset[0]['b']) == 8
    assert len(dataset[0]['label']) == 8


def test_reward_function_from_graph():
    args = _Args()
    graph, a, b, a_contain_b, b_answer, reward_function = get_graph(args)
    assert graph.name == 'global_PMD'
    # The default reward is an exact-count constraint: reward is 1.0 only when
    # exactly `expected_atLeastL` (3) answers equal the expected value (0/"zero").
    assert reward_function(['zero', 'zero', 'zero']).tolist() == [1.0]
    assert reward_function(['zero']).tolist() == [0.0]


def test_main_returns_graph_and_dataset():
    args = _Args()
    graph, dataset = main(args)
    assert graph.name == 'global_PMD'
    assert len(dataset) == 1


def _train_and_measure(model_kind, epoch=120):
    np.random.seed(0)
    torch.manual_seed(0)
    args = _Args()
    args.model = model_kind
    args.epoch = epoch
    program, dataset, _b_answer = build_program(args, device='cpu')
    baseline = program.evaluate_reward(dataset, num_samples=200)
    program.train(
        dataset,
        train_epoch_num=args.epoch,
        Optim=lambda p: torch.optim.Adam(p, lr=args.lr),
        device='cpu',
    )
    trained = program.evaluate_reward(dataset, num_samples=200)
    return baseline, trained


def test_importance_weighted_estimator_improves_reward():
    baseline, trained = _train_and_measure('sampling')
    assert 0.0 <= baseline <= 1.0 and 0.0 <= trained <= 1.0
    assert trained > baseline, f"importance_weighted reward did not improve: {baseline} -> {trained}"


def test_reinforce_estimator_improves_reward():
    baseline, trained = _train_and_measure('reinforce')
    assert 0.0 <= baseline <= 1.0 and 0.0 <= trained <= 1.0
    assert trained > baseline, f"reinforce reward did not improve: {baseline} -> {trained}"


def test_training_loss_is_finite():
    np.random.seed(0)
    torch.manual_seed(0)
    args = _Args()
    program, dataset, _b_answer = build_program(args, device='cpu')
    program.to('cpu')
    program.opt = torch.optim.Adam(program.model.parameters(), lr=args.lr)
    losses = [loss for loss, _metric, _dn in program.train_epoch(dataset)]
    assert losses and all(torch.isfinite(l) for l in losses)
