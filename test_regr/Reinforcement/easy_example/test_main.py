from pathlib import Path
import sys

RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from graph import get_graph
from main import main
from utils import create_dataset


class _Args:
    beta = 10
    device = 'cpu'
    counting_tnorm = 'SP'
    atLeastL = False
    atMostL = False
    epoch = 500
    expected_atLeastL = 3
    expected_atMostL = 3
    expected_value = 0
    N = 10
    M = 8
    model = 'sampling'
    sample_size = -1
    use_gumbel = True
    initial_temp = 2.0
    final_temp = 0.1
    hard_gumbel = False


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
    assert reward_function('zero').tolist() == [1.0]


def test_main_returns_graph_and_dataset():
    args = _Args()
    graph, dataset = main(args)
    assert graph.name == 'global_PMD'
    assert len(dataset) == 1
