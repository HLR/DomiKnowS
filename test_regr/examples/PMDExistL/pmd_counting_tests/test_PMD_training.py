import pytest
import torch
import numpy as np
from main import parse_arguments, main
from graph import get_graph
from utils import create_dataset, setup_graph, train_model, evaluate_model
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.program.metric import MacroAverageTracker
from domiknows.program.loss import NBCrossEntropyLoss


@pytest.fixture
def setup_environment():
    """Setup environment for each test"""
    np.random.seed(0)
    torch.manual_seed(0)
    device = "cpu"
    yield device


@pytest.fixture
def create_args():
    """Create default arguments for testing"""
    class Args:
        beta = 10.0
        device = "cpu"
        counting_tnorm = "G"
        atLeastL = False
        atMostL = False
        epoch = 50
        expected_atLeastL = 2
        expected_atMostL = 5
        expected_value = 0
        N = 10
        M = 5
        model = "PMD"
        sample_size = -1
    
    return Args()


def test_basic_training(setup_environment, create_args):
    """Test basic training loop executes without errors"""
    device = setup_environment
    args = create_args
    
    graph, a, b, a_contain_b, b_answer = get_graph(args)
    dataset = create_dataset(args.N, args.M)
    
    from main import setup_graph as main_setup_graph
    answer_module = main_setup_graph(args, a, b, a_contain_b, b_answer, device=device)
    
    program = PrimalDualProgram(
        graph, SolverModel, poi=[a, b, b_answer],
        inferTypes=['local/softmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        beta=args.beta, device=device, tnorm="L", counting_tnorm=args.counting_tnorm
    )
    
    train_model(program, dataset, num_epochs=10)
    
    assert answer_module is not None
    assert program is not None


def test_exact_constraint_satisfaction(setup_environment, create_args):
    """Test that exactL constraint is satisfied after training"""
    device = setup_environment
    args = create_args
    args.expected_atLeastL = 2
    args.expected_value = 0
    args.epoch = 100
    
    graph, a, b, a_contain_b, b_answer = get_graph(args)
    dataset = create_dataset(args.N, args.M)
    
    from main import setup_graph as main_setup_graph
    answer_module = main_setup_graph(args, a, b, a_contain_b, b_answer, device=device)
    
    program = PrimalDualProgram(
        graph, SolverModel, poi=[a, b, b_answer],
        inferTypes=['local/softmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        beta=args.beta, device=device, tnorm="L", counting_tnorm=args.counting_tnorm
    )
    
    train_model(program, dataset, num_epochs=2, constr_loss_only=False)
    train_model(program, dataset, num_epochs=args.epoch, constr_loss_only=True)
    
    program.inferTypes = ['local/argmax']
    actual_count = evaluate_model(program, dataset, b_answer).get(args.expected_value, 0)
    
    assert actual_count == args.expected_atLeastL


def test_atleast_constraint_satisfaction(setup_environment, create_args):
    """Test that atLeastL constraint is satisfied after training"""
    device = setup_environment
    args = create_args
    args.atLeastL = True
    args.expected_atLeastL = 2
    args.expected_value = 0
    args.epoch = 100
    
    graph, a, b, a_contain_b, b_answer = get_graph(args)
    dataset = create_dataset(args.N, args.M)
    
    from main import setup_graph as main_setup_graph
    answer_module = main_setup_graph(args, a, b, a_contain_b, b_answer, device=device)
    
    program = PrimalDualProgram(
        graph, SolverModel, poi=[a, b, b_answer],
        inferTypes=['local/softmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        beta=args.beta, device=device, tnorm="L", counting_tnorm=args.counting_tnorm
    )
    
    train_model(program, dataset, num_epochs=2, constr_loss_only=False)
    train_model(program, dataset, num_epochs=args.epoch, constr_loss_only=True)
    
    program.inferTypes = ['local/argmax']
    actual_count = evaluate_model(program, dataset, b_answer).get(args.expected_value, 0)
    
    assert actual_count >= args.expected_atLeastL


def test_atmost_constraint_satisfaction(setup_environment, create_args):
    """Test that atMostL constraint is satisfied after training"""
    device = setup_environment
    args = create_args
    args.atMostL = True
    args.expected_atMostL = 3
    args.expected_value = 0
    args.epoch = 100
    
    graph, a, b, a_contain_b, b_answer = get_graph(args)
    dataset = create_dataset(args.N, args.M)
    
    from main import setup_graph as main_setup_graph
    answer_module = main_setup_graph(args, a, b, a_contain_b, b_answer, device=device)
    
    program = PrimalDualProgram(
        graph, SolverModel, poi=[a, b, b_answer],
        inferTypes=['local/softmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        beta=args.beta, device=device, tnorm="L", counting_tnorm=args.counting_tnorm
    )
    
    train_model(program, dataset, num_epochs=2, constr_loss_only=False)
    train_model(program, dataset, num_epochs=args.epoch, constr_loss_only=True)
    
    program.inferTypes = ['local/argmax']
    actual_count = evaluate_model(program, dataset, b_answer).get(args.expected_value, 0)
    
    assert actual_count <= args.expected_atMostL


@pytest.mark.parametrize("tnorm", ["G", "P", "L", "SP"])
def test_different_tnorms(setup_environment, create_args, tnorm):
    """Test training with different t-norm implementations"""
    device = setup_environment
    args = create_args
    args.counting_tnorm = tnorm
    args.epoch = 50
    
    graph, a, b, a_contain_b, b_answer = get_graph(args)
    dataset = create_dataset(args.N, args.M)
    
    from main import setup_graph as main_setup_graph
    answer_module = main_setup_graph(args, a, b, a_contain_b, b_answer, device=device)
    
    program = PrimalDualProgram(
        graph, SolverModel, poi=[a, b, b_answer],
        inferTypes=['local/softmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        beta=args.beta, device=device, tnorm="L", counting_tnorm=args.counting_tnorm
    )
    
    train_model(program, dataset, num_epochs=10)
    
    assert program is not None