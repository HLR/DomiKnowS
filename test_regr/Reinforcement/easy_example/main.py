import argparse
from graph import get_graph
from utils import create_dataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Machine Learning Experiment')
    parser.add_argument('--beta', default='10', type=float, help='Beta parameter')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'], help='Device to use')
    parser.add_argument('--counting_tnorm', choices=['G', 'P', 'L', 'SP'], default='SP', help='The tnorm method to use for the counting constraints')
    parser.add_argument('--atLeastL', default=False, type=bool, help='Use at least L constraint')
    parser.add_argument('--atMostL', default=False, type=bool, help='Use at most L constraint')
    parser.add_argument('--epoch', default=500, type=int, help='Number of training epochs')
    parser.add_argument('--expected_atLeastL', default=3, type=int, help='Expected value for at least L')
    parser.add_argument('--expected_atMostL', default=3, type=int, help='Expected value for at most L')
    parser.add_argument('--expected_value', default=0, type=int, help='Expected value')
    parser.add_argument('--N', default=10, type=int, help='N parameter')
    parser.add_argument('--M', default=8, type=int, help='M parameter')
    parser.add_argument('--model', default='sampling', type=str, help='Model Types [Sampling/PMD/gumbel_pmd/gumbel_sampling]')
    parser.add_argument('--sample_size', default=-1, type=int, help='Sample size for sampling program')
    parser.add_argument('--use_gumbel', default=True, type=bool, help='Enable Gumbel-Softmax')
    parser.add_argument('--initial_temp', default=2.0, type=float, help='Initial temperature for Gumbel-Softmax')
    parser.add_argument('--final_temp', default=0.1, type=float, help='Final temperature for Gumbel-Softmax')
    parser.add_argument('--hard_gumbel', default=False, type=bool, help='Use hard Gumbel-Softmax')
    return parser.parse_args()


def main(args: argparse.Namespace):
    graph, a, b, a_contain_b, b_answer, reward_function = get_graph(args)
    dataset = create_dataset(args.N, args.M)

    # TODO: wire reward_function into the program.
    # TODO: build the program, train, and evaluate.
    # TODO: add any reward-based dataset transformation if needed.

    return graph, dataset


if __name__ == '__main__':
    main(parse_arguments())
