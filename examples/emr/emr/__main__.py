import sys
from argparse import ArgumentParser


parser = ArgumentParser(
    description='Entity-Mention-Relation example using `regr`.')
parser.add_argument(
    '--bert',
    action='store_true',
    help='Run the full example with BERT.')
parser.add_argument(
    '--glove',
    action='store_true',
    help='Run the full example with GloVe.')
parser.add_argument(
    '-b', '--bilou',
    action='store_true',
    help='Run the example with BILOU tagging.')
parser.add_argument(
    '-s', '--simple',
    action='store_true',
    help='Run the simple example with "people", "organization", and "work for" relationship between them.')
args = parser.parse_args()

def main():
    if args.simple:
        if __package__ is None or __package__ == '':
            from emr_simple import main
        else:
            from .emr_simple import main
    elif args.bilou:
        if __package__ is None or __package__ == '':
            from emr_bilou import main
        else:
            if __package__ is None or __package__ == '':
                from emr_bilou import main
            else:
                from .emr_bilou import main
    elif args.bert:
        if __package__ is None or __package__ == '':
            from emr_bert import main
        else:
            from .emr_bert import main
    elif args.glove:
        if __package__ is None or __package__ == '':
            from emr_glove import main
        else:
            from .emr_glove import main
    else:
        if __package__ is None or __package__ == '':
            from emr_full import main
        else:
            from .emr_full import main
    return main()


if __name__ == '__main__':
    main()
