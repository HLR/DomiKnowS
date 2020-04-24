import sys
from argparse import ArgumentParser


parser = ArgumentParser(
    description='Entity-Mention-Relation example using `regr`.')
args = parser.parse_args()

def main():
    if __package__ is None or __package__ == '':
        from emr_bert import main
    else:
        from .emr_bert import main
    return main()


if __name__ == '__main__':
    main()
