import sys
from argparse import ArgumentParser

if len(__package__) == 0:
    __package__ = 'emr'

from .emr_full import main as emr_main
#from .emr_simple import main as emr_simple_main


parser = ArgumentParser(description='Entity-Mention-Relation example using `regr`.')
parser.add_argument('-s', '--simple',
                    action='store_true',
                    help='Run the simple example with "people", "organization", and "work for" relationship between them.')
args = parser.parse_args()


def main():
    if args.simple:
        return emr_simple_main()
    else:
        return emr_main() 


if __name__ == '__main__':
    main()
