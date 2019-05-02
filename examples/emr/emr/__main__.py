if __package__ is None or __package__ == '':
    # uses current directory visibility
    from emr import main as emr_main
    from ner import main as ner_main
else:
    # uses current package visibility
    from .emr import main as emr_main
    from .ner import main as ner_main

def main():
    emr_main() # TODO: add option to run ner_main


if __name__ == '__main__':
    main()
