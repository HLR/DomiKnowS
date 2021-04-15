# Install

- [Install](#install)
  - [Requirements](#requirements)
    - [Dependencies](#dependencies)
    - [`PYTHONPATH`](#pythonpath)
    - [Additional for testing](#additional-for-testing)
    - [Additional for Examples](#additional-for-examples)
  - [Testing](#testing)
  - [Examples](#examples)

## Requirements

### Dependencies

- [Ubuntu](https://ubuntu.com) 18.04
- [python](https://www.python.org) 3.7
- [PyTorch](https://pytorch.org) 1.4.0
- [Gurobi](https://gurobi.com) 8.0
- [graphviz](https://graphviz.org/)
- other dependencies specified in [`requirements.txt`](/requirements.txt), that are installed by `pip`.

```bash
python -m pip install -r requirements.txt
```

The framework is built and tested on above specific version of softwares.

### `PYTHONPATH`

Add project root to python module search path, for example, by adding it to environment variable [`PYTHONPATH`](https://docs.python.org/3.7/using/cmdline.html#envvar-PYTHONPATH).

```bash
cd path/to/DomiKnowS/
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Additional for testing

We test with [pytest](https://pytest.org).
Additional dependencies for testing in [`requirements-dev.txt`](/requirements-dev.txt), that can be installed by `pip`.

```bash
python -m pip install -r requirements-dev.txt
```

### Additional for Examples

Make sure to check out if there is any **additional prerequirements** or setup steps in specific example directory `README.md`, if you want to run an example.

## Testing

Testing cases are in [`test_regr`](/test_regr)

After installing [dependency of testing](#additional-for-testing), one can run test cases with [pytest](https://pytest.org):

```bash
cd path/to/DomiKnowS/
pytest
```

## Examples

There are some [examples](/examples) to see if the design really work.

- [Entity Relation Extraction with CoNLL04](/examples/emr)
- [Entity Relation Extraction with CoNLL04 (AllenNLP)](/examples/emr-allennlp)
- [Entity Relation Extraction with ACE05](/examples/new_interface)
- [Entity Relation Extraction with ACE05 + class hierarchy](/examples/hierarchyACE)
- [Spatial Role Labeling with CLEF 2017 mSpRL](/examples/SpRL)
