# Getting Started

Clone the repository and cd into Domiknows folder. Install the requirements by running:

```
python -m pip install -r requirements.txt
```

### Verification

To ensure that Domiknows works correctly,we can verify the installation by running sample Domiknows example code.

Following is an example of a simple classifier:

```
import sys
sys.path.append("Domiknows/regr")
import logging,torch

Graph.clear()
Concept.clear()
Relation.clear()

with Graph(name='global') as graph:
    TODO

def main():
    TODO

if __name__ == '__main__':
    main()
```



### Dependencies

- [Ubuntu](https://ubuntu.com/) 18.04
- [Python](https://www.python.org/) 3.7
- [PyTorch](https://pytorch.org/) 1.4.0
- [Gurobi](https://gurobi.com/) 8.0
- [Graphviz](https://graphviz.org/)
- Other dependencies specified in [`requirements.txt`](https://github.com/HLR/DomiKnowS/blob/develop_newLC/requirements.txt), that are installed by `pip`.
