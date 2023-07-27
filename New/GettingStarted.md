# DomiKnowS: Getting Started

Welcome to DomiKnowS, a library for integrating symbolic domain knowledge in deep learning! This readme provides a quick overview of DomiKnowS and step-by-step instructions on how to get started with the library.

## What is DomiKnowS?

DomiKnowS is a Python library that enables the integration of symbolic domain knowledge into deep learning models. It allows you to express domain knowledge in the form of logical constraints over model outputs or latent variables, which can improve model interpretability, robustness, and generalization. DomiKnowS is built on top of popular deep learning frameworks such as TensorFlow and PyTorch, making it easy to incorporate into your existing deep learning workflows.

## Installation

To install DomiKnowS, you can use `pip`, the Python package manager. Open a terminal and run the following command:

```bash
pip install DomiKnowS
```


Note that DomiKnowS relies on Gurobi to solve the inference-time optimization. You need to install Gurobi separately by following the instructions [here](link_to_gurobi_readme).

## Requirements

The DomiKnowS library is integrated with the ILP solver [Gurobi](https://www.gurobi.com/). The library provides necessary capability to find solutions for logical constraints encoded as an ILP model.

The [requirements](https://github.com/HLR/DomiKnowS/blob/main/requirements.txt) file for the library includes the Gurobi Python package [gurobipy](https://pypi.org/project/gurobipy/).
This package comes with a trial license that allows solving problems of limited size. 
As a student or staff member of an academic institution, you qualify for a free, full product license. For more information, see:

- [https://www.gurobi.com/academia/academic-program-and-licenses/](https://www.gurobi.com/academia/academic-program-and-licenses/)

For a commercial evaluation, you can request an evaluation license.


## Examples

To help you get started quickly, we provide several examples in [run with jupytor](https://github.com/HLR/DomiKnowS/tree/Doc/Run%20With%20Jupyter)) section of this repository. These examples demonstrate how to use DomiKnowS for various tasks, including classification, regression, and optimization problems.

Refer to documentation!

