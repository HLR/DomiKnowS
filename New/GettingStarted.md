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

## Usage

Using DomiKnowS involves the following steps:

1. **Graph Declaration**: Define the structure of the data symbolically via graph declarations using DomiKnowS. This step allows you to express domain knowledge in the form of logical constraints over outputs or latent variables.

2. **Model Declaration**: Declare the deep learning model architecture using DomiKnowS. This step involves specifying the neural network layers, activation functions, and other relevant configurations.

3. **Program Initialization**: Initialize the DomiKnowS program by specifying the learning problem you want to solve. This step involves setting up the optimization objective, constraints, concepts, and other necessary configurations.

4. **Program Composition and Execution**: Compose and execute the DomiKnowS program to train the deep learning model with integrated domain knowledge. This step involves solving the optimization problem and updating the model's parameters iteratively.

For detailed documentation on how to use DomiKnowS, please refer to the [official documentation](link_to_official_documentation).

## Examples

To help you get started quickly, we provide several examples in [run with jupytor](https://github.com/HLR/DomiKnowS/tree/Doc/Run%20With%20Jupyter)) section of this repository. These examples demonstrate how to use DomiKnowS for various tasks, including classification, regression, and optimization problems.

Refer to documentation!

