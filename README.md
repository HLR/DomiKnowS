# DomiKnowS: Declarative Knowledge Integration with Deep Neural Models

DomiKnowS is a Python library that facilitates the integration of domain knowledge in deep learning architectures. With DomiKnowS, you can express the structure of your data symbolically via graph declarations and seamlessly add logical constraints over outputs or latent variables to your deep models. This allows you to define domain knowledge explicitly, improving your models' explainability, performance, and generalizability, especially in low-data regimes. 

While several approaches for integrating symbolic and sub-symbolic models have been introduced, no generic library facilitates programming for such integration with various underlying algorithms. DomiKnowS aims to simplify the programming for knowledge integration in training and inference phases while separating the knowledge representation from learning algorithms.

## Branch Status

| Branch | CI Status |
|--------|-----------|
| main | [![CI](https://github.com/HLR/DomiKnowS/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HLR/DomiKnowS/actions) |
| develop | [![CI](https://github.com/HLR/DomiKnowS/actions/workflows/CI.yml/badge.svg?branch=develop)](https://github.com/HLR/DomiKnowS/actions) |

## Contents

- [Getting Started](https://github.com/HLR/DomiKnowS/blob/Doc/Getting%20Started.md): Provides detailed instructions on how to get started with DomiKnowS, including installation, setting up the environment, and basic usage.
- [Example Tasks](https://github.com/HLR/DomiKnowS/blob/Doc/Getting%20Started.md): Contains examples that demonstrate the usage of DomiKnowS for various tasks, such as image classification, sequence modeling, and reinforcement learning. ( For more example see [Examples Branch](https://github.com/HLR/DomiKnowS/tree/Tasks) )
- [Documentation](https://github.com/HLR/DomiKnowS/tree/Doc): Provides comprehensive documentation on the DomiKnowS, including classes, methods, and their usage.
- [Contributing](https://github.com/HLR/DomiKnowS/blob/Doc/Suggestions.md): Explains how you can contribute to the development of DomiKnowS, including reporting issues, suggesting enhancements, and submitting pull requests.
- [License](https://github.com/HLR/DomiKnowS/blob/Doc/LICENSE.md): Contains information about the license of DomiKnowS and its terms of use.
- [DomiKnowS Website](https://hlr.github.io/domiknows-nlp/): Contains documentation, example links, and an introductory video to DomiKnowS


## Quick Start

1. Install DomiKnowS using `pip install DomiKnowS`.
2. Install Gurobi following the instructions [here](https://github.com/HLR/DomiKnowS/blob/develop/GurobiREADME.md).
3. Refer to the [Getting Started](https://github.com/HLR/DomiKnowS/blob/Doc/Getting%20Started.md) documentation for detailed instructions on how to define graph declarations, model declarations, initialize programs, and compose and execute programs using DomiKnowS.

## Executable Query Inference

Runtime queries are evaluated from a `DataNode` through two related APIs:

- `DataNode.inferExecutableResults(...)` is the generic query-answering
  interface. It supports registered and temporary ad hoc executable queries
  using t-norm traversal, circuit inference, or ILP through its `mode`
  argument.
- `DataNode.inferILPResults(...)` is the existing lower-level, in-place ILP
  inference method. It builds and solves the constrained model, writes the
  selected `<concept>/ILP` assignments, and persists answers for active
  executable constraints.

Calling `inferExecutableResults(mode="ilp", ...)` prepares the executable query
and delegates the actual constrained solving to `DataNode.inferILPResults()`.
See the [DataNode inference-method documentation](domiknows/graph/README.md#datanode-inference-methods)
for signatures, return values, ad hoc queries, and examples.

## Inference Explanations and Counterfactuals

`DataNode.explainInference(..., mode="ilp" | "circuit" | "tnorm")` returns
isolated, JSON-safe reports with grounded rule evidence. Use
`report.counterfactual({prediction_id: label})` to force ILP labels, condition
circuit queries, or replace t-norm inputs without changing existing predictions.
See the [explanation workflow documentation](domiknows/solver/explanation/README.md)
and [three-mode example](test_regr/examples/explanation_demo.py).

## Incremental Inference

`DataNode.incrementalInference(..., mode="ilp" | "circuit" | "tnorm")` creates
a persistent session for changing predictions, entities, relations, and rules.
Call `session.infer()` to detect changes automatically and publish normal outputs,
or `session.infer(populate=False)` for detached results. See the
[incremental workflow documentation](domiknows/solver/incremental/README.md)
and [three-mode example](test_regr/examples/incremental_demo.py).

## Constraint-Guided Active Learning

`ConstraintActiveLearner` selects whole examples for annotation using uncertainty,
grounded constraint conflict, and available constrained-label disagreement. It
returns ranked entity/relation targets, label suggestions, and JSON-safe evidence,
with optional diversity and persistent pool sessions. See the
[active-learning documentation](domiknows/solver/active_learning/README.md)
and [annotation-loop example](test_regr/examples/active_learning_demo.py).

## Constraint Induction with Human Review

`ConstraintInducer` proposes allowlisted logical rules from annotations, model
predictions, and natural-language domain guidance. It separates discovery and
held-out evidence, supports isolated impact previews, and requires an explicit
review decision before activation. See the
[induction documentation](domiknows/solver/constraint_induction/README.md)
and [review workflow example](test_regr/examples/constraint_induction_demo.py).

## Publications

- [DomiKnowS: A Library for Integration of Symbolic Domain Knowledge in Deep Learning](https://arxiv.org/abs/2108.12370)
- [GLUECons: A Generic Benchmark for Learning Under Constraints](https://arxiv.org/abs/2302.10914)


## Acknowledgements

DomiKnowS is developed and maintained by [HLR](https://hlr.github.io/). We would like to acknowledge the contributions of the open-source community and express our gratitude to the developers of Gurobi for their excellent optimization solver.

## Citation

If you use DomiKnowS in your research or work, please cite our paper:

```
@inproceedings{rajaby-faghihi-etal-2021-domiknows,
    title = "{D}omi{K}now{S}: A Library for Integration of Symbolic Domain Knowledge in Deep Learning",
    author = "Rajaby Faghihi, Hossein  and
      Guo, Quan  and
      Uszok, Andrzej  and
      Nafar, Aliakbar  and
      Kordjamshidi, Parisa",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-demo.27",
    doi = "10.18653/v1/2021.emnlp-demo.27",
    pages = "231--241",
}
```

