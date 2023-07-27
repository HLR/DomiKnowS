# DomiKnowS: Declarative Knowledge Integration with Deep Neural Models

DomiKnowS is a Python library that facilitates the integration of domain knowledge in deep learning architectures. With DomiKnowS, you can express the structure of your data symbolically via graph declarations and seamlessly add logical constraints over outputs or latent variables to your deep models. This allows you to define domain knowledge explicitly, improving the explainability, performance, and generalizability of your models, especially in low-data regimes. 

While several approaches for integrating symbolic and sub-symbolic models have been introduced, there is no generic library that facilitates programming for such integration with various underlying algorithms. DomiKnowS aims to simplify the programming for knowledge integration in both training and inference phases, while separating the knowledge representation from learning algorithms.

## Usage

Using DomiKnowS involves the following steps:

1. **Graph Declaration**: Define the structure of the data symbolically via graph declarations using DomiKnowS. This step allows you to express domain knowledge in the form of logical constraints over outputs or latent variables.

2. **Model Declaration**: Declare the deep learning model architecture using DomiKnowS. This step involves specifying the neural network layers, activation functions, and other relevant configurations.

3. **Program Initialization**: Initialize the DomiKnowS program by specifying the learning problem you want to solve. This step involves setting up the optimization objective, constraints, concepts, and other necessary configurations.

4. **Program Composition and Execution**: Compose and execute the DomiKnowS program to train the deep learning model with integrated domain knowledge. This step involves solving the optimization problem and updating the model's parameters iteratively.

For detailed documentation on how to use DomiKnowS, please refer to the [official documentation](link_to_official_documentation).


## Contents

- [Getting Started](https://github.com/HLR/DomiKnowS/blob/Doc/New/GettingStarted.md): Provides detailed instructions on how to get started with DomiKnowS, including installation, setting up the environment, and basic usage.
- [Documentation](https://github.com/HLR/DomiKnowS/tree/Doc/apis): Provides comprehensive documentation on the DomiKnowS, including classes, methods, and their usage.
- [Example Tasks](https://github.com/HLR/DomiKnowS/blob/Doc/Getting%20Started.md): Contains examples that demonstrate the usage of DomiKnowS for various tasks, such as image classification, sequence modeling, and reinforcement learning. ( For more example see [Examples Branch](https://github.com/HLR/DomiKnowS/tree/Tasks) )
- [Contributing](https://github.com/HLR/DomiKnowS/blob/Doc/IssueReport.md): Explains how you can contribute to the development of DomiKnowS, including reporting issues, suggesting enhancements, and submitting pull requests.
- [License](https://github.com/HLR/DomiKnowS/blob/Doc/Licence.md): Contains information about the license of DomiKnowS and its terms of use.
- [DomiKnowS Website](https://hlr.github.io/domiknows-nlp/): Contains documentation, example links, and an introductory video to DomiKnowS

## Quick Start

To start using DomiKnowS, follow these steps:

1. Install DomiKnowS using `pip install DomiKnowS`.
2. Install Gurobi following the instructions [here](https://github.com/HLR/DomiKnowS/blob/develop/GurobiREADME.md).
3. Refer to the [Getting Started](https://github.com/HLR/DomiKnowS/blob/Doc/New/GettingStarted.md) documentation for detailed instructions on how to define graph declarations, model declarations, initialize programs, and compose and execute programs using DomiKnowS.

## Acknowledgements

DomiKnowS is developed and maintained by [HLR](https://hlr.github.io/). We would like to acknowledge the contributions of the open-source community and express our gratitude to the developers of Gurobi for their excellent optimization solver.
TODO add paper link

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

