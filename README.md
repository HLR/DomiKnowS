# DomiKnowS: Declarative Knowledge Integration with Deep Neural Models

DomiKnowS is a Python library that facilitates the integration of domain knowledge in deep learning architectures. With DomiKnowS, you can express the structure of your data symbolically via graph declarations and seamlessly add logical constraints over outputs or latent variables to your deep models. This allows you to define domain knowledge explicitly, improving your models' explainability, performance, and generalizability, especially in low-data regimes. 

While several approaches for integrating symbolic and sub-symbolic models have been introduced, no generic library facilitates programming for such integration with various underlying algorithms. DomiKnowS aims to simplify the programming for knowledge integration in training and inference phases while separating the knowledge representation from learning algorithms.


## Contents

- [Getting Started](Getting%20Started.md): Provides detailed instructions on how to get started with DomiKnowS, including installation, setting up the environment, and basic usage.
- [Main Components](Main%20Components/): Provides comprehensive documentation on the DomiKnowS, including classes, methods, and their usage.
  - [Knowledge Declaration (Graph)](Main%20Components/Knowledge%20Declaration%20%28Graph%29.md)
  - [Model Declaration (Sensor)](Main%20Components/Model%20Declaration%20%28Sensor%29.md)
  - [Query and Access (Datanode)](Main%20Components/Query%20and%20Access%20%28Datanode%29.md)
  - [Workflow (Training)](Main%20Components/Workflow%20%28Training%29.md)
  - [Inference (ILP)](Main%20Components/Inference%20%28ILP%29.md)
- [Walkthrough Example](Walkthrough%20Examples/): Contains examples that demonstrate the usage of DomiKnowS for various tasks, such as image classification, sequence modeling, and reinforcement learning. ( For more examples see [Examples Branch](https://github.com/HLR/DomiKnowS/tree/Tasks) )
- [Tutorial Examples](Tutorial%20Examples/): Simple and diverse examples are outlined with detailed explanations in a Jupyter google colab file to run.
- [License](LICENSE.md): Contains information about the license of DomiKnowS and its terms of use.
- [DomiKnowS Website](https://hlr.github.io/domiknows-nlp/): Contains documentation, example links, and an introductory video to DomiKnowS

## Contribute to DomiKnowS:

- Report Issues: Encounter a problem? Let us know by [submitting an issue](Issue%20Report.md).
- Suggest Enhancements: Have an idea to make DomiKnowS better? [Share your suggestion](Suggestions.md).
- Submit Pull Requests: Ready to contribute code or features directly? [Create a pull request](Pull%20Request.md).

