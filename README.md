# DomiKnowS: Declarative Knowledge Integration with Deep Neural Models

DomiKnowS is a Python library that facilitates the integration of domain knowledge in deep learning architectures. With DomiKnowS, you can express the structure of your data symbolically via graph declarations and seamlessly add logical constraints over outputs or latent variables to your deep models. This allows you to define domain knowledge explicitly, improving your models' explainability, performance, and generalizability, especially in low-data regimes. 

While several approaches for integrating symbolic and sub-symbolic models have been introduced, no generic library facilitates programming for such integration with various underlying algorithms. DomiKnowS aims to simplify the programming for knowledge integration in training and inference phases while separating the knowledge representation from learning algorithms.


- [Getting Started](https://github.com/HLR/DomiKnowS/blob/Doc/User%20API/Getting%20Started.md): Provides detailed instructions on how to get started with DomiKnowS, including installation, setting up the environment, and basic usage.
- [Documentation](https://github.com/HLR/DomiKnowS/tree/Doc/): Provides comprehensive documentation on the DomiKnowS, including classes, methods, and their usage.
  - Contribute to Domiknows: Report Issues [see here](https://github.com/HLR/DomiKnowS/blob/Doc/Issue%20Report.md), share your suggestions [see here](https://github.com/HLR/DomiKnowS/blob/Doc/Suggestions.md) and create a pull request [see here](https://github.com/HLR/DomiKnowS/blob/Doc/Pull%20Request.md).
- [Walkthrough Example](https://github.com/HLR/DomiKnowS/blob/Doc/User%20API/Walkthrough%20Example.md): Contains examples that demonstrate the usage of DomiKnowS for various tasks, such as image classification, sequence modeling, and reinforcement learning. ( For more examples see [Examples Branch](https://github.com/HLR/DomiKnowS/tree/Tasks) )
- [FAQ](https://github.com/HLR/DomiKnowS/blob/Doc/FAQ.md): Read our FAQ file if you have any questions.
- [License](https://github.com/HLR/DomiKnowS/blob/Doc/Licence.md): Contains information about the license of DomiKnowS and its terms of use.
- [DomiKnowS Website](https://hlr.github.io/domiknows-nlp/): Contains documentation, example links, and an introductory video to DomiKnowS

## Contribute to DomiKnowS:

- Report Issues: Encounter a problem? Let us know by [submitting an issue](https://github.com/HLR/DomiKnowS/blob/Doc/Issue%20Report.md).
- Suggest Enhancements: Have an idea to make DomiKnowS better? [Share your suggestion](https://github.com/HLR/DomiKnowS/blob/Doc/Suggestions.md).
- Submit Pull Requests: Ready to contribute code or features directly? [Create a pull request](https://github.com/HLR/DomiKnowS/blob/Doc/Pull%20Request.md).

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

