# Walkthrough Example

The followings are the user's steps to using our framework for the Intermediate example.

- **Dataset**
- Knowledge Declaration
- Model Declaration
- Training and Testing
- Inference

## Dataset

### Dataset Introduction

The CoNLL (Conference on Natural Language Learning) dataset, specifically the CoNLL-2003 dataset, is a widely recognized dataset in the field of named entity recognition (NER) and relation extraction. It's primarily designed for detecting entities such as persons, locations, organizations, and miscellaneous entities from English and German news articles. It consists of: Tokenized sentences from news articles and Labels tagging each token with its respective entity type or as being outside of any entity (often represented as "O").

### Task
The primary task for the CoNLL-2003 dataset is to identify and classify entities within a sentence. For example, in the sentence "Barack Obama was born in Hawaii," the model should recognize "Barack Obama" as a PERSON and "Hawaii" as a LOCATION. Furthermore, the dataset can be used for relation extraction, where the objective is not only to identify entities but also to detect relationships between them. For instance, if the sentence was "Barack Obama was born in Hawaii," the potential relation might be a "born-in" relation between "Barack Obama" and "Hawaii."

### Refrence

See [here](https://huggingface.co/datasets/tomaarsen/conll2003) for more information
____
[Goto next section (Knowledge Declaration)](Knowledge%20Declaration.md)



