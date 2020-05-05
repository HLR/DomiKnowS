# Inference-Masked Loss for Deep Structured Output Learning

[Quan Guo](https://github.com/guoquan),
[Hossein Rajaby Faghihi](https://github.com/hfaghihi15),
[Yue Zhang](https://github.com/zhangyuejoslin),
[Andrzej Uszok](https://github.com/auszok),
[Parisa Kordjamshidi](https://github.com/kordjamshidi)


This repository contains the code for the experiments of our research paper "_Inference-Masked Loss for Deep Structured Output Learning_" accepted by IJCAI 2020.

## Abstract

> Structured learning algorithms usually involve an inference phase that selects the best global output variables assignments based on the local scores of all possible assignments.
We extend deep neural networks with structured learning to combine the power of learning representations and leveraging the use of domain knowledge in the form of output constraints during training.
Introducing the non-differentiable inference to gradient-based training is a critical challenge.
Compared to using conventional loss functions that penalize every local error independently, we propose an inference-masked loss that takes into account the effect of inference and does not penalize the local errors that can be corrected by the inference.
We empirically show the inference-masked loss combined with the negative log-likelihood loss improves the performance on different tasks, namely entity relation recognition on CoNLL04 and ACE2005 corpora, and spatial role labeling on CLEF 2017 mSpRL dataset. We show the proposed approach helps to achieve better generalizability, particularly in the low-data regime.

## Cite

If you find this work useful, please also consider cite our paper:

Quan Guo, Hossein Rajaby Faghihi, Yue Zhang, Andrzej Uszok, Parisa Kordjamshidi, Inference-Masked Loss for Deep Structured Output Learning, in _Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI 2020)_, 2020

```bibtex
@inproceedings{guo2020inference,
   Author = {Quan Guo and Hossein Rajaby Faghihi and Yue Zhang and Andrzej Uszok and Parisa Kordjamshidi},
   Title = {Inference-Masked Loss for Deep Structured Output Learning},
   Booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI 2020)},
   Year = {2020}
}
```

## Code

### Requirement

#### Dependencies

* Ubuntu
* python 3.7
* [PyTorch](https://pytorch.org) 1.1.0
* [Gurobi](https://www.gurobi.com) 8.0
* other dependencies specified in [`requirements.txt`](requirements.txt)

Our code is tested on Ubuntu 18.04 and with specific version of software mentioned above. It should work with other versions of Ubuntu, however we did not tested.

Please install above listed dependencies and install other dependencies by
```bash
python -m pip install -r requirements.txt
```

#### Setup

Add project root to python module search path, for example, by adding it to environment variable [`PYTHONPATH`](https://docs.python.org/3.7/using/cmdline.html#envvar-PYTHONPATH).
```bash
cd path/to/project/
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Experiments

> We evaluate the proposed approach with several structured learning tasks: 
Two different entity relation extraction (ER) tasks and spatial role labeling (SpRL) task. We investigate the entity and relation recognition corpora (CoNLL04) and ACE 2005 Corpus (ACE2005) for ER task. The two datasets contain different types of entities and relationships. For SpRL task, CLEF 2017 mSpRL dataset (SpRL2017) is investigated.

#### [`CoNLL04`](examples/conll04/)

> [CoNLL04](https://cogcomp.seas.upenn.edu/page/resource_view/43) is a publicly available corpus for ER.
The task is to recognize four types of entities among tokens in a sentence and classify five types of relations between entities.
This corpus contains `5,516` sentences, `11,182` entities, and `2,048` relations.
The applied hard constraints are between the types of relations and the types of their two entities.

#### [`ACE05`](examples/ACE/) / [`ACE05 (w/ hierarchy)`](examples/ACE_hierarchy/)

> The ACE dataset contains documents with annotations defined for several tasks, including Named Entity Recognition, Relation Extraction, and Event Detection and Recognition. The dataset contains seven types of entities and `45` sub-entity types. We use the same data split used in previous work. The training set contains `10,360` sentences each of which includes at least one entity. The test set contains `2,637` sentences  some of which may not contain any entities. The total number of entities within the sentences of the training set is `47,406`, while the testing set contains `10,675` of them.

#### [`SpRL2017`](examples/SpRL)

> The SpRL task is to identify and classify the spatial arguments of the spatial expression in a sentence. To be specific, we identify spatial roles, including "Trajector", "Spatial Indicator", and "Landmark", and detect their spatial triplet relation. We evaluated with CLEF 2017 mSpRL dataset, which has `600` sentences in the training set and `613` sentences in the testing set. The dataset is more challenging because of the complicated triplet relations and fewer examples compared to other tasks.

### Files

* [`examples/`](examples/): all experiments.
   * [`conll04/`](examples/conll04/): experiment code for "CoNLL04".
   * [`ACE/`](examples/ACE/): experiment code for "ACE2005" without hierarchy.
   * [`ACE_hierarchy/`](examples/ACE_hierarchy/): experiment code for "ACE2005" with hierarchy.
   * [`SpRL/`](examples/SpRL): experiment code for "SpRL2017". 
* [`regr/`](regr/): base classes.
* [`requirements.txt`](requirements.txt): `pip` dependency file.
* [`README.md`](README.md): this readme file.

### Steps

Please follow the steps in [requirement](#requirement) to setup an environment for the experiments.

And then, please refer to [specific example directory](#experiments) and and corresponding section in our paper for steps to run the experiments.
