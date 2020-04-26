# Inference-Masked Loss for Deep Structured Output Learning

[Quan Guo](https://github.com/guoquan),
[Hossein Rajaby Faghihi](https://github.com/hfaghihi15),
[Yue Zhang](https://github.com/zhangyuejoslin),
[Andrzej Uszok](https://github.com/auszok),
[Parisa Kordjamshidi](https://github.com/kordjamshidi)


This repository contains the code for the experiments of our research paper _Inference-Masked Loss for Deep Structured Output Learning_ accepted by IJCAI 2020.

## Abstract

> Structured learning algorithms usually involve an inference phase that selects the best global output variables assignments based on the local scores of all possible assignments.
We extend deep neural networks with structured learning to combine the power of learning representations and leveraging the use of domain knowledge in the form of output constraints during training. This brings in several challenges: imposing the structure of output and domain knowledge, introducing the non-differentiable inference to gradient-based training, and the demand for parameters and examples, which is even more significant for the case of complex outputs.
Compared to using conventional loss functions that penalize every local error independently, we use an inference-masked loss that takes into account the effect of inference and does not penalize the local errors that can be corrected by the inference.
We empirically show the inference-masked loss combined with the negative log-likelihood loss improves the performance on different tasks, namely entity relation recognition on CoNLL04 and ACE2005 corpora, and spatial role labeling on CLEF 2017 mSpRL dataset. We show the proposed approach helps to achieve better generalizability, particularly in the low-data regime.

## Cite

If you find this repository useful, please also consider cite our paper:

Quan Guo, Hossein Rajaby Faghihi, Yue Zhang, Andrzej Uszok, Parisa Kordjamshidi, Inference-Masked Loss for Deep Structured Output Learning, in _Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI 2020)_, 2020

```bibtex
@inproceedings{guo2020inferenced,
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
* other dependencies specified in `requirements.txt`

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

### Files

* [`examples/`](examples/): all experiments.
   * [`conll04/`](examples/conll04/): experiment code for "CoNLL04".
   * [`ACE/`](examples/ACE/): experiment code for "ACE2005" without hierarchy.
   * [`ACE_hierarchy/`](examples/ACE_hierarchy/): experiment code for "ACE2005" with hierarchy.
   * [`SpRL/`](examples/SpRL): experiment code for "CLEF 2017 mSpRL". 
* [`regr/`](regr/): base classes.
* [`requirements.txt`](requirements.txt): `pip` dependency file.
* [`README.md`](README.md): this readme file.

### Steps

Please refer to specific example directory and and corresponding section in our paper for steps to run the experiments.
