# WIQA question answering Example

## Requirements

```
pip install networkx
pip install transformers
pip install torch
pip install wget
pip install numpy
pip install gitdb
pip install graphviz
```

## Data Set

The WIQA dataset V1 has 39705 questions containing a perturbation and a possible effect in the context of a paragraph. The dataset is split into 29808 train questions, 6894 dev questions and 3003 test questions.

[link to the Data Set](https://allenai.org/data/wiqa)

## How To Run
```
python WIQA_aug.py --cuda 3 --epoch 100 --lr 2e-7 --samplenum 1000000000 --batch 15 --beta 1.0

python WIQA_PD.py --cuda 3 --epoch 100 --lr 2e-7 --samplenum 1000000000 --batch 15 --beta 1.0 --pd True

python WIQA_aug.py --cuda 3 --epoch 100 --lr 2e-7 --samplenum 1000000000 --batch 15 --beta 1.0 --sample True


```
#### inputs

- cuda: the number of GPU you want to use
- epoch: how many epoch you want to train the program
- lr: learning rate for the AdamW optimiser
- samplenum: number of questions to use for training and testing
- batch: batch size
- beta: primal dual coefficient
- pd: include this to use primal dual traning
