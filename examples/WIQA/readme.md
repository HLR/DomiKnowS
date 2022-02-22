# WIQA question answering Example

<!-- ## Requirements

```
pip install networkx
pip install transformers
pip install torch
pip install wget
pip install numpy
pip install gitdb
pip install graphviz
``` -->

## Create the same excutable conda environment (Chen writing):

I export the conda environment and save in the conda_env_domi.yaml. All you need to do is run the follow command lines:
```
conda env create -f conda_env_domi.yaml
source activate domi
```

In this environment, we use Python 3.9 and torch + CUDA 11.

I once meet the issue that I cannot run the wiqa code. Then I found the reason is that I use CUDA 10...
Please make sure you install the torch library with CUDA 11 version. Here is the pip command: 

```
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Data Set (Darius writing)

The WIQA dataset V1 has 39705 questions containing a perturbation and a possible effect in the context of a paragraph. The dataset is split into 29808 train questions, 6894 dev questions and 3003 test questions.

[link to the Data Set](https://allenai.org/data/wiqa)

## Run the code (Chen writing)

If you want to run the wiqa neural model + cross entrogy:
```
python WIQA_aug.py --cuda 0 --epoch 20 --lr 2e-7 --samplenum 1000000000 --batch 8 --beta 1.0
```

If you want to run the wiqa neural model + primal dual:
```
python WIQA_aug.py --cuda 0 --epoch 20 --lr 2e-7 --samplenum 1000000000 --batch 8 --beta 1.0 --pd True
```


If you want to run the wiqa neural model + semantic loss (sampling loss):
```
python WIQA_aug.py --cuda 0 --epoch 20 --lr 2e-7 --samplenum 1000000000 --batch 8 --beta 1.0 --semantic_loss True
```

P.S.: Dont worry about the setting 'samplenum' because it is the number of questions to use for training and testing, but not relevant to the neural model itself.

P.S.: If you want to run wiqa model on cpu environment, please add the ''--cpu True'' in the command line.


#### inputs (Darius writing):

- cuda: the number of GPU you want to use
- epoch: how many epoch you want to train the program
- lr: learning rate for the AdamW optimiser
- samplenum: number of questions to use for training and testing
- batch: batch size
- beta: primal dual coefficient
- pd: include this to use primal dual traning


## The bugs (Chen writing):

>- GPU out of memory on sampling loss: I have store the error log in the wiqa_sampling_loss.out
>- GPU out of memory on cross entrogy loss: I have store the error log in the wiqa_cross_entrogy.out

I think one possible reason is related to the Pytorch Variable. I meet this error: 
```
File "/home/zhengchen/anaconda/anaconda3/envs/domi/lib/python3.9/site-packages/torch/autograd/__init__.py", line 154, in backward
    Variable._execution_engine.run_backward(
RuntimeError: CUDA out of memory.:
```

New bug: After I git merge the develop_newlc branch to my branch, I meet the error on the cross entrogy. I save the log file to ''wiqa_run_cross_entropy.out''.


## Solutions (Chen writing):

I try to use bitsandbyte library.
```
pip install bitsandbytes-cuda111
```
