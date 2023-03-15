
# BeliefBank

## Requirements
install requirements:
```
pip install -r requirements.txt
```

spacy `en_core_web_sm` model:

```
pip install spacy
python -m spacy download en_core_web_sm
```


## How to run

simply write

```
!python -m beliefbank
```
After running the example for the first time the dataset is downloaded from the AllenAI official website. ( If the link is deprecated, check [AllenAI Website](https://allenai.org/data/beliefbank))

## Arguments

+ --cuda: cuda number to train the models on
+ --epoch: number of epochs you want your model to train on
+ --samplenum:sample sizes for low data regime 10,20,40 max 37
+ --simple_model: use the simplet model
+ --pd: whether or not to use primaldual constriant learning
+ --iml: whether or not to use IML constriant learning
+ --sam: whether or not to use sampling learning
+ --batch: batch size for neural network training
+ --beta: primal dual or IML multiplier
+ --lr: learning rate of the adam optimiser

