# Natural Language Inference

The task is question answering to determine whether hypothesis inference the information 
from premise consisting of three answer, entailment(yes), contradiction(no), and neutral(unconditioned)

### Train the model with SNLI dataset and test with both original SNLI test set and augmented dataset

```
python main_group_sentences.py --lr 1e-5 --training_sample 100000 --epoch 5 --batch_size 16 --sym_relation True
```
- cuda: the number of GPU you want to use
- epoch: number of epoch to train program
- lr: learning rate for the AdamW optimizer
- training_sample: number of training sample will be used
- testing_sample: number of testing sample will be used
- batch_size: batch size of sample
- sym_relation: using symmetric relation or not
- tran_relation: using transitive relation or not
- pmd: whether to use primal dual model(PD)
- sampleloss: whether to use sampling loss model
- beta: beta value to use in PD model
- sampling_size: sampling size to use in sampling loss model
- model_name: model name to save after training

