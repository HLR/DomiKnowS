# Natural Language Inference

The task is question answering to determine whether hypothesis inference the information 
from premise consisting of three answer, entailment(yes), contradiction(no), and neutral(unconditioned)

### Train the model with SNLI dataset and test with original SNLI test set with pytorch model

```
python main_pytorch.py --lr 1e-5 --training_sample 100000 --epoch 5 --batch_size 16
```
- cuda: the number of GPU you want to use
- epoch: number of epoch to train program
- lr: learning rate for the AdamW optimizer
- training_sample: number of training sample will be used
- testing_sample: number of testing sample will be used
- batch_size: batch size of sample

