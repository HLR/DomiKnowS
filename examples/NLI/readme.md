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



### Result  
#### Small Train Set, 10%
| Training sample size | epoch | learning rate |       model       | Accuracy (%) | Accuracy on Augmented test only (%) |
|----------------------|:-----:|:-------------:|:-----------------:|:------------:|------------------------------------:|
| 50000                |   5   |     1e-5      |        POI        |    84.15     |                               68.65 |
| 50000                |   5   |     1e-5      |        PD         |    84.52     |                               71.75 |
| 50000                |   5   |     1e-5      |    Sample Loss    |    84.45     |                               69.45 |
| 50000                |   5   |     1e-5      |        ILP        |    86.47     |                               76.25 |
| 50000                |   5   |     1e-5      |     PD + ILP      |    86.28     |                               78.60 |
| 50000                |   5   |     1e-5      | Sample Loss + ILP |    86.55     |                               75.55 |

#### Result Testing Augment
#### Large Train Set
| Training sample size | epoch | learning rate |       model        | Accuracy (%) | Accuracy on Augmented test only (%) |
|----------------------|:-----:|:-------------:|:------------------:|:------------:|------------------------------------:|
| Paper                |   ?   |     1e-5      |         ?          |    87.25     |                               60.78 |
| Paper                |   ?   |     1e-5      | Applied Regularize |    87.55     |                               73.32 |
| 550146               |   5   |     1e-5      |        POI         |    87.46     |                               74.00 |
| 550146               |   5   |     1e-5      |         PD         |    87.45     |                               74.25 |
| 550146               |   5   |     1e-5      |    Sample Loss     |      ?       |                               74.55 |
| 550146               |   5   |     1e-5      |        ILP         |    89.88     |                               82.90 |
| 550146               |   5   |     1e-5      |      PD + ILP      |    89.64     |                               82.75 |
| 550146               |   5   |     1e-5      | Sample Loss + ILP  |      ?       |                               75.55 |