# PyTorch implementation for NER with CoNLL 2003 using pre-trained BERT

This repository tries to replicate BERT's results on CoNLL 2003 NER task.

With `BERT-BASE-CASED`, the result is as follows on `eval` set:

```
           precision    recall  f1-score   support

      LOC       0.97      0.97      0.97      1837
     MISC       0.89      0.92      0.90       922
      PER       0.97      0.98      0.98      1836
      ORG       0.92      0.94      0.93      1341

micro avg       0.95      0.96      0.95      5936
macro avg       0.95      0.96      0.95      5936
```

On the `test` set:
```
           precision    recall  f1-score   support

      PER       0.96      0.95      0.96      1615
      LOC       0.92      0.93      0.93      1666
     MISC       0.80      0.83      0.82       702
      ORG       0.88      0.91      0.89      1661

micro avg       0.91      0.92      0.91      5644
macro avg       0.91      0.92      0.91      5644

```

To reproduce:
```
 python run.py --batch_size 32 --lr 3e-5 --n_epochs 5 --train
```

Credits:
Google Colab was used to build this - this repo is just the cleaner version. The original notebook is [here](https://colab.research.google.com/drive/1tX6Le-MQoSI6dYDPQIzl7Jl1SOD3706V)

Key ideas to get this working are due to [this github issue/comment](https://github.com/huggingface/transformers/issues/64#issuecomment-443703063).