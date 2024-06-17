## Setting
ARG-0/ARG-1 only, LSTM w/ highway connection, Viterbi decoding

GBI parameters: `lr=1e-3, reg_weight=1.0, limit_spans = 40, limit_words = 20`

## Regular model predictions:
```
Pre exact match (all spans): 0.64, Pre satisfaction: 0.88, n=1000

Token-level metrics:
              precision    recall  f1-score   support

           0       0.94      0.96      0.95     10768
           1       0.84      0.82      0.83       725
           2       0.86      0.78      0.81      2578

    accuracy                           0.92     14071
   macro avg       0.88      0.85      0.87     14071
weighted avg       0.92      0.92      0.92     14071
```

## Initially unsatisfied + GBI:
```
Post exact match (all spans): 0.50, Post satisfaction: 0.94, n=121

Token-level metrics:
              precision    recall  f1-score   support

           0       0.87      0.95      0.91      1163
           1       0.94      0.81      0.87       182
           2       0.89      0.76      0.82       556

    accuracy                           0.88      1901
   macro avg       0.90      0.84      0.86      1901
weighted avg       0.88      0.88      0.88      1901
```

64 -> 70

## Initially unsatisfied + ILP:
```
Pre exact match (all spans): 0.46, Pre satisfaction: 1.00, n=121

Token-level metrics:
              precision    recall  f1-score   support

           0       0.90      0.84      0.87      1163
           1       0.80      0.88      0.84       182
           2       0.75      0.83      0.79       556

    accuracy                           0.84      1901
   macro avg       0.82      0.85      0.83      1901
weighted avg       0.84      0.84      0.84      1901
```

64 -> ~70