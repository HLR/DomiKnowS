This task involves imposing constraints on generated tokens, loosely based off of [Yao et al., 2023](https://arxiv.org/abs/2307.08689).

For example, a constraint could be: if the generated sentence contains the token " The", then there must be at least three tokens generated total. We specify this as:
```python
ifL(
    atLeastAL(get_token_concept(' The'), 1),
    atMostAL(token, 3)
)
```

For efficiency, this example currently uses models trained on the [TinyStories dataset](https://arxiv.org/pdf/2305.07759), which have a limited vocabulary. We also restrict the vocabulary to the top-k most used tokens in the corpus.

Run `python build_vocab.py` to find and output the most used tokens in the corpus.

Then, run `python program.py --vocab_file vocab_val.pkl` where the `vocab_file` argument specifies the path to the output from `build_vocab.py`.
