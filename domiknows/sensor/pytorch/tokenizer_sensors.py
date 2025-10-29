from sensors import FunctionalSensor

class SpacyTokenizorSensor(FunctionalSensor):
    """Tokenize text using spaCy's English tokenizer.

    Returns:
        A list of spaCy ``Doc`` objects, one per input string.
    """

    from spacy.lang.en import English
    nlp = English()

    def forward(self, sentences):
        """Tokenize each input string and return spaCy ``Doc`` objects.

        Args:
            sentences (Iterable[str]): Collection of strings to tokenize.

        Returns:
            list: List of spaCy ``Doc`` objects.
        """
        tokens = self.nlp.tokenizer.pipe(sentences)
        return list(tokens)


class BertTokenizorSensor(FunctionalSensor):
    """Tokenize text with a Hugging Face BERT tokenizer.

    Attributes:
        TRANSFORMER_MODEL (str): Name of the pretrained tokenizer.
    """

    TRANSFORMER_MODEL = 'bert-base-uncased'

    @property
    def tokenizer(self):
        """load and cache the underlying ``BertTokenizer`` instance."""
        if self._tokenizer is None:
            from transformers import BertTokenizer
            self._tokenizer = BertTokenizer.from_pretrained(self.TRANSFORMER_MODEL)
        return self._tokenizer

    def forward(self, sentences):
        """Batch-tokenize input text and return model-ready tensors.

        Args:
            sentences (Iterable[str]): Collection of input strings.

        Returns:
            dict: A dictionary including at least ``input_ids`` and
                ``attention_mask`` as PyTorch tensors. Additionally, a
                ``tokens`` key is added with the tokenizer's string tokens
                derived from ``input_ids`` (special tokens skipped).
        """
        tokens = self.tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            return_attention_mask=True,
        )
        tokens['tokens'] = self.tokenizer.convert_ids_to_tokens(tokens['input_ids'], skip_special_tokens=True)
        return tokens
