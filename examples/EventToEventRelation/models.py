from transformers import RobertaTokenizer
from utils import *


class Roberta_Tokenizer:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', unk_token='<unk>')

    def __call__(self, content, token_span_SENT=None):
        encoded = self.tokenizer.encode(content)
        roberta_subword_to_ID = encoded
        roberta_subwords = []
        roberta_subwords_no_space = []
        for index, i in enumerate(encoded):
            r_token = self.tokenizer.decode([i])
            roberta_subwords.append(r_token)
            if r_token[0] == " ":
                roberta_subwords_no_space.append(r_token[1:])
            else:
                roberta_subwords_no_space.append(r_token)

        roberta_subword_span = tokenized_to_origin_span(content, roberta_subwords_no_space[1:-1])  # w/o <s> and </s>
        roberta_subword_map = []
        if token_span_SENT is not None:
            roberta_subword_map.append(-1)  # "<s>"
            for subword in roberta_subword_span:
                roberta_subword_map.append(token_id_lookup(token_span_SENT, subword[0], subword[1]))
            roberta_subword_map.append(-1)  # "</s>"
            return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, roberta_subword_map
        else:
            return roberta_subword_to_ID, roberta_subwords, roberta_subword_span, -1
