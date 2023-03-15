import abc
from typing import Any, Dict
from itertools import chain
from domiknows.sensor.sensor import Sensor
from domiknows.graph.property import Property
from domiknows.utils import consume

import torch


class SkipSensor(Exception):
    pass


class TorchSensor(Sensor):
    def __init__(self, target=False):
        super().__init__()
        self.target = target

    def propagate_context(self, data_item, node, force=False):
        if not self.target:  # avoid target to be mixed in forward calculation
            super().propagate_context(data_item, node, force)

    def mask(self, data_item: Dict[str, Any]) -> Any:
        # allow safely skip mask
        raise SkipSensor


class DataSensor(TorchSensor):
    def __init__(self, key, target=False):
        super().__init__(target=target)
        self.key = key

    def forward(self, data_item):
        return data_item[self.key]


class LabelSensor(DataSensor):
    def __init__(self, key):
        super().__init__(key, target=True)


class FunctionalSensor(TorchSensor):
    def __init__(self, *pres, target=False):
        super().__init__(target)
        self.pres = pres

    def get_args(self, data_item, skip_none_prop=False, sensor_fn=None, sensor_filter=None):
        for pre in self.pres:
            if isinstance(pre, Property):
                for _, sensor in pre.items():
                    if sensor_filter and not sensor_filter(sensor):
                        continue
                    try:
                        if sensor_fn:
                            yield sensor_fn(sensor, data_item)
                        else:
                            yield sensor(data_item)
                        break
                    except SkipSensor:
                        pass
                else:  # no break
                    raise RuntimeError('Not able to find a sensor for {} as prereqiured by {}'.format(
                        pre.fullname, self.fullname))
            else:
                if not skip_none_prop:
                    yield pre

    @abc.abstractmethod
    def forward_func(self, *args, **kwargs):
        raise NotImplementedError

    def forward(
        self,
        data_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        args = self.get_args(data_item, sensor_filter=lambda s: not s.target)
        return self.forward_func(*args)

    def mask(self, data_item):
        masks = list(self.get_args(data_item, sensor_fn=lambda s, c: s.mask(c)))
        masks_num = len(masks)
        mask = masks[0]
        mask = mask.float()
        for i in range(1, masks_num):
            for j in range(i, masks_num):
                masks[j].unsqueeze_(-2)
            masks[i] = masks[i].float()
            mask = mask.unsqueeze_(-1).matmul(masks[i])
        return mask


class CartesianSensor(FunctionalSensor):
    def forward_func(self, *inputs):
        # torch cat is not broadcasting, do repeat manually
        input_iter = iter(inputs)
        output = next(input_iter)
        for input in input_iter:
            dob, *dol, dof = output.shape
            dib, *dil, dif = input.shape
            assert dob == dib
            output = output.view(dob, *dol, *(1,)*len(dil), dof).repeat(1, *(1,)*len(dol), *dil, 1)
            input = input.view(dib, *(1,)*len(dol), *dil, dif).repeat(1, *dol, *(1,)*len(dil), 1)
            output = torch.cat((output, input), dim=-1)
        return output


from collections import OrderedDict, Counter

class NorminalSensor(FunctionalSensor):
    UNK = '__UNK__'
    SPECIAL_TOKENS = [UNK]

    def __init__(self, *pres, vocab_counter=None, least_count=0, special_tokens=[], target=False):
        super().__init__(*pres, target=target)
        self.least_count = least_count
        self.special_tokens = self.SPECIAL_TOKENS.copy()
        self.special_tokens.extend(special_tokens)
        self.fixed = False
        self._vocab = None
        self.set_vocab(vocab_counter)

    def set_vocab(self, vocab_counter=None):
        self.vocab_counter = vocab_counter or Counter()
        self.update_vocab(fixed=True)

    def update_counter(self, outputs):
        comsum(self.vocab_counter.update, chain(*outputs))

    def update_vocab(self, fixed=None):
        if self.fixed:
            raise RuntimeError('Vocab of {} has been fixed!'.format(self))
        vocab_list = self.special_tokens.copy()
        vocab_count = list(filter(lambda kv: kv[1] > self.least_count, self.vocab_counter))
        if vocab_count:
            vocab_to_add, _ = zip(*vocab_count)
        else:
            vocab_to_add = []
        vocab_list.extend(vocab_to_add)
        self._vocab = OrderedDict((token, idx) for idx, token in enumerate(vocab_list))
        if fixed is not None:
            self.fixed = fixed

    def encode(self, output):
        return self._vocab.get(output, self._vocab[self.UNK])

    def forward(self, data_item):
        raw_outputs = super().forward(data_item)
        if not self.fixed:
            self.update_counter(raw_outputs)
        encoded_outputs = [list(map(self.encode, raw_output)) for raw_output in raw_outputs]
        max_len = max(map(len, encoded_outputs))
        def pad(output):
            return torch.cat((
                torch.tensor(output, dtype=torch.long), 
                torch.zeros(max_len - len(output), dtype=torch.long)
                ))
        encoded_tensor = torch.stack(tuple(map(pad, encoded_outputs)))
        return encoded_tensor, raw_outputs

class SpacyTokenizorSensor(NorminalSensor):
    from spacy.lang.en import English
    nlp = English()

    def forward_func(self, sentences):
        tokens = self.nlp.tokenizer.pipe(sentences)
        return list(tokens)


TRANSFORMER_MODEL = 'bert-base-uncased'

class BertTokenizorSensor(FunctionalSensor):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_MODEL)

    def forward_func(self, sentences):
        tokens = self.tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            return_attention_mask=True,
            #return_offsets_mapping=True,
        )
        tokens['tokens'] = self.tokenizer.convert_ids_to_tokens(tokens['input_ids'], skip_special_tokens=True)
        return tokens

class BertEmbeddingSensor(FunctionalSensor):
    from transformers import BertModel
    model = BertModel.from_pretrained(TRANSFORMER_MODEL)

    # {
    #     tokens: list[List[str]], sting based tokens
    #     input_ids: list[List[int]],
    #     -token_type_ids: list[List[int]] if return_token_type_ids is True (default)
    #     attention_mask: list[List[int]] if return_attention_mask is True (default)
    #     -overflowing_tokens: list[List[int]] if a ``max_length`` is specified and return_overflowing_tokens is True
    #     -num_truncated_tokens: List[int] if a ``max_length`` is specified and return_overflowing_tokens is True
    #     -special_tokens_mask: list[List[int]] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
    # }
    def forward_func(self, tokens):
        outputs = self.model(tokens['input_ids'])
        return outputs
