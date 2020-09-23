import abc
from typing import Any, Dict
from itertools import chain
from regr.sensor.sensor import Sensor
from regr.graph.property import Property
from regr.utils import consume

import torch


class SkipSensor(Exception):
    pass


class TorchSensor(Sensor):
    def __init__(self, target=False):
        super().__init__()
        self.target = target
        self._to_args = []
        self._to_kwargs = {}

    def propagate_context(self, data_item, node, force=False):
        if not self.target:  # avoid target to be mixed in forward calculation
            super().propagate_context(data_item, node, force)

    def mask(self, data_item: Dict[str, Any]) -> Any:
        # allow safely skip mask
        raise SkipSensor

    def to(self, *args, **kwargs):
        self._to_args = args
        self._to_kwargs = kwargs


class Key():
    def __init__(self, key):
        self.key = key


class DataSensor(TorchSensor):
    def __init__(self, key, target=False):
        super().__init__(target=target)
        if isinstance(key, Key):
            self.key = key.key
        else:
            self.key = key

    def forward(self, context):
        return context[self.key]


class ReaderSensor(DataSensor):
    def forward(self, data_item):
        return [True] * len(data_item[self.key]), data_item[self.key]


class LabelSensor(DataSensor):
    def __init__(self, key):
        super().__init__(key, target=True)


def _proc_prop(data_item, pre, sensor_fn=None, sensor_filter=None, **_):
    for _, sensor in pre.items():
        if sensor_filter and not sensor_filter(sensor):
            continue
        try:
            if sensor_fn:
                return sensor_fn(sensor, data_item)
            else:
                return sensor(data_item)
        except SkipSensor:
            pass
    else:  # no break
        raise RuntimeError('Not able to find a sensor for {} as prereqiured by {}'.format(
            pre.fullname, self.fullname))

class FunctionalSensor(TorchSensor):
    def __init__(self, *pres, target=False):
        super().__init__(target)
        self.pres = pres

    filters = [
        (Property, _proc_prop),
        (Key, lambda data_item, pre, **_: data_item[pre.key])
    ]
    def get_args(self, data_item, skip_none_prop=False, sensor_fn=None, sensor_filter=None):
        for pre in self.pres:
            for Type, func in self.filters:
                if isinstance(pre, Type):
                    yield func(data_item, pre, skip_none_prop=skip_none_prop, sensor_fn=sensor_fn, sensor_filter=sensor_filter)
                    break
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
        if self.sup is not None and self.sup.sup is not None:
            concept = self.sup.sup
            try:
                mask, *_ = concept['index'](data_item)
                return mask
            except KeyError:
                pass
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


class CartesianCandidateSensor(FunctionalSensor):
    def forward_func(self, *inputs):
        # torch cat is not broadcasting, do repeat manually
        input_iter = iter(inputs)
        output, *output_t = next(input_iter)
        for idx, output_raw in enumerate(output_t):
            if isinstance(output_raw, list):
                output_rawn = []
                for output_raw_sample in output_raw:
                    output_raw_sample = list(map(lambda x: (x,), output_raw_sample))
                    output_rawn.append(output_raw_sample)
                output_t[idx] = output_rawn
        for input, *input_t in input_iter:
            batch, *lengths = output.shape
            output = tensor_index_cartprod(output, input)

            output_tn = []
            for output_raw, input_raw in zip(output_t, input_t):
                if isinstance(output_raw, torch.Tensor):
                    if len(output_raw.shape) == len(lengths)+1:
                        output_raw = tensor_index_cartprod(output_raw, input_raw, func=lambda x, y: torch.stack((x,y),dim=-1))
                    elif len(output_raw.shape) == len(lengths)+2:
                        output_raw = tensor_cartprod(output_raw, input_raw)
                    output_tn.append(output_raw)
                elif isinstance(output_raw, list):
                    output_rawn = []
                    for output_raw_sample, input_raw_sample in zip(output_raw, input_raw):
                        output_rawn.append(nested_list_cartprod(output_raw_sample, input_raw_sample, lengths=lengths))
                    output_tn.append(output_rawn)
            output_t = tuple(output_tn)
        return (output, *output_t)

def tensor_index_cartprod(output, input, func=None):
    dob, *dol = output.shape
    dib, *dil = input.shape
    assert dob == dib
    output = output.view(dob, *dol, *(1,)*len(dil)).repeat(1, *(1,)*len(dol), *dil)
    input = input.view(dib, *(1,)*len(dol), *dil).repeat(1, *dol, *(1,)*len(dil))
    if func is None:
        output = output * input
    else:
        output = func(output, input)
    return output

def tensor_feat_cartprod(output_raw, input_raw):
    dob, *dol, dof = output_raw.shape
    dib, *dil, dif = input_raw.shape
    assert dob == dib
    output_raw = output_raw.view(dob, *dol, *(1,)*len(dil), dof).repeat(1, *(1,)*len(dol), *dil, 1)
    input_raw = input_raw.view(dib, *(1,)*len(dol), *dil, dif).repeat(1, *dol, *(1,)*len(dil), 1)
    output_raw = torch.cat((output_raw, input_raw), dim=-1)
    return output_raw

def nested_list_cartprod(output_raw, input_raw, lengths=None):
    #if lengths:
        #length = lengths[0]
        #assert len(output_raw) == lengths[0]
    if lengths or (isinstance(output_raw, list) and lengths is None):
        for idx, output_item in enumerate(output_raw):
            output_raw[idx] = nested_list_cartprod(output_item, input_raw, lengths=lengths[1:])
        return output_raw
    else:
        return [(*output_raw, input_item) for input_item in input_raw]


from collections import OrderedDict, Counter

class NorminalSensor(FunctionalSensor):
    PAD = '__PAD__'
    UNK = '__UNK__'
    SPECIAL_TOKENS = [PAD, UNK]

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
            padding = torch.full((max_len - len(output),), self.encode(self.PAD), dtype=torch.long).to(*self._to_args, **self._to_kwargs)
            return torch.cat((
                torch.tensor(output).to(*self._to_args, **self._to_kwargs),
                padding))
        encoded_tensor = torch.stack(tuple(map(pad, encoded_outputs)))
        return encoded_tensor, raw_outputs


class SpacyTokenizorSensor(NorminalSensor):
    from spacy.lang.en import English
    nlp = English()

    def forward_func(self, sentences):
        _, sentences = sentences
        tokens = self.nlp.tokenizer.pipe(sentences)
        return list(tokens)

    def forward(self, data_item):
        encoded_tokens, raw_outputs = super().forward(data_item)
        return encoded_tokens != self.encode(self.PAD), encoded_tokens, raw_outputs


def overlap(start1, end1, start2, end2):
    return ((start1 <= start2 and start2 < end1) or
            (start2 <= start1 and start1 < end2))


def match_in_spacy(doc, tokens, idx, token):
    start = 0
    for t in tokens[:idx]:
        start += len(t) + 1
    end = start + len(token)
    doc_iter = iter(doc)
    matched = []
    for tkn in doc_iter:
        if tkn.idx + len(tkn) > start:
            matched.append(tkn)
            break
    for tkn in doc_iter:
        if tkn.idx < end:
            matched.append(tkn)
        else:
            break
    return matched


class LabelAssociateSensor(FunctionalSensor):
    def __init__(self, *pres, target=True):
        super().__init__(*pres, target=target)

    def forward_func(self, encoded_tokens, tokens, labels, key):
        masks, encoded_tokens, raw_outputs = encoded_tokens
        label_tensor = torch.zeros_like(masks).to(*self._to_args, **self._to_kwargs)
        for sample_idx, (mask, encoded_token, raw_output, token, label) in enumerate(zip(masks, encoded_tokens, raw_outputs, tokens, labels)):
            for token_idx, (token_item, label_item) in enumerate(zip(token, label)):
                if label_item == key:
                    raw_i = match_in_spacy(raw_output, token, token_idx, token_item)[0].i
                    label_tensor[sample_idx, raw_i] = 1
        return label_tensor


class LabelRelationAssociateSensor(LabelAssociateSensor):
    def forward_func(self, encoded_tokens, tokens, relations, key):
        masks, encoded_tokens, raw_outputs = encoded_tokens
        label_tensor = torch.zeros_like(masks).to(*self._to_args, **self._to_kwargs)
        for sample_idx, (mask, encoded_token, raw_output, token, relation) in enumerate(zip(masks, encoded_tokens, raw_outputs, tokens, relations)):
            for rkey, (sidx, (stoken, *_)), (didx, (dtoken, *_)) in relation:
                if key != rkey:
                    continue
                doc = raw_output[0][0][0].doc
                si = match_in_spacy(doc, token, sidx, stoken)[0].i
                di = match_in_spacy(doc, token, didx, dtoken)[0].i
                label_tensor[sample_idx, si, di] = 1
        return label_tensor


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
