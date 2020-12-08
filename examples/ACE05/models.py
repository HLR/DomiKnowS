from itertools import chain

import torch
from transformers import BertTokenizer, BertTokenizerFast, BertModel

from ace05.graph import token


def cartesian_concat(*inputs):
    # torch cat is not broadcasting, do repeat manually
    input_iter = iter(inputs)
    output = next(input_iter)
    for input in input_iter:
        *dol, dof = output.shape
        *dil, dif = input.shape
        output = output.view(*dol, *(1,)*len(dil), dof).repeat(*(1,)*len(dol), *dil, 1)
        input = input.view(*(1,)*len(dol), *dil, dif).repeat(*dol, *(1,)*len(dil), 1)
        output = torch.cat((output, input), dim=-1)
    return output


TRANSFORMER_MODEL = 'bert-base-uncased'


class Tokenizer():
    def __init__(self) -> None:
        self.tokenizer = BertTokenizerFast.from_pretrained(TRANSFORMER_MODEL)

    def __call__(self, text):
        if isinstance(text, str):
            text = [text, text[:10]]
        tokens = self.tokenizer(text, padding=True, return_tensors='pt', return_offsets_mapping=True)

        ids = tokens['input_ids']
        mask = tokens['attention_mask']
        offset = tokens['offset_mapping']

        idx = mask.nonzero()[:, 0].unsqueeze(-1)
        mapping = torch.zeros(idx.shape[0], idx.max()+1)
        mapping.scatter_(1, idx, 1)

        mask = mask.bool()
        ids = ids.masked_select(mask)
        offset = torch.stack((offset[:,:,0].masked_select(mask), offset[:,:,1].masked_select(mask)), dim=-1)

        return mapping, ids, offset 


# emb_model = BertModel.from_pretrained(TRANSFORMER_MODEL)
# to freeze BERT, uncomment the following
# for param in emb_model.base_model.parameters():
#     param.requires_grad = False
class BERT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = BertModel.from_pretrained(TRANSFORMER_MODEL)
        # to freeze BERT, uncomment the following
        # for param in emb_model.base_model.parameters():
        #     param.requires_grad = False

    def forward(self, input):
        input = input.unsqueeze(0)
        out, *_ = self.module(input)
        assert out.shape[0] == 1
        out = out.squeeze(0)
        return out


class SpanClassifier(torch.nn.Module):
    def __init__(self, token_emb_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(token_emb_dim*2, 1024)
        self.fc2 = torch.nn.Linear(1024, 2)

    def forward(self, emb):
        # NxNxM
        # emb = cartesian_concat(token_emb, token_emb)
        emb1 = self.fc1(emb)
        emb1 = torch.nn.functional.relu(emb1)
        emb2 = self.fc2(emb1)
        return emb2


def find_is_a(graph, base_concept):
    for name, concept in graph.concepts.items():
        if base_concept in map(lambda rel: rel.dst, concept.is_a()):
            yield concept


def find_event_arg(events_graph, event_type, role_arg_type=None, arg_type=None):
    from regr.graph.logicalConstrain import LogicalConstrain, ifL, orL, andL

    for _, constraint in events_graph.logicalConstrains.items():
        if not isinstance(constraint, ifL):
            continue
        role_arg, xy, implication = constraint.e
        if role_arg_type is not None and not role_arg.relate_to(role_arg_type):
            continue
        event_type_, x, *arg_implication = implication.e
        if event_type is not event_type_:
            continue
        if arg_type is None:
            yield role_arg
            continue
        if len(arg_implication) == 1:
            *concepts, y = arg_implication[0].e
        else:
            concept, y = arg_implication
            concepts = (concept,)
        if arg_type in concepts:
            yield role_arg
            continue


def token_to_span_label(span_label):
    # span_label: NxNx2
    spans = []
    for i, row in enumerate(span_label):
        for j, val in enumerate(row):
            if i > j or j - i > 20:
                continue
            if i == j or val[1] > val[0]:
                spans.append((i, j))
    if len(spans) > 1:
        spans[-1] = spans[-1] + (0,)  # to avoid being converted to tensor
    return [spans]
    # return torch.tensor([[0,1,1,0,0,0,0], [0,0,0,1,1,1,0]], device=span_label.device)


def token_to_span_candidate_emb(token_emb):
    return cartesian_concat(token_emb, token_emb)


def span_to_pair_emb(span_emb):
    return cartesian_concat(span_emb, span_emb)


def makeSpanPairs(current_spans, span, span_anno):
    start = span.getChildDataNodes(conceptName=token)[0].getAttribute('offset')[0]
    end = span.getChildDataNodes(conceptName=token)[-1].getAttribute('offset')[1]
    start_anno = span_anno.getAttribute('start')
    end_anno = span_anno.getAttribute('end')
    # exact match
    if start == start_anno and end == end_anno:
    # overlap
    # if (start < start_anno and start_anno < end) or (start < end_anno and end_anno < end):
        return True
    else:
        return False


def makeSpanAnchorPairs(current_spans, span, anchor_anno):
    start = span.getChildDataNodes(conceptName=token)[0].getAttribute('offset')[0]
    end = span.getChildDataNodes(conceptName=token)[-1].getAttribute('offset')[1]
    start_anno = anchor_anno.getAttribute('start')
    end_anno = anchor_anno.getAttribute('end')
    # exact match
    if start == start_anno and end == end_anno:
    # overlap
    # if (start < start_anno and start_anno < end) or (start < end_anno and end_anno < end):
        return True
    else:
        return False
