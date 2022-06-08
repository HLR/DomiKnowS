import torch


def tokenized_to_origin_span(text, token_list):
    token_span = []
    pointer = 0
    for token in token_list:
        while True:
            if token[0] == text[pointer]:
                start = pointer
                end = start + len(token) - 1
                pointer = end + 1
                break
            else:
                pointer += 1
        token_span.append([start, end])
    return token_span


def id_lookup(span_SENT, start_char):
    for ind, token_span in enumerate(span_SENT):
        start, end = token_span
        if start <= start_char <= end:
            return ind
    raise ValueError("Not Found Position")


# Padding function
def padding(sent, pos=False, max_sent_len=512):
    if pos == False:
        one_list = [1] * max_sent_len
        one_list[0:len(sent)] = sent[:]
        return torch.tensor(one_list, dtype=torch.long)
    else:
        one_list = ["None"] * max_sent_len
        one_list[0:len(sent)] = sent[:]
        return one_list


def span_SENT_to_DOC(token_span_SENT, sent_start):
    token_span_DOC = []
    for token_span in token_span_SENT:
        start, end = token_span
        token_span_DOC.append([start + sent_start, end + sent_start])
    return token_span_DOC


def sent_id_lookup(dict, start_char, end_char=None):
    if end_char is None:
        end_char = start_char
    for cur_dict in dict['sentences']:
        if start_char >= cur_dict['sent_start_char'] and end_char <= cur_dict['sent_end_char']:
            return cur_dict['sent_id']
    raise ValueError("Not Found Position")


def token_id_lookup(token_span_SENT, start_char, end_char):
    for index, token_span in enumerate(token_span_SENT):
        if start_char >= token_span[0] and end_char <= token_span[1]:
            return index


def check_symmetric(arg1, arg2):
    if arg1 == arg2:
        return False
    # Relation need to be within the same context
    context1 = arg1.getAttribute("context")
    context2 = arg2.getAttribute("context")
    if context1 != context2:
        return False
    # (e1, e2) => (e2, e1)
    eiid1, eiid2 = arg1.getAttribute("eiid1"), arg1.getAttribute("eiid2")
    eiid3, eiid4 = arg2.getAttribute("eiid1"), arg2.getAttribute("eiid2")
    return eiid2 == eiid3 and eiid1 == eiid4


def check_transitive(arg11, arg22, arg33):
    # Relation need to be within the same context
    if arg11 == arg22 or arg22 == arg33 or arg11 == arg33:
        return False
    # Relation need to be within the same context
    context1 = arg11.getAttribute("context")
    context2 = arg22.getAttribute("context")
    context3 = arg33.getAttribute("context")
    if not (context1 == context2 and context2 == context3):
        return False
    # (e1, e2) + (e2, e3) => (e1, e3)
    eiida1, eiida2 = arg11.getAttribute("eiid1"), arg11.getAttribute("eiid2")
    eiidb1, eiidb2 = arg22.getAttribute("eiid1"), arg22.getAttribute("eiid2")
    eiidc1, eiidc2 = arg33.getAttribute("eiid1"), arg33.getAttribute("eiid2")
    return eiida2 == eiidb1 and eiidb2 == eiidc2 and eiida1 == eiidc1
