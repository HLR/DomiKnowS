import codecs
import re

#Constants
# START_TAG = '<BOS>'
# STOP_TAG = '<PAD>'

def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)

def load_examples(path, zeros):
    """
    Load examples. A line must contain at least a word and its tag.
    example are separated by empty lines.
    """
    examples = []
    example = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(example) > 0:
                if 'DOCSTART' not in example[0][0]:
                    examples.append(example)
                example = []
        else:
            word = line.split()
            assert len(word) >= 2
            example.append(word)
    if len(example) > 0:
        if 'DOCSTART' not in example[0][0]:
            examples.append(example)
    return examples


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def word_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000 #UNK tag for unknown words
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word

def char_mapping(sentences):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    return dico, char_to_id, id_to_char

def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    # dico[START_TAG] = -1
    # dico[STOP_TAG] = -2
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag

def lower_case(x,lower=False):
    if lower:
        return x.lower()  
    else:
        return x




# def iob2(tags):
#     """
#     Check that tags have a valid BIO format.
#     Tags in BIO1 format are converted to BIO2.
#     """
#     for i, tag in enumerate(tags):
#         if tag == 'O':
#             continue
#         split = tag.split('-')
#         if len(split) != 2 or split[0] not in ['I', 'B']:
#             return False
#         if split[0] == 'B':
#             continue
#         elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
#             tags[i] = 'B' + tag[1:]
#         elif tags[i - 1][1:] == tag[1:]:
#             continue
#         else:  # conversion IOB1 to IOB2
#             tags[i] = 'B' + tag[1:]
#     return True

# def iob_iobes(tags):
#     """
#     the function is used to convert
#     BIO -> BIOES tagging
#     """
#     new_tags = []
#     for i, tag in enumerate(tags):
#         if tag == 'O':
#             new_tags.append(tag)
#         elif tag.split('-')[0] == 'B':
#             if i + 1 != len(tags) and \
#                tags[i + 1].split('-')[0] == 'I':
#                 new_tags.append(tag)
#             else:
#                 new_tags.append(tag.replace('B-', 'S-'))
#         elif tag.split('-')[0] == 'I':
#             if i + 1 < len(tags) and \
#                     tags[i + 1].split('-')[0] == 'I':
#                 new_tags.append(tag)
#             else:
#                 new_tags.append(tag.replace('I-', 'E-'))
#         else:
#             raise Exception('Invalid IOB format!')
#     return new_tags

# def update_tag_scheme(sentences, tag_scheme):
#     for i, s in enumerate(sentences):
#         tags = [w[-1] for w in s]
#         # Check that tags are given in the BIO format
#         if not iob2(tags):
#             s_str = '\n'.join(' '.join(w) for w in s)
#             raise Exception('Sentences should be given in BIO format! ' +
#                             'Please check sentence %i:\n%s' % (i, s_str))
#         if tag_scheme == 'BIOES':
#             new_tags = iob_iobes(tags)
#             for word, new_tag in zip(s, new_tags):
#                 word[-1] = new_tag
#         else:
#             raise Exception('Wrong tagging scheme!')

