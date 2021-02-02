from torch.utils.data import DataLoader


from .reader import Conll04CorpusReader


def collate(batch):
    sentences, relations = zip(*batch)
    # (tokens, pos, label)
    # (relation_type, (src_index, src_token), (dst_index, dst_token))
    tokens, postags, labels = zip(*sentences)
    data_item = {
        'sentence': [' '.join(token_list) for token_list in tokens],
        'tokens': list(tokens),
        'postag': list(postags),
        'label': list(labels),
        'relation': list(relations),
    }
    #import pdb; pdb.set_trace()
    return data_item


class NaiveDataLoader(DataLoader):
    def __init__(self, path, reader=None, **kwargs):
        self.path = path
        self.reader = reader or Conll04CorpusReader()
        sentences_list, relations_list = self.reader(path)
        samples = list(zip(sentences_list, relations_list))
        super().__init__(samples, collate_fn=collate, **kwargs)
