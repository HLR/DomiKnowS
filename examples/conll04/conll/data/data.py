import pickle
import hashlib
from torch.utils.data import DataLoader


from .reader import Conll04CorpusReader


class Conll04DataLoader(DataLoader):
    def __init__(self, path, reader=None, **kwargs):
        self.path = path
        self.reader = reader or Conll04CorpusReader()
        sentences_list, relations_list = self.reader(path)
        samples = list(zip(sentences_list, relations_list))
        super().__init__(samples, collate_fn=self._collate_fn, **kwargs)
    
    def _collate_fn(self, batch):
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

class SingletonDataLoader(Conll04DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, batch_size=1, **kwargs)

    def _collate_fn(self, batch):
        assert len(batch) == 1
        sentences, relations = zip(*batch)
        # (tokens, pos, label)
        # (relation_type, (src_index, src_token), (dst_index, dst_token))
        tokens, postags, labels = zip(*sentences)
        text = [' '.join(token.replace('/', ' ') for token in tokens[0])]
        data_item = {
            'id': hashlib.sha1(pickle.dumps(tokens[0])).hexdigest(),
            'text': text,
            'tokens': tokens[0],
            'postag': postags[0],
            'label': labels[0],
            'relation': relations[0],
        }
        return data_item