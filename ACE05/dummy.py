from domiknows.graph import Concept


class DummyObject:
    pass

class DummySpan(DummyObject):
    class Charseq(DummyObject):
        def __init__(self, text, start, end):
            self.start = start
            self.end = end
            self.text = text

    class Mention(DummyObject):
        def __init__(self, text, start, end):
            self.head = DummySpan.Charseq(text, start, end)

    def __init__(self, text, *start_end):
        self.mentions = {}
        for (start, end) in start_end:
            self.mentions[f'{start}_{end}'] = self.Mention(''.join(text[start:end]), start, end)


class DummyReader:
    def __iter__(self):
        text = 'John works for IBM .'
        yield {
            'text': text,
            'spans': {
                '1': DummySpan(text, (0,4)),
                '2': DummySpan(text, (5,14)),
                '3': DummySpan(text, (15,18))}
        }


class DummyDictReader(DummyReader):
    def _make_dict(self, obj):
        if isinstance(obj, DummyObject):
            obj = obj.__dict__
        if isinstance(obj, dict) and not isinstance(obj, Concept):
            obj = {k: self._make_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            obj = [self._make_dict(v) for v in obj]
        return obj

    def __iter__(self):
        yield from map(self._make_dict, super().__iter__())
