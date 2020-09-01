
class DummySpan:
    class Mention:
        def __init__(self, start, end):
            self.start = start
            self.end = end
    def __init__(self, start, end):
        self.mentions = []
        self.mentions.append(self.Mention(start, end))

class DummyReader:
    def __iter__(self):
        yield {
        'text': 'John works for IBM.',
        'spans': [DummySpan(0,1), DummySpan(3,4)]
        }
