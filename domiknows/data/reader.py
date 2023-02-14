import json
import re


class RegrReader:
    def __init__(self, file, type="json"):
        self.file = file
        if type == "json":
            with open(file, 'r') as myfile:
                data = myfile.read()
            # parse file
            self.objects = json.loads(data)
        else:
            self.objects = self.parse_file()

    # you should return the objects list here
    def parse_file(self):
        pass

    def make_object(self, item):
        result = {}
        pattern = re.compile("^get.+val$")
        _list = [method_name for method_name in dir(self)
                 if callable(getattr(self, method_name)) and pattern.match(method_name)]
        for func in _list:
            name = func.replace("get", "", 1)
            k = name.rfind("val")
            name = name[:k]
            result[name] = getattr(self, func)(item)
        return result

    def run(self):
        for item in self.objects:
            yield self.make_object(item)

    def __iter__(self):
        for item in self.objects:
            yield self.make_object(item)

    def __len__(self):
        return len(self.objects)