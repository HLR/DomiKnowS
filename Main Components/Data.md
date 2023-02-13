
# Reader

Users will have to write their own reader to read source data.
To be flexible, the framework does not require user to implement a specific reader class.
Instead, the user will need to provide an `Iterable` object for the input of the program and yield an sample, or a batch of samples, in a `dict`.
The reader will be provided to the program for specific workflow like `train()`, `test()` or `eval()`.
Sensors will also be invoked with a `dict` retrieved from the reader each time (detailed later).

For example, a `list` of `dict` is a simplest input reader.
Also, `torch.utils.data.DataLoader` instance is a good choice when working with PyTorch.
The framework also has a simple reader for JSON format input file.

There is also a default Reader class implemented in the framework.
```python
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
```
if you are loading a json file you do not have to rewrite the `parse_file` function for your customized reader. If you are using other data sources, you have to load a file and write the `parse_file` to parse it to a list of dictionary. 
To generate the outputs of each example you have to write functions in the format of `get$name$val`. each time one of the items in the list of dictionary is provided to your function and you should return the value of `$name` inside this function. At the end the output for each example will contain all the keys from the output of function `get$name$val` as `$name`. For getting one example of the reader, you have to call `run()` and it will yield one example at a time.
For an example, If you datasource is a json that has list of items looking like the following, you will just load it and set the keys.
```json
{
	"Sentence": "This book is interestingly boring. I tried to read it 10 times and each time I just felt sleep immediately.",
	"Label": 'negative',
	 "Optimistic": False
}
```
As we want our output for each item to contain the following keywords, we will define these functions.
```python
class SentimentReader(RegrReader):
	def getSentenceval(self, item):
		return item['Sentence']
		
	def getNegativeLabelval(self, item):
		if item['Label'] == 'negative':
			return True
		else:
			return False
			
	def getPositiveLabelval(self, item):
		if item['Label'] == 'positive':
			return True
		else:
			return False
	
	def getOptimisticLabelval(self, item):
		return item['Optimistic']
``` 

The output of such Reader will be a list of following items.
```json
{
	"Sentence": "This book is interestingly boring. I tried to read it 10 times and each time I just felt sleep immediately.",
	"PositiveLabel": False,
	"NegativeLabel": True,
	"Optimistic": False
}
```
