# Sentiment Analysis Tutorial
This tutorial is created to show the ability of the framework to do inference and learning altogether. 
## Problem
There is a text and we want to predict its polarity and sentiment by using deep neural networks.

## Framework Modeling
### Define the Graph
Each program in the Domiknows framework starts with a concept graph which defines the concepts interacting inside the problem world. 
Here we have a `sentence` , several `words1` and for our problem domain we have  `positive`, `negative`, and `optimistic` concepts.
```python
Graph.clear()  
Concept.clear()  
Relation.clear()  
  
with Graph('structure') as graph:  
        sentence = Concept(name='sentence') 
        word = Concept(name='word')  
	with Graph('problem')
        positive = Sentence(name='positive')    
        negative = Sentence(name='negative')            
        optimistic = Sentence(name='optimistic')
```

In addition to the concepts, we have to introduce the relationships between concepts in one of the forms of `contains` keyword. 
```python
Graph.clear()  
Concept.clear()  
Relation.clear()  
  
with Graph('structure') as graph:  
        sentence = Concept(name='sentence')  
        word = Concept(name='word') 
        (sentence_contains_word) = sentence.contains(word)
	with Graph('problem')
        positive = Sentence(name='positive')    
        negative = Sentence(name='negative')            
        optimistic = Sentence(name='optimistic')
```

on last thing that we have to introduce inside our graph declaration is the set of rules we want to apply on the inference.
We add the following line to the previous code.
``` python 
ifL(optimistic, ('x',), positive, ('x',))
```
This constraint is expressing that each if a sentence is `optimistic` it should also be `positive`.

### Define the Reader
Next step is to define the data of the problem. In this step we have to define a reader class which will load the inputs of the datasource into our framework. Reader class has a free style of coding and the only limitation is that it has to provide an iterative object over data which each data is a dictionary containing keyword and values. 
Here, we use the base Reader Class of our framework. 
our datasource is a list of objects, which each object is as follows.
```json
{
	"Sentence": "This book is interestingly boring. I tried to read it 10 times and each time I just felt sleep immediately.",
	"Label": 'negative',
	 "Optimistic": False
}
```
The base class of our framework for the reader is `RegrReader` which is located in `domiknows/data/reader.py`.
In order to define keywords for the output of the reader per example, we have to define functions in the pattern of `get$keyword$val(self, item)`.
```python
from domiknows.data.reader import RegrReader

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
This example only contains one world, and one set of data, but it still provides an iterative object as the result of the function `run()`. As an example, the same data from our datasource will be passed as follows.
```json
{
	"Sentence": "This book is interestingly boring. I tried to read it 10 times and each time I just felt sleep immediately.",
	"PositiveLabel": False,
	"NegativeLabel": True,
	"Optimistic": False
}
```