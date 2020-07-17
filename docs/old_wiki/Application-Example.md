
Data Model:
```python
import document, phrase, sentence, word, concept

email = document("email")
with email:
    property("sender")
    property("subject")
    property("receiver")
    property("attachment")
spam = email("spam")
business = email("business")
personal = email("personal")
advertisement = document("advertisement")
person = phrase("person")
location = phrase("location")
with advertisement:
    property("category")
    property("organization")
dateTimePhrase = phrase("dateTime")
with dateTimePhrase:
    property("time")
eventPhrase = phrase("event")
with phrase("greeting") as greetingPhrase: 
    property("formality")
phrasePair = concept("phrasePair")
phrasePair.has_a(phrase, phrase)
EventTimePair = phrasePair("EventTimeRelation")
EventTimePair.has_a(dateTimePhrase, eventPhrase)
with EventTimePair:
    property("correlated")
```
Data population declaration:
```python
import ReaderSensor

```

ComposedTrainer Class Description:
```python
import ReaderSensor
class ComposedTrainer():
	def __init__(self):
		pass
	@property
	def experiments():
		return {}

	def train(self, iterations, readers, experiment_name, constraints=None):
		_readers = {} 
		for key, reader in readers.items():
			_readers[key] = []
		    _readers[key].extend(itertools.tee(reader, iterations))
		for iteration in range(iterations):
			data = {}
			for key in readers.keys():
				data[key] = next(_readers[key][iteration])

			for _input in self.experiments[experiment_name][inputs]:
				for sensor in _input.find(ReaderSensor):
					sensor.fill_data(data[sensor.reader_key])
			for output in self.experiments[experiment_name][outputs]:
				if isinstance(output, Property):
					pass
				elif isinstance(output, Concept):
					output = output['is_a']
				else:
					raise TypeError
				
			
			
			
			

	def predict(self, readers, constraints=None):
		pass
```


Spam Detector Trainer:
```python
import ComposedTrainer

class CalendarManager(ComposedTrainer):
	def __init__(self):
		super().__init__()
		self.inputs = [document, ]
		self.outputs = [person, eventPhrase, phrase[DateTimePhrase], ]
		self.experiments = {
			"trainpipeline": [{"inputs": [sentence], "output": [person]}, {"input": [document], "output":[eventPhrase, phrase[DateTimePhrase]}], 
			"trainJointly" : [{"input": [document], "output":[eventPhrase, phrase[DateTimePhrase], person}],
			"trainJointlyPreTrained": [{"inputs": [sentence], "output": [eventPhrase, person]}, {"input": [document], "output":[eventPhrase, phrase[DateTimePhrase]}]
		}	    	    
	def loss_func(self, pred, truth):
		pass
		
	def train(self, iterations, readers, experiment_name, constraints=None):
		pass

	def predict(self, reader, constraints=None):
		pass

	def traincostrainted(self, iterations, readers, experiment_name):
		cnstraints = [Class1OfConstraints, ]
		# Strcutured Loss
		self.train(iterations, readers, experiment_name, constraints)
		
	def predictconstrainted(self, reader):
		cnstraints = [Class1OfConstraints, ]
		#Optimization Problem over predictions
		self.predict(readers, constraints)

	
	    
	    
```
Applications:
```python
#Spam Detector Application:
document.populate(ReaderSensor(path))
x = spamDetector.run()
if spam.predict(x.email) >= 0.7:
    inbox.locate(x, "spam Folder")
    
#Calendar updater:
x = CalendarManager.predict(new_info)
x = {
    "document": {
        "d1": {"body": "Hi john, please be at the group meeting tomorrow 9 am in my office", "sentence": ["#s1", "#s2"]}
    },
    "sentences": {
        "s1": {
            "raw": "Hi john",
            "phrases": ["#p1", "#p2"]
        },
        "s2": {
            "raw": "please be at the group meeting tomorrow 9 am in my office",
            "phrases": ["#p3", "#p4", "#p5", "#p6"]
        }
    },
    "phrases": {
        "p1":
            {
                "raw": "Hi",
                eventPhrase: 0.1, dateTimePhrase: 0.05, greetingPhrase: 0.7, person: 0.04, location: 0.4
            },
        "p2":
            {
                "raw": "John",
                eventPhrase: 0.02, dateTimePhrase: 0.1, greetingPhrase: 0.07, person: 0.64, location: 0.3
            },
        "p3":
            {
                "raw": "group meeting",
                eventPhrase: 0.7, dateTimePhrase: 0.15, greetingPhrase: 0.0003, person: 0.01, location: 0.5
            },
        "p4":
            {
                "raw": "tomorrow",
                eventPhrase: 0.1, dateTimePhrase: 0.7, greetingPhrase: 0.03, person: 0.14, location: 0.5
            },
        "p5":
            {
                "raw": "tomorrow 9:30 am",
                eventPhrase: 0.3, dateTimePhrase: 0.9, greetingPhrase: 0.1, person: 0.04, location: 0.01
            },
        "p6":
            {
                "raw": "my office room 1204",
                eventPhrase: 0.1, dateTimePhrase: 0.04, greetingPhrase: 0.09, person: 0.14, location: 0.81
            }
    },
    "greetingPhrases": {
        "p1": {
            "formality": 0.2
        }
    },
    "dateTimePhrase": {
        "p1":
            {
                "time": None, "is_a": 0
            },
        "p2":
            {
                "time": None, "is_a": 0
            },
        "p3":
            {
                "time": None, "is_a": 0.15
            },
        "p4":
            {
                "time": None, "is_a": 0.7
            },
        "p5":
            {
                "time": "9:30", "is_a": 0.9
            },
        "p6":
            {
                "time": "1204", "is_a": 0
            },
    },
    "pair": {
        ("p1", "p1"): {"In Sentence Distance": 0},
        ("p1", "p2"): {"In Sentence Distance": 1},
        (): {},
        ("p3", "p4"): {"In Sentence Distance": 1},
        ("p3", "p5"): {"In Sentence Distance": 2},
    },
    "eventTimePair": {
        ("p1", "p1"): {"is_a": 0, 'correlated': 0.4},
        ("p1", "p2"): {"is_a": 0, 'correlated': 0.9},
        (): {},
        ("p3", "p4"): {"is_a": 1, 'correlated': 0.7},
        ("p3", "p5"): {"is_a": 1, 'correlated': 0.8},
    },
    "email": {
        "d1": {
            "sender": "f", "receiver": "b", "attachment": "path", "subject": "meeting"
        }
    },
}
for ph in x.phrases:
    if eventPhrase.predict(ph) >= 0.5:
        calendar.add(ph, EventTimePair.find(arg1=ph, threshold=0.6, key="correlated"))
        calendar.add(ph, EventTimeRelation.find(arg1=ph))


# Appropriate Advertisement Finder
x = AdvertisementAttentionSolver.predict(new_info)

```
       

        
  