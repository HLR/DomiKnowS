# Walkthrough Example

The followings are the user's steps to using our framework.

- Dataset
- Knowledge Declaration
- **Model Declaration**
- Training and Testing
- Inference

## Model Declaration


In the model declaration, the user defines how external resources (raw data), external procedures (preprocessing), and trainable deep learning modules are associated with the concepts and properties in the graph.
We use `Reader`s, `Sensor`s, and `Learner`s accordingly for the model declaration to create a *"full program"* as `Program`.

To create a program, the user needs to first assign `Sensor`s and `Learner`s to `Property`s of `Concept`s in the graph. Then initiate a `Program` with the graph.

There are different pre-defined sensors for basic data operation with PyTorch. Users can also extend [base `Sensor`](./Technical%20API/Sensor/Class%20Sensor.md#TorchSensor) to customize for their task [by overriding `forward()` method](developer/MODEL.md#overriding-forward).

```python
phrase['text'] = ReaderSensor(keyword='tokens')
phrase['postag'] = ReaderSensor(keyword='postag')
```

In the example above, the first `ReaderSensor` is assigned to the properties `tokens`, `postag`. After reading the data about the Phrases, we also read the annotation of labels into the graph by using a customized ReaderSensor which we call FunctionalReaderSensor. In addition to having access to the DataLoader output, the FunctionalReaderSensor accepts a function as input at initialization which it will apply on the read data before generating the outputs. Remember to put label=True Whenever you are reading an annotation to the graph.

```python
def find_label(label_type):
        def find(data):
            try:
                label = torch.tensor([item==label_type for item in data])
            except:
                print(data)
                raise
            return label # torch.stack((~label, label), dim=1)
        return find
        raise
phrase[people] = FunctionalReaderSensor(keyword='label', forward=find_label('Peop'), label=True)
phrase[organization] = FunctionalReaderSensor(keyword='label', forward=find_label('Org'), label=True)
phrase[location] = FunctionalReaderSensor(keyword='label', forward=find_label('Loc'), label=True)
phrase[other] = FunctionalReaderSensor(keyword='label', forward=find_label('Other'), label=True)
phrase[o] = FunctionalReaderSensor(keyword='label', forward=find_label('O'), label=True)
```

Next, We define a word to vector sensor using a FunctionalSensor to generate the word representation from their string utilizing the spacy library.

```python
def word2vec(text):
    texts = list(map(lambda x: ' '.join(x.split('/')), text))
    tokens_list = list(nlp.pipe(texts))
    return torch.tensor([tokens.vector for tokens in tokens_list])

phrase['w2v'] = FunctionalSensor('text', forward=word2vec)
```
Now as our data included the phrases directly, we want to merge phrases in the same sentence to create the concept nodes for sentence. To generate a concept from another one, we have to use the edges defined on the graph between those two concepts. Here we use the contains edge defined on the graph and as the default relationship was defined from a sentence to a phrase, here we use the keyword reversed after the relationship variable to indicate the reverse direction of the same relationship.

Whenever, you want to define the connection, you have to return a matric connecting the nodes from the source to the target concept in a form of 0 and 1s. Apart from that we also want to generate the feature Text containing the actual string feature for a sentence in the same sensor, because of that, we use a JointSensor which is able to connect one sensor to multiple proprties on the graph.

```python
def merge_phrase(phrase_text):
    return [' '.join(phrase_text)], torch.ones((1, len(phrase_text)))
sentence['text', rel_sentence_contains_phrase.reversed] = JointSensor(phrase['text'], forward=merge_phrase)
```

Next, we define a set of learners on top of the phrases to decide about each phrase class. In this scenario, we are assigning independent boolean classifers for each phrase type

```python
phrase[people] = ModuleLearner('w2v', module=Classifier(FEATURE_DIM))
phrase[organization] = ModuleLearner('w2v', module=Classifier(FEATURE_DIM))
phrase[location] = ModuleLearner('w2v', module=Classifier(FEATURE_DIM))
phrase[other] = ModuleLearner('w2v', module=Classifier(FEATURE_DIM))
phrase[o] = ModuleLearner('w2v', module=Classifier(FEATURE_DIM))
```

Next, we define the pair concept and use the ComposionCandidateSensor to generate pair candidates based on our phrases. This sensor receives one instance of each argument at a time and return True to make a candidate pair for that combination or False to skip the combination.

```python
pair[rel_pair_phrase1.reversed, rel_pair_phrase2.reversed] = CompositionCandidateSensor(
    phrase['w2v'],
    relations=(rel_pair_phrase1.reversed, rel_pair_phrase2.reversed),
    forward=lambda *_, **__: True)
```

Next, we define a property emb for the pair candidates based on the representation of their arguments. We use the relation links defined on the previous sensor to retrieve the w2v properties on each argument and concat them in the forward function of the FunctionalSensor.


```python
pair['emb'] = FunctionalSensor(
    rel_pair_phrase1.reversed('w2v'), rel_pair_phrase2.reversed('w2v'),
    forward=lambda arg1, arg2: torch.cat((arg1, arg2), dim=-1))
```

Then, we define our classifiers on top of the pair emb properties to decide about the type of each pair.

```python
pair[work_for] = ModuleLearner('emb', module=Classifier(FEATURE_DIM*2))
pair[located_in] = ModuleLearner('emb', module=Classifier(FEATURE_DIM*2))
pair[live_in] = ModuleLearner('emb', module=Classifier(FEATURE_DIM*2))
pair[orgbase_on] = ModuleLearner('emb', module=Classifier(FEATURE_DIM*2))
pair[kill] = ModuleLearner('emb', module=Classifier(FEATURE_DIM*2))

def find_relation(relation_type):
    def find(arg1m, arg2m, data):
        label = torch.zeros(arg1m.shape[0], dtype=torch.bool)
        for rel, (arg1,*_), (arg2,*_) in data:
            if rel == relation_type:
                i, = (arg1m[:, arg1] * arg2m[:, arg2]).nonzero(as_tuple=True)
                label[i] = True
        return label # torch.stack((~label, label), dim=1)
    return find
pair[work_for] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Work_For'), label=True)
pair[located_in] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Located_In'), label=True)
pair[live_in] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Live_In'), label=True)
pair[orgbase_on] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('OrgBased_In'), label=True)
pair[kill] = FunctionalReaderSensor(pair[rel_pair_phrase1.reversed], pair[rel_pair_phrase2.reversed], keyword='relation', forward=find_relation('Kill'), label=True)
```

`ModuleLearner` in the above code is used to calculate our predicted labels. Being a Learner, this sensor's parameters will change and update itself during training later. in the following. To assign the actaul labels we also use the `ReaderSensor`. Instead of overwriting the assignment, "Multiple Assignment" indicates the consistency of the `Sensor`s and `Learner`s assigned to a single `Property`. See ["Multiple Assignment" semantic](./developer/MODEL.md#multiple-assigment-convention) for more information.

____
[Goto next section (Training and Testing)](Training%20and%20Testing.md)
