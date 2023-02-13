# Step 1
## example1
```python
with Graph() as graph:
    sentence = Concept()
    phrase = Concept()
    word = Concept()
    pair = Relation(arguments = 2)
    sentence_contains_phrase = sentence.contains(phrase)
    sentence_contains_word = sentence.contains(word)
    phrase_contains_word = phrase.contains(word)
    with Graph() as sub_graph():
        people = word()
        organization = word()
        work_for = pair(people, organization)
```
## example2
```python
with Graph() as graph:
    review = Concept()
    sentence = Concept()
    word = Concept()
    review_contains_sentence = review.contains(sentence)
    sentence_contains_word = sentence.contains(word)
    with Graph() as sub_graph():
        sentiment = review()
```
# Step 2
Reader output will be something like this.
## example1
```python
{
  "sentence" : "John works for Michigan State University",
  "words" : ["John", "works", "for", "Michigan", "state", "university"],
  "words_labels" : ["L", "NONE", "NONE", "B", "I", "L"],
  "phrases" : [(0,1), (3,3)],
  "people" : [(0,1), ],
  "organization" : [(3,3), ],
  "work_for" : [[(0,1),(3,3)], ],
}
```
## example2
The task is sentiment analysis 
```python
{
  "review" : "Excellent experience! Friendly staff, very good food. Wasn't busy during standard lunch rush hours, which I found surprising after tasting the food!",
  "label" : 1.00
}
```
# Step 3
## example1
```python
sentence['raw'] = ReaderSensor('sentence')
# --> str:  "John works for Michigan State University"
phrase[people] = LabelReaderSensor('people')
# --> array[1] : [(0,1)]
phrase[organization] = LabelReaderSensor('organization')
# --> array[1] : [(3,3)]
relation[work_for] = LabelReaderSensor('work_for')
# --> [((0,1), (3,3))]
word['boundary'] = LabelReaderSensor('words_labels') 
```
## example2
```python
review['raw'] = ReaderSensor('review')
# --> str:  "Excellent experience! Friendly staff, very good food. Wasn't busy during standard lunch rush hours, which I found surprising after tasting the food!"
review[sentiment] = LabelReaderSensor('label')
# --> float : 1.00
```
# Step 4
## example1
```python
sentence_contains_word.forward = SimpleTokenizorSensor('raw')
# --> array[6]: ["John", "works", "for", "Michigan", "state", "university"]
word['embed'] = BertLearner('raw') 
# --> array[6,N] ( N is the size of feature vector) 
word['boundary'] = BilLearner('embed') 
# --> array[6, N*M] ( N is size of BIL which is 3 and M is size of labels with is 2]
phrase_contains_word.backward = BilTransformSensor('boundary', transform='mean') 
# --> array[L, 2] ( L is the number of phrases obtained from this function) the 2 indicates the (start, length) 
phrase[people] = LogisticRegression('embed') 
# --> array[L]
pair[work_for] = LogisticRegression('embed', "(x, y) -> (pos(x) ="NP" & people(x)>0.5, org(y)>0.5)") 
# --> array[L1,L2] --> L1 is a set of candidates generated on the first argument and L2 is the set of candidates # generated in the second argument
```
## example2
```python
review_contains_sentence.forward = SentenceSlicer('raw')
# --> array[3]: ["Excellent experience!", "Friendly staff, very good food.", "Wasn't busy during standard lunch rush hours, which I found surprising after tasting the food!"]
sentence_contains_word.forward = SimpleTokenizerSensor('raw', function="concat")
# --> array[22]: ["Excellent", "experience!", "Friendly", "staff", "very", "good", "food.", "Wasn't", "busy", "during", "standard", "lunch", "rush", "hours", "which", "I", "found", "surprising", "after", "tasting", "the", "food!"]
word['embed'] = BertSensor('raw') 
# --> array[6,N] ( N is the size of feature vector) 
word['boundary'] = BilLearner('embed') 
# --> array[6, N*M] ( N is size of BIL which is 3 and M is size of labels with is 2]
phrase_contains_word.backward = BilTransformSensor('boundary', transform='mean') 
# --> array[L, 2] ( L is the number of phrases obtained from this function) the 2 indicates the (start, length) 
phrase[people] = LogisticRegression('embed') 
# --> array[L]
pair[work_for] = LogisticRegression('embed', "(x, y) -> (pos(x) ="NP" & people(x)>0.5, org(y)>0.5)") 
# --> array[L1,L2] --> L1 is a set of candidates generated on the first argument and L2 is the set of candidates # generated in the second argument
```
# Step 5
Data Nodes : 
```python
DN{'raw': "John works for Michigan State University", words:[DN{'raw' : "John", "embed" : x, "boundary" : y}, 5 more DN], phrases:[DN{"raw" : mean of tokens, "embed" : mean of x, people: 0.8}, some more DN] }
DN{'work_for' : [[x1, y1, 0.8]]}
```

Rule number 1: there must be a hierarchy which every intermediate concept (e.g. phrase) declares its output based on the physical address of the finest level of granularity (e.g. word, character). We need an absolute address for everything. 

Rule number 2: The `raw` keyword in concept property definition should always refer to the actual data

Q1 : edges should have different functions to convert different properties from node to node? [we need to have aggregation functions.]

