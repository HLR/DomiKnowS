# Graph Coloring Tutorial
This tutorial is created to show the ability of the framework to do inference on problems. 
## Problem
There are a bunch of cities and each city can have a firestation in it. Each city has a set of neighbors and we want to find the optimum assignment of firestation to cities so that each city either has a firestation or has a neighbor that has a firestation. 

## Framework Modeling
### Define the Graph
Each program in the Domiknows framework starts with a concept graph which defines the concepts interacting inside the problem world. 
Here we have a `world` , several `city`, `firestationCity`, and the concept of `neighbor`
```python
with Graph('global') as graph:  
        world = Concept(name='world')  
        city = Concept(name='city')    
        neighbor = Concept(name='neighbor')            
        firestationCity = city(name='firestationCity')
```
Notice here, for `firestationCity`, instead of creating another `Concept`, we can use concept `city` to create a concept which will impose a hierachy between `city` and `firestationCity`. Please refer to ['inherit declaration'](/docs/KNOWLEDGE.md#inherit-declaration) for more details. This is equivalent to
```python
firestationCity = city(name='firestationCity')
firestationCity.is_a(city)
```
where `is_a()` is a relationship, as will be introduced below.

In addition to the concepts, we have to introduce the relationships between concepts in one of the forms of `is_a`, `has_a` or `contains` keywords. 
```python
(world_contains_city,) = world.contains(city)  
(city1, city2) = neighbor.has_a(arg1=city, arg2=city)  
```

on last thing that we have to introduce inside our graph declaration is the set of rules we want to apply on the inference.
We add the following line to the previous code.
``` python 
orL(firestationCity, ('x',), existsL(('y',), andL(neighbor, ('x', 'y'), firestationCity, ('y',))), ('x',))
```
This constraint is expressing that each city is either of type `firestationcity` or `has_a` `neighbor` that is a `firestationCity`.
More constraint notion usage can be find in [Constraint](docs/KNOWLEDGE.md#constraints) section of the documentation.

Please refer to [User Pipeline](/docs/PIPELINE.md#1-knowledge-declaration) and [Knowledge Declaration](/docs/KNOWLEDGE.md) in the documentation for more specification.

### Define the Reader
Next step is to define the data of the problem. In this step we have to define a reader class which will load the inputs of the datasource into our framework. Reader class has a free style of coding and the only limitation is that it has to provide an iterative object over data which each data is a dictionary containing keyword and values. 
```python
class CityReader:  
    def __init__(self,):  
        self.data = [[1, 2, 3, 4, 5, 6, 7, 8, 9]]  
        self.links = [{  
	          1: [1, 2, 3, 4, 5],  
			  2: [1, 2, 6],  
			  3: [1, 3],  
			  4: [1, 4],  
			  5: [1, 5],  
			  6: [6, 7, 8, 9, 2],  
			  7: [6, 7],  
			  8: [6, 8],  
			  9: [6, 9]  
			  }]  
  
    def run(self,):  
        for i in range(len(self.data)):  
            item = {'world':[0], 'city': self.data[i], 'links': self.links[i]}  
            yield item
```     
This example only contains one world, and one set of data, but it still provides an iterative object as the result of the function `run()`

### Sensor Declaration
The interaction between the data and the graph inside the framework is by using a set of sensors. Furtheremore, sensors are used to prepare the data for any further processing in the flow of the framework before preparing the outcome for the inference phase. 
We can use the default `ReaderSensor` to read the `city` , `world` and use default `CandidateReaderSensor` to read `neighbor` instances.
We should define an Edge sensor connecting the world instances to the cities read from our datasource. 
**There is a problem in the definition of this sensor**
```python
class DummyCityEdgeSensor(TorchEdgeSensor): # Get world to city edge  
  def forward(self,) -> Any:  
        self.inputs.append(self.context_helper[self.edges[0].fullname])  
        return self.inputs[0]
```
This sensor  maps a world instance to a set of cities. 

As each program in our framework requires variable as input to the inference phase and variables are only valid as outcomes of a learner. We define a learner on the `city` conept and the subconcept of `firestationCity`.
**This learner doesn't seem right to me**
```python
class DummyCityLearner(TorchLearner):
  def forward(self,) -> Any:  
        result = torch.zeros(len(self.inputs[0]), 2)  
        for t in result: # Initially all cities are firestation cities  
		  t[1] = 1  
		  t[0] = 0  
	  return result
  ```
To enable learning on each learner, we have to define a label and assign this to the same instance in the graph based on a [Multiple Assignment Convention](/docs/MODEL.md#multiple-assigment-convention). 
```python
class DummyCityLabelSensor(TorchSensor): # Get Truth for Fire station classification  
  def __init__(self, *pres, label=True):  
        super().__init__(*pres, label=label)  
  
    def forward(self,) -> Any:  
        return []
```    

### Model Declaration
The next step toward solving a problem in our framework is to define a model flow or declaration for each example of the data.
```python
def model_declaration():
```
we start by linking the `ReaderSensor`s to the concepts and properties of the graph.
```python
world['index'] = ReaderSensor(keyword='world')
```
then, we link the `world` and `city` concepts:
```python
world_contains_city['forward'] = TorchEdgeReaderSensor(to='index', keyword='city', mode='forward')
```
Then we define the `CandidateReaderSensor` to read the `neighbor` concept into the graph.
```python
def readNeighbors(data, datanodes_edges, index, datanode_concept1, datanode_concept2):  
    if index[1] + 1 in data[index[0] + 1]: # data contain 'links' from reader  
		return 1  
	else:  
        return 0    
neighbor['raw'] = CandidateReaderSensor(label=False, forward=readNeighbors, keyword='links')
```
Next, we apply the learner and the label sensor to the `city[firestationCity]`.
```python
city[firestationCity] = DummyCityLearner('raw', edges=[world_contains_city['forward'], neighbor['raw']])  
city[firestationCity] = DummyCityLabelSensor(label=True)
```
Add the end of the definition of the `model_declaration` function we have to add the following lines to return an executable instance from the declaration of the graph attached to the new sensors and learners.
```python
program = LearningBasedProgram(graph, PoiModel)  
return program
```
  
  ### Model Execution
  To run the model, you have to call the reader and the `model_declaration`.
```python
  lbp = model_declaration()  
  dataset = CityReader().run()  # Adding the info on the reader
```
Next, we have to populate `datanode` and call inference on each sample.
```python
for datanode in lbp.populate(dataset=dataset, inference=True):  
  # call solver  
  conceptsRelations = (firestationCity, neighbor)    
  datanode.inferILPConstrains(*conceptsRelations, fun=None, minimizeObjective=True)   
  
  result = []  
  for child_node in datanode.getChildDataNodes():  
      s = child_node.getAttribute('raw')  
      f = child_node.getAttribute(firestationCity, 'ILP').item()  
      if f > 0:  
          r = (s, True)  
      else:  
          r = (s, False)  
      result.append(r)
```   
