if __package__ is None or __package__ == '':
    from graph import Graph
else:
    from .graph import Graph
    
# Class representing single data instance
class DataNode:
    instance = None                        # The data instance (e.g. paragraph number, sentence number, token number, etc.)
    
    ontologyNode = None                    # Reference to the ontology graph node which this instance of 
   
    childInstanceNodes =  list()           # List child data nodes this instance was segmented into
    learnedModel = None                    # Reference to learned model able to provide probability for instance classification or relation
    calculatedTypesOfPredictions = dict()  # Dictionary with types of calculated predictions (for learned model, from constrain solver, etc.) results for elements of the instance
    
    def __init__(self, instance = None, ontologyNode = None, childInstanceNodes = None, learnedModel = None):
        self.instance = instance
        self.ontologyNode = ontologyNode
        self.childInstanceNodes = childInstanceNodes
        self.learnedModel = learnedModel
        
    def getInstance(self):
        return self.instance
    
    def getOntologyNode(self):
        return self.ontologyNode
    
    def getChildInstanceNodes(self):
        return self.childInstanceNodes
    
    # Get set of types of calculated prediction stored in the graph
    def getCalculatedTypesOfPredictions(self):
        return self.calculatedTypesOfPredictions


    # Set the prediction result data of the given type for the given concept
    def setPredictionResultForConcept(self, typeOfPrediction, conceptName, prediction=None):
        if typeOfPrediction not in self.calculatedTypesOfPredictions:
            self.calculatedTypesOfPredictions[typeOfPrediction] = dict()
            
        if "Concept" not in self.calculatedTypesOfPredictions[typeOfPrediction]:
            self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"] = dict()
        
        self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"][conceptName] = prediction
   
    # Set the prediction result data of the given type for the given relation, positionInRelation indicates where is the current instance
    def setPredictionResultForRelation(self, typeOfPrediction, positionInRelation, relationName, *instances, prediction=None):
        if typeOfPrediction not in self.calculatedTypesOfPredictions:
            self.calculatedTypesOfPredictions[typeOfPrediction] = dict()
            
        if "Relation" not in self.calculatedTypesOfPredictions[typeOfPrediction]:
            self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"] = dict()
             
        if relationName not in self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"]:
            self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"][relationName] = dict()
            
        instanceRelationDict = self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"][relationName]
               
        updatedInstances = instances[0 : positionInRelation - 1] + (self.instance,) + instances[positionInRelation-1: ]
        
        for relatedInstanceIndex in range(len(updatedInstances) - 1):    
            currentInstance = updatedInstances[relatedInstanceIndex] 
            
            if currentInstance not in instanceRelationDict:
                instanceRelationDict[currentInstance] = dict()
            
            instanceRelationDict = instanceRelationDict[currentInstance]
        
        instanceRelationDict[updatedInstances[len(updatedInstances) - 1]] = prediction
   
    # Get the prediction result data of the given type for the given concept
    def getPredictionResultForConcept(self, typeOfPrediction, conceptName):
        myReturn = None
        
        if typeOfPrediction not in self.calculatedTypesOfPredictions:
            pass
        elif "Concept" not in self.calculatedTypesOfPredictions[typeOfPrediction]:
            pass
        elif conceptName not in self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"]:
            pass
        else:
            myReturn = self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"][conceptName]
                                                                                         
        if (myReturn is None) and (self.learnedModel is not None):
            myReturn = self.learnedModel.getPredictionResultForConcept(self.instance, conceptName)
            
            if myReturn != None:
                self.setPredictionResultForConcept(typeOfPrediction, conceptName, prediction = myReturn)
            
        return myReturn
            
    # Get the prediction result data of the given type for the given concept or relation (then *tokens is not empty)
    def getPredictionResultForRelation(self, typeOfPrediction, positionInRelation, relationName, *instances):
        myReturn = None

        if typeOfPrediction not in self.calculatedTypesOfPredictions:
            pass         
        elif "Relation" not in self.calculatedTypesOfPredictions[typeOfPrediction]:
            pass
        else:
            instanceRelationDict = self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"][relationName]
            
            updatedInstances = instances[0 : positionInRelation - 1] + (self.instance,) + instances[positionInRelation-1: ]

            for relatedInstanceIndex in range(len(updatedInstances) - 1):    
                currentInstance = updatedInstances[relatedInstanceIndex] 
                
                if currentInstance not in instanceRelationDict:
                    break
                
                instanceRelationDict = instanceRelationDict[currentInstance]
                
                if relatedInstanceIndex == len(updatedInstances) - 2:
                    myReturn = instanceRelationDict[updatedInstances[len(updatedInstances) - 1]]
            
        if (myReturn is None) and (self.learnedModel is not None):
            myReturn = self.learnedModel.getPredictionResultForRelation(self.instance, relationName, instances)
            
            if myReturn != None:
                self.setPredictionResultForRelation(typeOfPrediction, relationName, instances, prediction = myReturn)
            
        return myReturn       