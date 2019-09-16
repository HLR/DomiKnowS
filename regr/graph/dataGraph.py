if __package__ is None or __package__ == '':
    from graph import Graph
else:
    from .graph import Graph
    
# Class storing results of predictions for a single data instance
class DataGraph:
    instance = None                        # The data instance (e.g. sentence, paragraph, image, etc.)
    
    ontologyGraph = None                   # Reference to the graph representing the structure of the knowledge the system has learned to extract or classify  
    rootNode = None                        # Reference to the node in the ontology graph, which depict the root of the subgraph used for this data; if None then the root of the whole graph is used
    leafNodes = set()                      # Set with nodes indicating leafs of subgraph starting from rootNode, used for this data; if None then the whole graph is used
    
    childInstanceGraphs =  list()          # List with child data graph building the current data graph, if None then simple data 
    calculatedTypesOfPredictions = dict()  # Dictionary with types of calculated predictions (for learner, from constrainssolver, etc.)results for elements of the instance
    
    def __init__(self, instance = None, ontologyGraph = None, childInstanceGraphs = None, rootNode = None, leafNodes = None):
        self.instance = _instance

        self.childInstanceGraphs = childInstanceGraphs
        
        self.ontologyGraph = ontologyGraph
        self.rootNode = rootNode
        self.leafNodes = leafNodes
        
    def getInstance(self):
        return self.instance
    
    def getOntologyGraph(self):
        return self.ontologyGrpah
    
    def getChildInstanceGraphs(self):
        return self.childInstanceGraphs
    
    # Get set of types of calculated prediction stored in the graph
    def getCalculatedTypesOfPredictions(self):
        return self.calculatedTypesOfPredictions

     # Set the prediction result data of the given type for the given token - relation - other tokens
    def setPredictionResult(self, typeOfPrediction, token, conceptName, *tokens, prediction=None):
        if typeOfPrediction not in self.calculatedTypesOfPredictions:
            self.calculatedTypesOfPredictions[typeOfPrediction] = dict()
            
        if len(tokens) == 0:
            if "Concept" not in self.calculatedTypesOfPredictions[typeOfPrediction]:
                 self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"] = dict()
            
            if token not in self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"]:
                self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"][token] = dict()
        
            self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"][token][conceptName] = prediction
        else:
            relationName = conceptName
            
            if "Relation" not in self.calculatedTypesOfPredictions[typeOfPrediction]:
                self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"] = dict()
                
            if token not in self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"]:
                self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"][token] = dict()
            
            if relationName not in self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"][token]:
                self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"][token][relationName] = dict()
                
            tokenRelationDict = self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"][token][relationName]
            
            for relatedTokenIndex in range(len(tokens) - 1):    
                currentToken = tokens[relatedTokenIndex] 
                
                if currentToken not in tokenRelationDict:
                    tokenRelationDict[currentToken] = dict()
                
                tokenRelationDict = tokenRelationDict[currentToken]
            
            tokenRelationDict[tokens[len(tokens) - 1]] = prediction
   
    def getPredictionResult(self, typeOfPrediction, token, conceptName, *tokens):
        if typeOfPrediction not in self.calculatedTypesOfPredictions:
            return None
            
        if len(tokens) == 0:
            if "Concept" not in self.calculatedTypesOfPredictions[typeOfPrediction]:
                return None
            elif token not in self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"]:
                return None
            elif conceptName not in self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"][token]:
                return None
            else:
                return self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"][token][conceptName]
        else:
            relationName = conceptName
            
            if "Relation" not in self.calculatedTypesOfPredictions[typeOfPrediction]:
                return None
            elif token not in self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"]:
                return None
            elif relationName not in self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"][token]:
                return None
            else:
                tokenRelationDict = self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"][token][relationName]
                
                for relatedTokenIndex in range(len(tokens) - 1):    
                    currentToken = tokens[relatedTokenIndex] 
                    
                    if currentToken not in tokenRelationDict:
                        return None
                    
                    tokenRelationDict = tokenRelationDict[currentToken]
                
                return  tokenRelationDict[tokens[len(tokens) - 1]]