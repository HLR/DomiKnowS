import numpy as np

if __package__ is None or __package__ == '':
    from graph import Graph
    from graph import Trial
    from solver import ilpOntSolverFactory
else:
    from .graph import Graph
    from . import Trial
    from ..solver import ilpOntSolverFactory

# Class representing single data instance and links to the children data nodes which represent sub instances this instance was segmented into 
class DataNode:
   
    def __init__(self, instanceID = None, instanceValue = None, ontologyNode = None, childInstanceNodes = None, attributes = None):
        self.instanceID = instanceID                 # The data instance id (e.g. paragraph number, sentence number, token number, etc.)
        self.instanceValue = instanceValue           # Optional value of the instance 
        self.ontologyNode = ontologyNode             # Reference to the ontology graph node (e.g. Concept) which is the type of this instance
        self.childInstanceNodes = childInstanceNodes # List of child data nodes this instance was segmented into
        self.attributes = attributes                 # Dictionary with additional node's attributes
        self.calculatedTypesOfPredictions = dict()   # Dictionary with types of calculated predictions results (for learned model, from constrain solver, etc.)
        
    def getInstanceID(self):
        return self.instanceID
    
    def getInstanceValue(self):
        return self.instanceValue
    
    def getOntologyNode(self):
        return self.ontologyNode
    
    def getAttributes(self):
        return self.attributes.kesy()
    
    def getAttribute(self, key):
        if key in self.attributes:
            return self.attributes[key]
        else:
            return None
            
    def getChildInstanceNodes(self):
        return self.childInstanceNodes
    
    # Get set of types of calculated prediction stored in the graph
    def getCalculatedTypesOfPredictions(self):
        return self.calculatedTypesOfPredictions

    # Set the prediction result data of the given type for the given concept
    def setPredictionResultForConcept(self, typeOfPrediction, concept, prediction=None):
        if typeOfPrediction not in self.calculatedTypesOfPredictions:
            self.calculatedTypesOfPredictions[typeOfPrediction] = dict()
            
        if "Concept" not in self.calculatedTypesOfPredictions[typeOfPrediction]:
            self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"] = dict()
        
        self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"][concept.name()] = prediction
   
    # Set the prediction result data of the given type for the given relation, positionInRelation indicates where is the current instance
    def setPredictionResultForRelation(self, typeOfPrediction, positionInRelation, relation, *instances, prediction=None):
        if typeOfPrediction not in self.calculatedTypesOfPredictions:
            self.calculatedTypesOfPredictions[typeOfPrediction] = dict()
            
        if "Relation" not in self.calculatedTypesOfPredictions[typeOfPrediction]:
            self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"] = dict()
             
        if relation.name() not in self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"]:
            self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"][relation.name()] = dict()
            
        instanceRelationDict = self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"][relation.name()]
               
        updatedInstances = instances[0 : positionInRelation - 1] + (self.instance,) + instances[positionInRelation-1: ]
        
        for relatedInstanceIndex in range(len(updatedInstances) - 1):    
            currentInstance = updatedInstances[relatedInstanceIndex] 
            
            if currentInstance not in instanceRelationDict:
                instanceRelationDict[currentInstance] = dict()
            
            instanceRelationDict = instanceRelationDict[currentInstance]
        
        instanceRelationDict[updatedInstances[len(updatedInstances) - 1]] = prediction
   
    # Get the prediction result data of the given type for the given concept
    def getPredictionResultForConcept(self, typeOfPrediction, concept):
        myReturn = None
        
        if typeOfPrediction not in self.calculatedTypesOfPredictions:
            pass
        elif "Concept" not in self.calculatedTypesOfPredictions[typeOfPrediction]:
            pass
        elif concept.name() not in self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"]:
            pass
        else:
            myReturn = self.calculatedTypesOfPredictions[typeOfPrediction]["Concept"][concept.name()]
                                                                                         
        if (myReturn is None) and (self.learnedModel is not None):
            myReturn = concept.getPredictionFor(self.instance)
            
            if myReturn != None:
                self.setPredictionResultForConcept(typeOfPrediction, concept.name(), prediction = myReturn)
            
        return myReturn
            
    # Get the prediction result data of the given type for the given concept or relation (then *tokens is not empty)
    def getPredictionResultForRelation(self, typeOfPrediction, positionInRelation, relation, *instances):
        myReturn = None

        if typeOfPrediction not in self.calculatedTypesOfPredictions:
            pass         
        elif "Relation" not in self.calculatedTypesOfPredictions[typeOfPrediction]:
            pass
        else:
            instanceRelationDict = self.calculatedTypesOfPredictions[typeOfPrediction]["Relation"][relation.name()]
            
            updatedInstances = instances[0 : positionInRelation - 1] + (self.instance,) + instances[positionInRelation-1: ]

            for relatedInstanceIndex in range(len(updatedInstances) - 1):    
                currentInstance = updatedInstances[relatedInstanceIndex] 
                
                if currentInstance not in instanceRelationDict:
                    break
                
                instanceRelationDict = instanceRelationDict[currentInstance]
                
                if relatedInstanceIndex == len(updatedInstances) - 2:
                    myReturn = instanceRelationDict[updatedInstances[len(updatedInstances) - 1]]
            
        if (myReturn is None) and (self.learnedModel is not None):
            updatedInstances = instances[0 : positionInRelation - 1] + (self.instance,) + instances[positionInRelation-1: ]
            myReturn = relation.getPredictionFor(updatedInstances)
            
            if myReturn != None:
                self.setPredictionResultForRelation(typeOfPrediction, relation.name(), instances, prediction = myReturn)
            
        return myReturn       
    
    # Calculate ILP prediction for this instance based on the provided list of concepts (or/and relations) 
    def inferILPConstrains(self, model_trail, *conceptsRelations):
        with model_trail:
            # Collect all the candidates for concepts and relations in conceptsRelations
            _instanceID = set() # Stores all the candidates to consider in the ILP constrains
            for currentConceptOrRelation in conceptsRelations:
                
                currentCandidates = currentConceptOrRelation.candidates(self)
                if currentCandidates is None:
                    continue
                
                for currentCandidate in currentCandidates:
                    for candidateElement in currentCandidate:
                        _instanceID.add(candidateElement)
    
            if len(_instanceID) == 0:
                return 
            
            infer_candidatesID = []
            for currentCandidate in _instanceID:
                infer_candidatesID.append(currentCandidate.instanceID)

            no_candidateds = len(infer_candidatesID)    # Number of candidates
            
            # Collect probabilities for candidates 
            graphResultsForPhraseToken = dict()
            graphResultsForPhraseRelation = dict()
            graphResultsForTripleRelations = dict()
    
            for currentConceptOrRelation in conceptsRelations:
                currentCandidates = currentConceptOrRelation.candidates(self)
                
                if currentCandidates is None:
                    continue
                
                for currentCandidate in currentCandidates:
                    if len(currentCandidate) == 1:   # currentConceptOrRelation is a concept thus candidates tuple has only single element
                        graphResultsForPhraseToken[currentConceptOrRelation.name] = np.zeros((no_candidateds, ))
                        
                    elif len(currentCandidate) == 2: # currentConceptOrRelation is a pair thus candidates tuple has two element
                        graphResultsForPhraseRelation[currentConceptOrRelation.name] = np.zeros((no_candidateds, no_candidateds))
                        
                    elif len(currentCandidate) == 3: # currentConceptOrRelation is a triple thus candidates tuple has three element
                        graphResultsForTripleRelations[currentConceptOrRelation.name] = np.zeros((no_candidateds, no_candidateds, no_candidateds))
                            
                    else: # No support for more then three candidates yet
                        pass
                    
            for currentConceptOrRelation in conceptsRelations:
                currentCandidates = currentConceptOrRelation.candidates(self)
                
                if currentCandidates is None:
                    continue
                
                for currentCandidate in currentCandidates:
                    if len(currentCandidate) == 1:   # currentConceptOrRelation is a concept thus candidates tuple has only single element
                        currentProbability = currentConceptOrRelation.predict(self, (currentCandidate[0].instanceID, ))
                        if currentProbability:
                            graphResultsForPhraseToken[currentConceptOrRelation.name][infer_candidatesID[currentCandidate[0].instanceID]] = currentProbability
                        
                    elif len(currentCandidate) == 2: # currentConceptOrRelation is a pair thus candidates tuple has two element
                        currentProbability = currentConceptOrRelation.predict(self, (currentCandidate[0].instanceID, currentCandidate[1].instanceID))
                        if currentProbability:
                            graphResultsForPhraseRelation[currentConceptOrRelation.name][infer_candidatesID[currentCandidate[0].instanceID]][infer_candidatesID[currentCandidate[1].instanceID]] = currentProbability
    
                    elif len(currentCandidate) == 3: # currentConceptOrRelation is a triple thus candidates tuple has three element     
                        currentProbability = currentConceptOrRelation.predict(self, (currentCandidate[0].instanceID, currentCandidate[1].instanceID, currentCandidate[2].instanceID))
                        if currentProbability:
                            graphResultsForTripleRelations[currentConceptOrRelation.name] \
                                [infer_candidatesID[currentCandidate[0].instanceID]][infer_candidatesID[currentCandidate[1].instanceID]][infer_candidatesID[currentCandidate[2].instanceID]]= currentProbability
                        
                    else: # No support for more then three candidates yet
                        pass
    
            # Call ilpOntsolver with the collected probabilities for chosen candidates
            myOntologyGraphs = {self.ontologyNode.getOntologyGraph()}
            
            for currentConceptOrRelation in conceptsRelations:
                currentOntologyGraph = currentConceptOrRelation.getOntologyGraph()
                
                if currentOntologyGraph is not None:
                    myOntologyGraphs.add(currentOntologyGraph)
                    
            myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(myOntologyGraphs)
            tokenResult, pairResult, tripleResult = myilpOntSolver.calculateILPSelection(infer_candidatesID, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForTripleRelations)
            
            #  Create Trail with returned results
            infered_trial = Trial()

            for concept in tokenResult:
                for infer_candidate in infer_candidatesID:
                    infered_trial[concept, (infer_candidate, )] = tokenResult[concept][infer_candidate] 
                    
            for concept in pairResult:
                for infer_candidate1 in infer_candidatesID:
                    for infer_candidate2 in infer_candidatesID:
                        if infer_candidate1 != infer_candidate2:
                            infered_trial[concept, (infer_candidate1, infer_candidate2)] = pairResult[concept][infer_candidate1, infer_candidate2] 
                            
            for concept in tripleResult:
                for infer_candidate1 in infer_candidatesID:
                    for infer_candidate2 in infer_candidatesID:
                        if infer_candidate1 != infer_candidate2:
                            for infer_candidate3 in infer_candidatesID:
                                if infer_candidate2 != infer_candidate3:
                                    infered_trial[concept, (infer_candidate1, infer_candidate2, infer_candidate3)] = tripleResult[concept][infer_candidate1, infer_candidate2, infer_candidate3] 
                
            return infered_trial
        
            # TODO later
            # Add ILPConstrain results to appropriate dataNodes and create Trail with returned results
            for childDataNode in self.childInstanceNodes:
                
                if childDataNode.instance in infer_candidatesID:
                    
                    for concept in tokenResult:
                        childDataNode.setPredictionResultForConcept("ILPConstrain", concept, prediction=tokenResult[concept][infer_candidatesID.index(childDataNode.instance)])
                        
                    for concept in pairResult:
                        currentCandidates = concept.candidates(self)
                        for currentCandidate in currentCandidates:
                            if len(currentCandidate) != 2:
                                continue
                            
                            if currentCandidate[0] == childDataNode.instance:
                                prediction=tokenResult[concept][infer_candidatesID.index(childDataNode.instance)][infer_candidatesID.index(currentCandidate[1])]
                                childDataNode.setPredictionResultForConcept("ILPConstrain", 1, concept, currentCandidate[1], prediction=current_prediction)
                                
                    for concept in tripleResult:
                        currentCandidates = concept.candidates(self)
                        for currentCandidate in currentCandidates:
                            if len(currentCandidate) != 3:
                                continue
                            
                            if currentCandidate[0] == childDataNode.instance:
                                current_prediction=tokenResult[concept][infer_candidatesID.index(childDataNode.instance)][infer_candidatesID.index(currentCandidate[1])][infer_candidatesID.index(currentCandidate[2])]
                                childDataNode.setPredictionResultForConcept("ILPConstrain", 1, concept, currentCandidate[1], currentCandidate[2], prediction=current_prediction)
            