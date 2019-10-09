import numpy as np

if __package__ is None or __package__ == '':
    from graph import Graph
else:
    from .graph import Graph
    
# Class representing single data instance
class DataNode:
   
    def __init__(self, instanceID = None, instanceValue= None, ontologyNode = None, childInstanceNodes = None):
        self.instanceID = instanceID                 # The data instance (e.g. paragraph number, sentence number, token number, etc.)
        self.instanceValue = instanceValue
        self.ontologyNode = ontologyNode             # Reference to the ontology graph node which this instance of 
        self.childInstanceNodes = childInstanceNodes # List child data nodes this instance was segmented into
        self.calculatedTypesOfPredictions = dict()   # Dictionary with types of calculated predictions (for learned model, from constrain solver, etc.) results for elements of the instance
        
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
    
    def inferILPConstrains(self, model_trail, *conceptsRelations):
        with model_trail:
            # collect all the candidates for concept and relation in conceptsRelations
            infer_candidates = set()
            for currentConceptOrRelation in conceptsRelations:
                
                currentCandidates = currentConceptOrRelation.candidates(self)
                if currentCandidates is None:
                    continue
                
                for currentCandidate in currentCandidates:
                    for candidateElement in currentCandidate:
                        infer_candidates.add(candidateElement)
    
            if len(infer_candidates) == 0:
                return 
            
            infer_candidates = list(infer_candidates) # change set to list
            no_candidateds = len(infer_candidates)
            
            # Collect probabilities for candidates 
            graphResultsForPhraseToken = dict()
            graphResultsForPhraseRelation = dict()
            graphResultsForTripleRelations = dict()
    
            for currentConceptOrRelation in conceptsRelations:
                currentCandidates = currentConceptOrRelation.candidates(self)
                
                if currentCandidates is None:
                    continue
                
                for currentCandidate in currentCandidates:
                    if len(currentCandidate) == 1:
                        if currentConceptOrRelation not in graphResultsForPhraseToken:
                            graphResultsForPhraseToken[currentConceptOrRelation] = np.zeros((no_candidateds, ))
                        
                        currentProbability = currentConceptOrRelation.predict(self, (currentCandidate[0], ))
                        graphResultsForPhraseToken[currentConceptOrRelation][infer_candidates.index(currentCandidate[0])] = currentProbability
                        
                    elif len(currentCandidate) == 2:
                        if currentConceptOrRelation not in graphResultsForPhraseRelation:
                            graphResultsForPhraseRelation[currentConceptOrRelation] = np.zeros((no_candidateds, no_candidateds))
                        
                        currentProbability = currentConceptOrRelation.predict(self, (currentCandidate[0], currentCandidate[1]))
                        graphResultsForPhraseRelation[currentConceptOrRelation][infer_candidates.index(currentCandidate[0])][infer_candidates.index(currentCandidate[1])] = currentProbability
    
                    elif len(currentCandidate) == 3:
                        if currentConceptOrRelation not in graphResultsForTripleRelations:
                            graphResultsForTripleRelations[currentConceptOrRelation] = np.zeros((no_candidateds, no_candidateds, no_candidateds))
                            
                        currentProbability = currentConceptOrRelation.predict(self, (currentCandidate[0], currentCandidate[1], currentCandidate[2]))
                        graphResultsForTripleRelations[currentConceptOrRelation][infer_candidates.index(currentCandidate[0])][infer_candidates.index(currentCandidate[1])][infer_candidates.index(currentCandidate[2])] = currentProbability
                        
                    else:
                        pass
    
            # Call ilpOntsolver with the collected probabilities
            myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(self.ontologyNode.getOntologGraph())
            tokenResult, pairResult, tripleResult = myilpOntSolver.calculateILPSelection(infer_candidates, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForTripleRelations)
            
            #  Create Trail with returned results
            infered_trial = Trial()

            for concept in tokenResult:
                for infer_candidate in infer_candidates:
                    infered_trial[concept, (infer_candidate, )] = tokenResult[concept][infer_candidate] 
                    
            for concept in pairResult:
                for infer_candidate1 in infer_candidates:
                    for infer_candidate2 in infer_candidates:
                        if infer_candidate1 != infer_candidate2:
                            infered_trial[concept, (infer_candidate1, infer_candidate2)] = tokenResult[concept][infer_candidate1, infer_candidate2] 
                            
            for concept in tripleResult:
                for infer_candidate1 in infer_candidates:
                    for infer_candidate2 in infer_candidates:
                        if infer_candidate1 != infer_candidate2:
                            for infer_candidate3 in infer_candidates:
                                if infer_candidate2 != infer_candidate3:
                                    infered_trial[concept, (infer_candidate1, infer_candidate2, infer_candidate3)] = tokenResult[concept][infer_candidate1, infer_candidate2, infer_candidate3] 
                
            return infered_trial
        
            # TODO later
            # Add ILPConstrain results to appropriate dataNodes and create Trail with returned results
            for childDataNode in self.childInstanceNodes:
                
                if childDataNode.instance in infer_candidates:
                    
                    for concept in tokenResult:
                        childDataNode.setPredictionResultForConcept("ILPConstrain", concept, prediction=tokenResult[concept][infer_candidates.index(childDataNode.instance)])
                        
                    for concept in pairResult:
                        currentCandidates = concept.candidates(self)
                        for currentCandidate in currentCandidates:
                            if len(currentCandidate) != 2:
                                continue
                            
                            if currentCandidate[0] == childDataNode.instance:
                                prediction=tokenResult[concept][infer_candidates.index(childDataNode.instance)][infer_candidates.index(currentCandidate[1])]
                                childDataNode.setPredictionResultForConcept("ILPConstrain", 1, concept, currentCandidate[1], prediction=current_prediction)
                                
                    for concept in tripleResult:
                        currentCandidates = concept.candidates(self)
                        for currentCandidate in currentCandidates:
                            if len(currentCandidate) != 3:
                                continue
                            
                            if currentCandidate[0] == childDataNode.instance:
                                current_prediction=tokenResult[concept][infer_candidates.index(childDataNode.instance)][infer_candidates.index(currentCandidate[1])][infer_candidates.index(currentCandidate[2])]
                                childDataNode.setPredictionResultForConcept("ILPConstrain", 1, concept, currentCandidate[1], currentCandidate[2], prediction=current_prediction)
            