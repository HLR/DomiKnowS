import numpy as np
import torch
from collections import OrderedDict 
import logging
import re

from regr.solver.ilpConfig import ilpConfig 
from torch.tensor import Tensor

if __package__ is None or __package__ == '':
    from graph import Graph
    from solver import ilpOntSolverFactory
else:
    from .graph import Graph
    from ..solver import ilpOntSolverFactory

# Class representing single data instance and links to the children data nodes which represent sub instances this instance was segmented into 
class DataNode:
   
    PredictionType = {"Learned" : "Learned", "ILP" : "ILP"}
   
    def __init__(self, instanceID = None, instanceValue = None, ontologyNode = None, childInstanceNodes = None, attributes = None):
        self.myLogger = logging.getLogger(ilpConfig['log_name'])

        self.instanceID = instanceID                     # The data instance id (e.g. paragraph number, sentence number, phrase  number, image number, etc.)
        self.instanceValue = instanceValue               # Optional value of the instance (e.g. paragraph text, sentence text, phrase text, image bitmap, etc.)
        self.ontologyNode = ontologyNode                 # Reference to the ontology graph node (e.g. Concept) which is the type of this instance (e.g. paragraph, sentence, phrase, etc.)
        self.childInstanceNodes = {}                     # Dictionary mapping child ontology type to List of child data nodes this instance was segmented into based on this type
        self.attributes = attributes                     # Dictionary with additional node's attributes
        self.predictions = {}                            # Dictionary with types of calculated predictions results (inferred: from learned model - Learned, by constrain solver - ILP, etc.)
      
    def __str__(self):
        if self.instanceValue:
            return self.instanceValue
        else:
            return 'instanceID ' + self.instanceID
        
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
    
    # Calculate ILP prediction for this instance based on the provided list of concepts (or/and relations) 
    def inferILPConstrains(self, *conceptsRelations):
        # Collect all the candidates for concepts and relations in conceptsRelations
        _instanceID = set() # Stores all the candidates to consider in the ILP constrains
        candidates_currentConceptOrRelation = OrderedDict()
        for currentConceptOrRelation in conceptsRelations:
            
            currentCandidates = currentConceptOrRelation.candidates(self)
            if currentCandidates is None:
                continue
            
            _currentInstanceIDs = set()
            for currentCandidate in currentCandidates:
                _currentInstanceIDs.add(currentCandidate)
                for candidateElement in currentCandidate:
                    _instanceID.add(candidateElement)

            candidates_currentConceptOrRelation[currentConceptOrRelation] = _currentInstanceIDs
          
        if len(_instanceID) == 0:
            return 
        
        infer_candidatesID = []
        for currentCandidate in _instanceID:
            infer_candidatesID.append(currentCandidate.instanceID)
        
        infer_candidatesID.sort()

        no_candidateds = len(infer_candidatesID)    # Number of candidates
        
        # Collect probabilities for candidates 
        graphResultsForPhraseToken = dict()
        graphResultsForPhraseRelation = dict()
        graphResultsForTripleRelations = dict()

        conceptOrRelationDict = {}
        for currentConceptOrRelation in conceptsRelations:
            conceptOrRelationDict[currentConceptOrRelation.name] = currentConceptOrRelation
            currentCandidates = candidates_currentConceptOrRelation[currentConceptOrRelation]
            
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
            currentCandidates = candidates_currentConceptOrRelation[currentConceptOrRelation]
            
            if currentCandidates is None:
                continue
            
            for currentCandidate in currentCandidates:
                if len(currentCandidate) == 1:   # currentConceptOrRelation is a concept thus candidates tuple has only single element
                    currentCandidate = currentCandidate[0]
                    currentProbability = currentConceptOrRelation.predict(self, (currentCandidate, ))
                    if currentProbability:
                        graphResultsForPhraseToken[currentConceptOrRelation.name][currentCandidate.instanceID] = currentProbability
                    
                elif len(currentCandidate) == 2: # currentConceptOrRelation is a pair thus candidates tuple has two element
                    currentCandidate1 = currentCandidate[0]
                    currentCandidate2 = currentCandidate[1]
                    currentProbability = currentConceptOrRelation.predict(self, (currentCandidate1, currentCandidate2))
                    if currentProbability:
                        graphResultsForPhraseRelation[currentConceptOrRelation.name][currentCandidate1.instanceID][currentCandidate2.instanceID] = currentProbability

                elif len(currentCandidate) == 3: # currentConceptOrRelation is a triple thus candidates tuple has three element     
                    currentCandidate1 = currentCandidate[0]
                    currentCandidate2 = currentCandidate[1]
                    currentCandidate3 = currentCandidate[2]
                    currentProbability = currentConceptOrRelation.predict(self, (currentCandidate1, currentCandidate2, currentCandidate3))
                    
                    self.myLogger.debug("currentConceptOrRelation is %s for relation %s and tokens %s %s %s - no variable created"%(currentConceptOrRelation,currentCandidate1,currentCandidate2,currentCandidate3,currentProbability))

                    if currentProbability:
                        graphResultsForTripleRelations[currentConceptOrRelation.name] \
                            [currentCandidate1.instanceID][currentCandidate2.instanceID][currentCandidate3.instanceID]= currentProbability
                    
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
        
        for concept_name in tokenResult:
            concept = conceptOrRelationDict[concept_name]
            currentCandidates = candidates_currentConceptOrRelation[concept]
            for infer_candidate in currentCandidates:
                if DataNode.PredictionType["ILP"] not in infer_candidate[0].predictions:
                     infer_candidate[0].predictions[DataNode.PredictionType["ILP"]] = {}
                     
                infer_candidate[0].predictions[DataNode.PredictionType["ILP"]][concept] = \
                    tokenResult[concept_name][infer_candidate[0].instanceID]
                
        for concept_name in pairResult:
            concept = conceptOrRelationDict[concept_name]
            currentCandidates = candidates_currentConceptOrRelation[concept]
            for infer_candidate in currentCandidates:
                if infer_candidate[0] != infer_candidate[1]:
                    if DataNode.PredictionType["ILP"] not in infer_candidate[0].predictions:
                        infer_candidate[0].predictions[DataNode.PredictionType["ILP"]] = {}
                     
                    infer_candidate[0].predictions[DataNode.PredictionType["ILP"]][concept, (infer_candidate[1])] = \
                        pairResult[concept_name][infer_candidate[0].instanceID, infer_candidate[1].instanceID]
                        
        for concept_name in tripleResult:
            concept = conceptOrRelationDict[concept_name]
            currentCandidates = candidates_currentConceptOrRelation[concept]
            for infer_candidate in currentCandidates:
                if infer_candidate[0] != infer_candidate[1] and infer_candidate[0] != infer_candidate[2] and infer_candidate[1] != infer_candidate[2]:
                    if DataNode.PredictionType["ILP"] not in infer_candidate[0].predictions:
                        infer_candidate[0].predictions[DataNode.PredictionType["ILP"]] = {}
                        
                    infer_candidate[0].predictions[DataNode.PredictionType["ILP"]][concept, (infer_candidate[1], infer_candidate[2])] = \
                        tripleResult[concept_name][infer_candidate[0].instanceID, infer_candidate[1].instanceID, infer_candidate[2].instanceID]

class DataNodeBuilder(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __changToTuple(self, v):
        if type(v) is (list or tuple):
            _v = []
            for v1 in v:
                _v.append(self.__changToTuple(v1))
                
            return tuple(_v)
        else:
            return v
        
    def __findDatanodes(self, dns, concept):
        if dns == None:
            return []
        
        if len(dns) == 0:
            return []

        if dns[0].ontologyNode == concept:
            return dns
        
        if dns[0].childInstanceNodes == None:
            return []
         
        if concept in dns[0].childInstanceNodes:
            returnDns = []

            for dn in dns:
                returnDns = returnDns + dn.childInstanceNodes[concept]
                
            return returnDns
         
        returnDns = []
        for dn in dns:
            for _concept in dn.childInstanceNodes:
                returnDns = returnDns + self.__findDatanodes(dn.childInstanceNodes[_concept], concept)

        return returnDns

    def __addSetitemCounter(self):
        globalCounterName = 'Counter' + '_setitem'
        if not dict.__contains__(self, globalCounterName):
            dict.__setitem__(self, globalCounterName, 1)
        else:
            currentCounter =  dict.__getitem__(self, globalCounterName)
            dict.__setitem__(self, globalCounterName, currentCounter + 1)
            
    def __addSensorCounters(self, skey, value):
        _value = value
        if isinstance(value, list):
            _value = self.__changToTuple(_value)
            
        counterNanme = 'Counter'
        for s in skey[2:]:
            counterNanme = counterNanme + '/' + s
            
        if not dict.__contains__(self, counterNanme):
            dict.__setitem__(self, counterNanme, {_value : 1})
        else:
            currentCounter =  dict.__getitem__(self, counterNanme)
            
            if _value in currentCounter:
                currentCounter[_value] = currentCounter[_value] + 1 
            else:
                currentCounter[_value] = 1
            
    def __findConcept(self, _conceptName, usedGraph):
        subGraph_keys = [key for key in usedGraph._objs]
        for subGraphKey in subGraph_keys:
            subGraph = usedGraph._objs[subGraphKey]
            
            for conceptNameItem in subGraph.concepts:
                if _conceptName == conceptNameItem:
                    concept = subGraph.concepts[conceptNameItem]
                    
                    return concept
        
        return None 
            
    def __setitem__(self, key, value):
        self.__addSetitemCounter()
       
        usedGraph = dict.__getitem__(self, "graph")
        global_key = usedGraph.name
                  
        skey = key.split('/')
        
        if len(skey) < 3:
            return dict.__setitem__(self, key, value)
        
        _conceptName = skey[2]
        concept = self.__findConcept(_conceptName, usedGraph)
        
        keyDataName = "".join(map(lambda x: '/' + x, skey[2:]))
        keyDataName = keyDataName[1:]

        if not concept:
            return dict.__setitem__(self, key, value)
        
        self.__addSensorCounters(skey, value)
            
        if value == None:
            return dict.__setitem__(self, key, value)
        has = concept.has_a()
        
        relation = False
        relationAttrs = []
        for arg_id, rel in enumerate(concept.has_a()): 
            relation = True
            relationName = rel.src.name
            conceptName = rel.dst.name
                            
            conceptAttr = self.__findConcept(conceptName, usedGraph)

            relationAttrs.append(conceptAttr)
            
        root = False
        # Check if the concept is root concept 
        if ('contains' not in concept._in):
            root = True
          
        contains = []  
        # Check if concept contains other concepts
        if ('contains' in concept._out):
            for _contain in concept._out['contains']:
                _containName = _contain.name
                _containNameSplit = _containName.split('-')
                
                containsConcept = self.__findConcept(_containNameSplit[3], usedGraph)
                
                if containsConcept:
                    contains.append(containsConcept)
                
        containedIn = []  
        # Check if concept is contained other concepts
        if ('contains' in concept._in):
            for _contain in concept._in['contains']:
                _containName = _contain.name
                _containNameSplit = _containName.split('-')
                
                containedConcept = self.__findConcept(_containNameSplit[0], usedGraph)
                
                if containedConcept:
                    containedIn.append(containedConcept)
        if relation:
            if (len(skey) == 4) and (skey[3] == "index"):# Check if this is relation index data 
                index = value
                dict.__setitem__(self, concept.name + "Index", index)
                
                existingRootDns = dict.__getitem__(self, 'dataNode') # Datanodes roots
                existingDnsForAttr = self.__findDatanodes(existingRootDns, relationAttrs[0]) # Datanodes of the relations attributes

                if dict.__contains__(self, concept.name + "Cache"): # Check if cache for relation has been already added
                    for _key in dict.__getitem__(self, concept.name + "Cache"):
                        _value = dict.__getitem__(self, _key)
                        
                        _skey = _key.split('/')
                        _keyDataName = "".join(map(lambda x: '/' + x, _skey[2:]))
                        _keyDataName = _keyDataName[1:]

                        for vIndex, v in enumerate(_value):
                            if len(_value) == len(index):
                                _i = index[vIndex]
                                existingDnsForAttr[_i[0]].predictions[(_keyDataName, _i[1])] = v
                            elif len(_value) == len(existingDnsForAttr):
                                existingDnsForAttr[vIndex].predictions[(_keyDataName, )] = v
                            else:
                                pass # ????
            elif dict.__contains__(self, concept.name + "Index"): # Check if index has been already added
                index = dict.__getitem__(self, concept.name + "Index")
                
                existingRootDns = dict.__getitem__(self, 'dataNode') # Datanodes roots
                existingDnsForAttr = self.__findDatanodes(existingRootDns, relationAttrs[0]) # Datanodes of the relations attributes
                
                for vIndex, v in enumerate(value):
                    if len(value) == len(index):
                        _i = index[vIndex]
                        existingDnsForAttr[_i[0]].predictions[(keyDataName, _i[1])] = v
                    elif len(value) == len(existingDnsForAttr):
                        existingDnsForAttr[vIndex].predictions[(keyDataName, )] = v
                    else:
                        pass # ????
            else:
                if dict.__contains__(self, concept.name + "Cache"): # Check if cache has been already added
                    cache = dict.__getitem__(self, concept.name + "Cache")
                    cache.append(key)
                else:
                    cache = [key]
                    dict.__setitem__(self, concept.name + "Cache", cache)
        else:                       
            if not dict.__contains__(self, 'dataNode'): # No datanode yet
                dns = []
                if type(value) is not Tensor: # value is not Tensor
                    instanceValue = ""
                    instanceID = dict.__getitem__(self, "READER")
                    _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = concept, childInstanceNodes = {})
                    
                    _dn.predictions[(keyDataName, )] = value
                    
                    dns.append(_dn)
                else: # value is Tensor
                    for vIndex, v in enumerate(value):
                        instanceValue = ""
                        instanceID = vIndex
                        _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = concept, childInstanceNodes = {})
                        
                        _dn.predictions[(keyDataName, )] = v
                        
                        dns.append(_dn)
                            
                dict.__setitem__(self, 'dataNode', dns)
            else: # Datanodes already created
                existingRootDns = dict.__getitem__(self, 'dataNode') # Datanodes roots
                existingDnsForConcept = self.__findDatanodes(existingRootDns, concept) # Datanodes of the current concept
                l = len(existingDnsForConcept)
                dns = []
                if len(existingDnsForConcept) == 0: # No Datanode of this concept created yet
                    if (type(value) is not Tensor) and (type(value) is not list) : # value is not Tensor or list
                        instanceValue = ""
                        instanceID = dict.__getitem__(self, "READER")
                        _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = concept, childInstanceNodes = {})
                        
                        _dn.predictions[(keyDataName, )] = value
                        
                        if root and len(contains) > 0:
                            for _contains in contains:
                                _existingDnsForConcept = self.__findDatanodes(existingRootDns, _contains) # Datanodes of the current concept
                                
                                if len(_existingDnsForConcept) > 0:
                                    _dn.childInstanceNodes[_contains] = _existingDnsForConcept
                                 
                            dns.append(_dn)   
                            dict.__setitem__(self, 'dataNode', dns) # New root
                        else:
                            pass # ????
    
                    else: # value is Tensor
                        for vIndex, v in enumerate(value):
                            instanceValue = ""
                            instanceID = vIndex
                            _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = concept, childInstanceNodes = {})
                            
                            _dn.predictions[(keyDataName, )] = v
                            
                            dns.append(_dn)
                        
                        if len(containedIn) > 0:
                            myContainedIn = self.__findDatanodes(existingRootDns, containedIn[0])
                            
                            if len(myContainedIn) > 0:
                                myContainedIn[0].childInstanceNodes[concept] = dns
                            
                            
                else: # Datanode with this concept already created
                    if (type(value) is not Tensor) and (type(value) is not list): # value is not Tensor or list
                        if type(existingDnsForConcept[0]) is DataNode:
                            existingDnsForConcept[0].predictions[(keyDataName, )] = value
                    else: # value is Tensor
                        for vIndex, v in enumerate(value):
                            if vIndex >= len(existingDnsForConcept):
                                break # ?????
                            
                            if type(existingDnsForConcept[vIndex]) is DataNode:
                                existingDnsForConcept[vIndex].predictions[(keyDataName, )] = v

        return dict.__setitem__(self, key, value)

# ----------------------------------------------
        # Check if this is the data defining datanote for the given concept
        if (skey[3] == 'raw' or skey[3] == 'raw_ready'): # Raw-ready - fix for sentence/word
            if not dict.__contains__(self, 'dataNode'):
                
                instanceValue = value
                instanceID = dict.__getitem__(self, "READER")
                dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = concept, childInstanceNodes = {})
                dict.__setitem__(self, 'dataNode', dn)
            else:
                dns= dict.__getitem__(self, 'dataNode')
                
                # Check if datanode exist for concept
                existingDNs = self.__findDatanodes(dns, concept)
                
                if (existingDNs == None) or (len(existingDNs) == 0):
                    if (skey[3] == 'raw_ready') and (len(skey) < 5) : # fix for concept word  - sentence/raw_ready ?
                        wordConcept = dict.__getitem__(self, "graph")['linguistic/word']
        
                        if wordConcept not in dn.childInstanceNodes:                          
                            dns = []
                            for word in value:
                                dnNew = DataNode(instanceID = word.idx, instanceValue = word.text, ontologyNode = wordConcept, childInstanceNodes = {})
                                dns.append(dnNew)
                
                            dn.childInstanceNodes[wordConcept] = dns
                    else:
                        # Check if the concept is root concept 
                        
                        # Check if concept contains other concepts
                        
                        wordConcept = dict.__getitem__(self, "graph")['linguistic/word']
        
                        if wordConcept in dn.childInstanceNodes:  
                            phraseConcept = dict.__getitem__(self, "graph")['linguistic/phrase']
                            
                            if phraseConcept not in dn.childInstanceNodes:
                                dns = []
                                for phraseId, phrase in enumerate(value):
                                    phraseText = None
                                    childInstanceNodes = []
                                    
                                    for wordId in range(phrase[0], phrase[1]+1):
                                        if phraseText:
                                            phraseText = phraseText + " " + dn.childInstanceNodes[wordConcept][wordId].instanceValue
                                        else:
                                            phraseText = dn.childInstanceNodes[wordConcept][wordId].instanceValue
                                            
                                        childInstanceNodes.append(dn.childInstanceNodes[wordConcept][wordId])
                        
                                    dnNew = DataNode(instanceID = phraseId, instanceValue = phraseText, ontologyNode = phraseConcept, childInstanceNodes = {wordConcept : childInstanceNodes})
                                    dns.append(dnNew)
                                
                                dn.childInstanceNodes[phraseConcept] = dns
                
        epsilon = 0.00001
        # Check if this is prediction data   
        if dict.__contains__(self, 'dataNode'):
            dataNode = dict.__getitem__(self, 'dataNode') 
            
            pattern = "<(.*?)\>"
            match = re.search(pattern, key)
            
            if match: 
                app_concept = match.group(1)
        
                if isinstance(value, torch.Tensor):
                    if (len(value.shape) > 1):
                        with torch.no_grad():
                            _list = [_it.cpu().detach().numpy() for _it in value]
                            
                            for _it in range(len(_list)):
                                if _list[_it][0] > 1-epsilon:
                                    _list[_it][0] = 1-epsilon
                                elif _list[_it][1] > 1-epsilon:
                                    _list[_it][1] = 1-epsilon
                                if _list[_it][0] < epsilon:
                                    _list[_it][0] = epsilon
                                elif _list[_it][1] < epsilon:
                                    _list[_it][1] = epsilon
                                    
                            _list = [np.log(_it) for _it in _list]
                            
                            wordConcept = dict.__getitem__(self, "graph")['linguistic/word']
                            if wordConcept in dataNode.childInstanceNodes:  
                                for dnChildIndex, dnChild in enumerate(dataNode.childInstanceNodes[wordConcept]):
                                    if DataNode.PredictionType["Learned"] in dnChild.predictions:
                                        dnChild.predictions[DataNode.PredictionType["Learned"]][(concept,)] = _list[dnChildIndex]
                                    else:
                                        dnChild.predictions[DataNode.PredictionType["Learned"]] = {(concept,) : _list[dnChildIndex]}
                
        pair_prediction_key = "pair"
        if (pair_prediction_key in key) and (value is not None):
            pattern = "<(.*?)\>"
            match = re.search(pattern, key)
            
            if match: 
                pair = match.group(1)

                if isinstance(value, torch.Tensor):
                    if (len(value.shape) > 1):
                        with torch.no_grad():
                            _list = [np.log(_it.cpu().detach().numpy()) for _it in value]
                            
                            if dict.__contains__(self, 'dataNode'):
                                dataNode = dict.__getitem__(self, 'dataNode')
                                
                            phraseConcept = dict.__getitem__(self, "graph")['linguistic/phrase']
                            if phraseConcept in dataNode.childInstanceNodes:
                                noPhrases = len(dataNode.childInstanceNodes[phraseConcept])
                                _result = np.zeros((noPhrases, noPhrases, 2))
                                
                                pairsIndexKey = global_key + "/linguistic/" + "pair/index"
                                if dict.__contains__(self, pairsIndexKey):
                                    pairsIndex = dict.__getitem__(self, pairsIndexKey)
                                    
                                    for _range in range(len(pairsIndex)):
                                        indexes = pairsIndex[_range]
                                        values = _list[_range]
                                        
                                        _result[indexes[0]][indexes[1]][0] = values[0]
                                        _result[indexes[1]][indexes[0]][0] = values[0]
                                        _result[indexes[0]][indexes[1]][1] = values[1]
                                        _result[indexes[1]][indexes[0]][1] = values[1]
                                        
                                    for _ii in range(noPhrases):
                                        _result[_ii][_ii] = np.log(np.array([0.999, 0.001]))
                                        
                                    for dnChildIndex, dnChild in enumerate(dataNode.childInstanceNodes[phraseConcept]):
                                        for _ii in range(noPhrases):
                                            if DataNode.PredictionType["Learned"] in dnChild.predictions:
                                                dnChild.predictions[DataNode.PredictionType["Learned"]][(pair, dataNode.childInstanceNodes[phraseConcept][dnChildIndex])] = _result[dnChildIndex][_ii]
                                            else:
                                                dnChild.predictions[DataNode.PredictionType["Learned"]] = {(pair, dataNode.childInstanceNodes[phraseConcept][dnChildIndex]) : _result[dnChildIndex][_ii]}  
                                                      
    def __delitem__(self, key):
        return dict.__delitem__(self, key)

    def __contains__(self, key):
        return dict.__contains__(self, key)
    
    def getDataNode(self):
        if dict.__contains__(self, 'dataNode'):
            dataNode = dict.__getitem__(self, 'dataNode')
            
            return dataNode
        
        return None

# ----------- To be Remove - for reference only  now  
def processContext(context, concepts, relations):
    global_key = "global/linguistic/"
        
    sentence = {"words": {}}
    
    with torch.no_grad():
        epsilon = 0.00001
        predictions_on = "word"
        for _concept in concepts:
            concept = str(_concept.name.name)
            
            _list = [_it.cpu().detach().numpy() for _it in context[global_key + predictions_on + "/" + "<"+ concept +">"]]
            
            for _it in range(len(_list)):
                if _list[_it][0] > 1-epsilon:
                    _list[_it][0] = 1-epsilon
                elif _list[_it][1] > 1-epsilon:
                    _list[_it][1] = 1-epsilon
                if _list[_it][0] < epsilon:
                    _list[_it][0] = epsilon
                elif _list[_it][1] < epsilon:
                    _list[_it][1] = epsilon
                    
            sentence[concept] = _list
            sentence['words'][concept] = [np.log(_it) for _it in _list]

        if len(relations):
            sentence['phrase'] = {}
            
            sentence['phrase']['raw'] = context[global_key + "phrase/raw"]
            sentence['phrase']['tag'] = [_it.item() for _it in context[global_key + "phrase/tag"]]
            sentence['phrase']['tag_name'] = [concepts[t].name.name for t in sentence['phrase']['tag']]
            sentence['phrase']['pair_index'] = context[global_key + "pair/index"]
            
            sentence['phrase']['entity'] = {}
            for _item in concepts:
                item = str(_item.name.name)

                _list = []
                
                for ph in sentence['phrase']['raw']:
                    _value = [1, 1]
                    for _range in range(ph[0], ph[1] + 1):
                        _value[0] = _value[0] * sentence[item][_range][0]
                        _value[1] = _value[1] * sentence[item][_range][1]
                        
                    _list.append(np.log(np.array(_value)))
                    
                sentence['phrase']['entity'][item] = [_val for _val in _list]
                
            sentence['phrase']['relations'] = {}
            pairs_on = "pair"
            for _item in relations:
                item = str(_item.name.name)

                _list = [np.log(_it.cpu().detach().numpy()) for _it in context[global_key + pairs_on + "/" + "<"+ item +">"]]
                _result = np.zeros((len(sentence['phrase']['raw']), len(sentence['phrase']['raw']), 2))
                
                for _range in range(len(sentence['phrase']['pair_index'])):
                    indexes = sentence['phrase']['pair_index'][_range]
                    values = _list[_range]
                    _result[indexes[0]][indexes[1]][0] = values[0]
                    _result[indexes[1]][indexes[0]][0] = values[0]
                    _result[indexes[0]][indexes[1]][1] = values[1]
                    _result[indexes[1]][indexes[0]][1] = values[1]
                    
                for _ii in range(len(sentence['phrase']['raw'])):
                    _result[_ii][_ii] = np.log(np.array([0.999, 0.001]))
                    
                sentence['phrase']['relations'][item] = _result
  
    if len(relations):
        phrases = [str(_it) for _it in range(len(sentence['phrase']['raw']))]
        graphResultsForPhraseToken = sentence['phrase']['entity']
        graphResultsForPhraseRelation = sentence['phrase']['relations']
        
    else:
        tokens = [str(_it) for _it in range(len(sentence['FAC']))]
        graphResultsForPhraseToken = sentence['words']
