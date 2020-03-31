import numpy as np
import torch
from collections import OrderedDict 
import logging
import re
import time
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
   
    def __init__(self, instanceID = None, instanceValue = None, ontologyNode = None, childInstanceNodes = {}, relationLinks = {}, attributes = {}):
        self.myLogger = logging.getLogger(ilpConfig['log_name'])

        self.instanceID = instanceID                     # The data instance id (e.g. paragraph number, sentence number, phrase  number, image number, etc.)
        self.instanceValue = instanceValue               # Optional value of the instance (e.g. paragraph text, sentence text, phrase text, image bitmap, etc.)
        self.ontologyNode = ontologyNode                 # Reference to the ontology graph node (e.g. Concept) which is the type of this instance (e.g. paragraph, sentence, phrase, etc.)
        if childInstanceNodes:
            self.childInstanceNodes = childInstanceNodes # Dictionary mapping child ontology type to List of child data nodes this instance was segmented into based on this type
        else:
             self.childInstanceNodes = {}
        if relationLinks:
            self.relationLinks = relationLinks           # Dictionary mapping relation name to the object representing the relation
        else:
            self.relationLinks = {}
        if attributes:
            self.attributes = attributes                 # Dictionary with additional node's attributes
        else:
             self.attributes = {}

    def __str__(self):
        if self.instanceValue:
            return self.instanceValue
        else:
            return 'instanceID {}'.format(self.instanceID)
        
    def getInstanceID(self):
        return self.instanceID
    
    def getInstanceValue(self):
        return self.instanceValue
    
    def getOntologyNode(self):
        return self.ontologyNode
    
    def getAttributes(self):
        return self.attributes.keys()
    
    def getAttribute(self, key):
        if key in self.attributes:
            return self.attributes[key]
        else:
            return None
            
    def getChildInstanceNodes(self):
        return self.childInstanceNodes
    
    # Find concept in the graph based on concept name
    def __findConcept(self, _conceptName, usedGraph = None):
        if not usedGraph:
            usedGraph = self.ontologyNode.getOntologyGraph()
            
        subGraph_keys = [key for key in usedGraph._objs]
        for subGraphKey in subGraph_keys:
            subGraph = usedGraph._objs[subGraphKey]
            
            for conceptNameItem in subGraph.concepts:
                if _conceptName == conceptNameItem:
                    concept = subGraph.concepts[conceptNameItem]
                    
                    return concept
        
        return None 
    
    def __isRelation(self, conceptRelation, usedGraph = None):
        if not usedGraph:
            usedGraph = self.ontologyNode.getOntologyGraph()
            
        if len(conceptRelation.has_a()) > 0:  
            return True
        
        for _isA in conceptRelation.is_a():
            _conceptRelation = _isA.dst
            
            if self.__isRelation(_conceptRelation, usedGraph):
                return True
        
        return False 

    def __getProbability(self, dataNode, conceptRelation, fun=None, epsilon = 0.00001):
        pass
        
    # Calculate ILP prediction for this instance based on the provided list of concepts (or/and relations) 
    def inferILPConstrains(self, fun=None, *_conceptsRelations):
        
        _instances = set() # Set of all the candidates across all the concepts to be consider in the ILP constrains
        candidates_currentConceptOrRelation = OrderedDict()
        conceptsRelations = []
        # Collect all the candidates for concepts and relations in conceptsRelations
        for _currentConceptOrRelation in _conceptsRelations:
            
            # Convert string to concept it provided as string
            if isinstance(_currentConceptOrRelation, str):
                currentConceptOrRelation = self.__findConcept(_currentConceptOrRelation)
                
                if currentConceptOrRelation is None:
                    continue
            else:
                currentConceptOrRelation = _currentConceptOrRelation
            
            conceptsRelations.append(currentConceptOrRelation)
                
            # Get candidates (dataNodes or relation links for the concept) from the graph starting from the current data node
            currentCandidates = currentConceptOrRelation.candidates(self)
            if currentCandidates is None:
                continue
            
            if self.__isRelation(currentConceptOrRelation): # Check if relation
                pass
            
            _currentInstances = set()
            for currentCandidate in currentCandidates:
                _currentInstances.add(currentCandidate)
                
                for candidateElement in currentCandidate:
                    _instances.add(candidateElement)

            candidates_currentConceptOrRelation[currentConceptOrRelation] = _currentInstances
          
        if len(_instances) == 0:
            return 
        
        # Get ids of the instances
        infer_candidatesID = []
        for currentCandidate in _instances:
            infer_candidatesID.append(currentCandidate.instanceID)
        
        infer_candidatesID.sort() # Sort the list of instances

        no_candidateds = len(infer_candidatesID)    # Number of candidates
        
        # Create numpy arrays for collected probabilities and zero them
        graphResultsForPhraseToken = dict()
        
        # Make this generic for any number of relation attributes
        graphResultsForPhraseRelation = dict()
        graphResultsForTripleRelations = dict()

        conceptOrRelationDict = {} # maps concept name to concept - Used to put the ILP calculated probabilities back into the data graph 
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
                
        # Collect probabilities for candidates 
        for currentConceptOrRelation in conceptsRelations:
            currentCandidates = candidates_currentConceptOrRelation[currentConceptOrRelation]
            
            if currentCandidates is None:
                continue
            
            for currentCandidate in currentCandidates:
                if len(currentCandidate) == 1:   # currentConceptOrRelation is a concept thus candidates tuple has only single element
                    currentCandidate = currentCandidate[0]
                    currentProbability = __getProbability(currentCandidate, currentConceptOrRelation, fun=fun, epsilon=epsilon)
                    #currentProbability = currentConceptOrRelation.predict(self, (currentCandidate, ))
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

        # Get ontology graph and then  ilpOntsolver
        myOntologyGraphs = {self.ontologyNode.getOntologyGraph()}
        
        for currentConceptOrRelation in conceptsRelations:
            currentOntologyGraph = currentConceptOrRelation.getOntologyGraph()
            
            if currentOntologyGraph is not None:
                myOntologyGraphs.add(currentOntologyGraph)
                
        myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(myOntologyGraphs)
        
        # Call ilpOntsolver with the collected probabilities for chosen candidates
        tokenResult, pairResult, tripleResult = myilpOntSolver.calculateILPSelection(infer_candidatesID, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForTripleRelations)
        
        return tokenResult, pairResult, tripleResult
    
        # Update this to the Relation Link
        for concept_name in tokenResult:
            concept = conceptOrRelationDict[concept_name]
            currentCandidates = candidates_currentConceptOrRelation[concept]
            for infer_candidate in currentCandidates:
                if DataNode.PredictionType["ILP"] not in infer_candidate[0].attributes:
                     infer_candidate[0].attributes[DataNode.PredictionType["ILP"]] = {}
                     
                infer_candidate[0].attributes[DataNode.PredictionType["ILP"]][concept] = \
                    tokenResult[concept_name][infer_candidate[0].instanceID]
                
        for concept_name in pairResult:
            concept = conceptOrRelationDict[concept_name]
            currentCandidates = candidates_currentConceptOrRelation[concept]
            for infer_candidate in currentCandidates:
                if infer_candidate[0] != infer_candidate[1]:
                    if DataNode.PredictionType["ILP"] not in infer_candidate[0].attributes:
                        infer_candidate[0].attributes[DataNode.PredictionType["ILP"]] = {}
                     
                    infer_candidate[0].attributes[DataNode.PredictionType["ILP"]][concept, (infer_candidate[1])] = \
                        pairResult[concept_name][infer_candidate[0].instanceID, infer_candidate[1].instanceID]
                        
        for concept_name in tripleResult:
            concept = conceptOrRelationDict[concept_name]
            currentCandidates = candidates_currentConceptOrRelation[concept]
            for infer_candidate in currentCandidates:
                if infer_candidate[0] != infer_candidate[1] and infer_candidate[0] != infer_candidate[2] and infer_candidate[1] != infer_candidate[2]:
                    if DataNode.PredictionType["ILP"] not in infer_candidate[0].attributes:
                        infer_candidate[0].attributes[DataNode.PredictionType["ILP"]] = {}
                        
                    infer_candidate[0].attributes[DataNode.PredictionType["ILP"]][concept, (infer_candidate[1], infer_candidate[2])] = \
                        tripleResult[concept_name][infer_candidate[0].instanceID, infer_candidate[1].instanceID, infer_candidate[2].instanceID]
                
        return tokenResult, pairResult, tripleResult

class RelationLink:
   
    def __init__(self, instanceID = None, ontologyNode = None, relationDataNodes = {}, attributes = {}):
        self.myLogger = logging.getLogger(ilpConfig['log_name'])

        self.instanceID = instanceID                     # The relation instance id

        self.ontologyNode = ontologyNode                 # Reference to the ontology graph node (e.g. Relation) which is the type of this instance (e.g. pair, etc.)
        if relationDataNodes:
            self.relationDataNodes = relationDataNodes   # Dictionary mapping data node id to datanode
        else:
             self.childInstanceNodes = {}
        if attributes:
            self.attributes = attributes                 # Dictionary with additional node's attributes
        else:
             self.attributes = {}

class DataNodeBuilder(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    # Change elements of value to tuple if they are list - in order to use the value as dictionary key
    def __changToTuple(self, v):
        if type(v) is list:
            _v = []
            for v1 in v:
                _v.append(self.__changToTuple(v1))
                
            return tuple(_v)
        else:
            return v
        
    # Find datanode in data graph of the given concept 
    def __findDatanodes(self, dns, concept):
        if dns is None:
            return []
        
        if len(dns) == 0:
            return []

        if dns[0].ontologyNode == concept:
            return dns
        
        if dns[0].childInstanceNodes is None:
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

    # Add or increase generic counter counting number of setitem calls
    def __addSetitemCounter(self):
        globalCounterName = 'Counter' + '_setitem'
        if not dict.__contains__(self, globalCounterName):
            dict.__setitem__(self, globalCounterName, 1)
        else:
            currentCounter =  dict.__getitem__(self, globalCounterName)
            dict.__setitem__(self, globalCounterName, currentCounter + 1)
            
    # Add or increase sensor counter counting number of setitem calls with the given sensor key
    def __addSensorCounters(self, skey, value):
        _value = value
        if isinstance(value, list):
            _value = self.__changToTuple(_value)
            
        counterNanme = 'Counter'
        for s in skey[2:]:
            counterNanme = counterNanme + '/' + s
            
        if not dict.__contains__(self, counterNanme):
            try:
               dict.__setitem__(self, counterNanme, {_value : {"counter": 1, "recent" : True}})
            except TypeError:
                t = type(_value)
                return False
            
            return False
        else:
            currentCounter =  dict.__getitem__(self, counterNanme)
            
            if _value in currentCounter:
                currentCounter[_value]["counter"] = currentCounter[_value]["counter"] + 1 
                
                if currentCounter[_value]["recent"]:
                    return True
                else:
                    currentCounter[_value]["recent"] = True
                    return False
            else:
                currentCounter[_value]  = {"counter": 1, "recent" : True}
                
                return False
            
    # Find concept in the graph based on concept name
    def __findConcept(self, _conceptName, usedGraph):
        subGraph_keys = [key for key in usedGraph._objs]
        for subGraphKey in subGraph_keys:
            subGraph = usedGraph._objs[subGraphKey]
            
            for conceptNameItem in subGraph.concepts:
                if _conceptName == conceptNameItem:
                    concept = subGraph.concepts[conceptNameItem]
                    
                    return concept
        
        return None 
             
    # Collect concept information defined in the graph
    def __findConceptInfo(self, usedGraph, concept):
        conceptInfo = {}
        
        conceptInfo['concept'] = concept
        
        conceptInfo['relation'] = False
        conceptInfo['relationAttrs'] = []
        for arg_id, rel in enumerate(concept.has_a()): 
            conceptInfo['relation'] = True
            relationName = rel.src.name
            conceptName = rel.dst.name
                            
            conceptAttr = self.__findConcept(conceptName, usedGraph)

            conceptInfo['relationAttrs'].append(conceptAttr)
            
        conceptInfo['root'] = False
        # Check if the concept is root concept 
        if ('contains' not in concept._in):
            conceptInfo['root'] = True
          
        conceptInfo['contains'] = []  
        # Check if concept contains other concepts
        if ('contains' in concept._out):
            for _contain in concept._out['contains']:
                _containName = _contain.name
                _containNameSplit = _containName.split('-')
                
                containsConcept = self.__findConcept(_containNameSplit[3], usedGraph)
                
                if containsConcept:
                    conceptInfo['contains'].append(containsConcept)
                
        conceptInfo['containedIn'] = []  
        # Check if concept is contained other concepts
        if ('contains' in concept._in):
            for _contain in concept._in['contains']:
                _containName = _contain.name
                _containNameSplit = _containName.split('-')
                
                containedConcept = self.__findConcept(_containNameSplit[0], usedGraph)
                
                if containedConcept:
                    conceptInfo['containedIn'].append(containedConcept)
        
        return conceptInfo
    
    # Process relation data cache
    def __processRelationDataCache(self, conceptInfo, value):
        if not dict.__contains__(self, conceptInfo['concept'].name + "Cache"): 
            return

        index = dict.__getitem__(self, conceptInfo['concept'].name + "Index")
        if not index:
            return

         # Check data graph for data nodes connected by this relation
        existingRootDns = dict.__getitem__(self, 'dataNode') # Datanodes roots
        
        if not existingRootDns:
            return
        
        existingDnsForAttr = []
        for attr in conceptInfo['relationAttrs']:
            _existingDnsForAttr = self.__findDatanodes(existingRootDns, conceptInfo['relationAttrs'][0]) # Datanodes of the relations attributes
            
            if not _existingDnsForAttr:
                return
            
            existingDnsForAttr.append(_existingDnsForAttr)
            
        for _key in dict.__getitem__(self, conceptInfo['concept'].name + "Cache"):
            
            if not dict.__contains__(self, _key):
                _value = value
            else:
                _value = dict.__getitem__(self, _key)
            
            _skey = _key.split('/')
            _keyDataName = "".join(map(lambda x: '/' + x, _skey[3:]))
            _keyDataName = _keyDataName[1:]

            if len(_value) == len(index):
                for vIndex, v in enumerate(_value):
                    
                    # Create of update Relation link !!
                    _i = index[vIndex]
                    
                    relationDataNodes = {}
                    for i in range(len(existingDnsForAttr)):
                        relationDataNodes[existingDnsForAttr[i][_i[i]].instanceID] = existingDnsForAttr[i][_i[i]]
                        
                    relationDataNodesIdsTuple = tuple(relationDataNodes.keys())
                    rl = None
                    
                    # Check if relation link exit for this relation
                    for i, dn in relationDataNodes.items():
                        if conceptInfo['concept'].name not in dn.relationLinks:
                            dn.relationLinks[conceptInfo['concept'].name] = {}
                            
                        if relationDataNodesIdsTuple not in dn.relationLinks[conceptInfo['concept'].name]:
                            if not rl:
                                # Add relation link
                                rlInstanceID = conceptInfo['concept'].name + '_' + str(tuple(relationDataNodes.keys()))
                                rl = RelationLink(instanceID = rlInstanceID, ontologyNode = conceptInfo['concept'], relationDataNodes = relationDataNodes, attributes = {})
                            
                            dn.relationLinks[conceptInfo['concept'].name][tuple(relationDataNodes.keys())] = rl
                        else:
                            rl = dn.relationLinks[conceptInfo['concept'].name][relationDataNodesIdsTuple]
                    
                     
                    t = rl is not None
                    
                    if t:
                        # Add sensor value to the relation link
                        rl.attributes[_keyDataName] = v
                    else:
                        pass # ????

            elif len(_value) == len(existingDnsForAttr): #Add value to datanode
                for vIndex, v in enumerate(_value):
                    existingDnsForAttr[0][vIndex].attributes[_keyDataName] = v
            else:
                pass # ????
                
        # Remove cache
        dict.__delitem__(self, conceptInfo['concept'].name + "Cache")
        
    # Build or update relation link in the data graph for a given key
    def __buildRelationLink(self, key, skey, value, conceptInfo):
        if (len(skey) == 4) and (skey[3] == "index"): # If this is relation index data 
            dict.__setitem__(self, conceptInfo['concept'].name + "Index", value)  # Keep the index data 
        else: # Cache key to data about relation
            if dict.__contains__(self, conceptInfo['concept'].name + "Cache"): # Check if cache has been already added
                cache = dict.__getitem__(self, conceptInfo['concept'].name + "Cache")
                cache.append(key)
            else:
                cache = [key]
                dict.__setitem__(self, conceptInfo['concept'].name + "Cache", cache)
                
        # Process relation data cache
        if dict.__contains__(self, conceptInfo['concept'].name + "Index"):
            self.__processRelationDataCache(conceptInfo, value)
               
    # Build or update data node in the data graph for a given key
    def __buildDataNode(self, value, conceptInfo, keyDataName):
        if not dict.__contains__(self, 'dataNode'): # No datanode yet
            dns = []
            if type(value) is not Tensor: # value is not Tensor
                instanceValue = ""
                instanceID = dict.__getitem__(self, "READER")
                _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'], childInstanceNodes = {})
                
                _dn.attributes[keyDataName] = value
                
                dns.append(_dn)
            else: # value is Tensor
                for vIndex, v in enumerate(value):
                    instanceValue = ""
                    instanceID = vIndex
                    _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'], childInstanceNodes = {})
                    
                    _dn.attributes[keyDataName] = v
                    
                    dns.append(_dn)
                        
            dict.__setitem__(self, 'dataNode', dns)
        else: # Datanodes already created
            existingRootDns = dict.__getitem__(self, 'dataNode') # Datanodes roots
            existingDnsForConcept = self.__findDatanodes(existingRootDns, conceptInfo['concept']) # Datanodes of the current concept
            l = len(existingDnsForConcept)
            dns = []
            if len(existingDnsForConcept) == 0: # No Datanode of this concept created yet
                if (type(value) is not Tensor) and (type(value) is not list) : # value is not Tensor or list
                    instanceValue = ""
                    instanceID = dict.__getitem__(self, "READER")
                    _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'], childInstanceNodes = {})
                    
                    _dn.attributes[keyDataName] = value
                    ''
                    if conceptInfo['root'] and len(conceptInfo['contains']) > 0:
                        for _contains in conceptInfo['contains']:
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
                        _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'], childInstanceNodes = {})
                        
                        _dn.attributes[keyDataName] = v
                        
                        dns.append(_dn)
                    
                    if len(conceptInfo['containedIn']) > 0:
                        myContainedIn = self.__findDatanodes(existingRootDns, conceptInfo['containedIn'][0])
                        
                        if len(myContainedIn) > 0:
                            myContainedIn[0].childInstanceNodes[conceptInfo['concept']] = dns
                        
            else: # Datanode with this concept already created
                if (type(value) is not Tensor) and (type(value) is not list): # value is not Tensor or list
                    if type(existingDnsForConcept[0]) is DataNode:
                        existingDnsForConcept[0].attributes[keyDataName] = value
                else: # value is Tensor
                    for vIndex, v in enumerate(value):
                        if vIndex >= len(existingDnsForConcept):
                            break # ?????
                        
                        if type(existingDnsForConcept[vIndex]) is DataNode:
                            existingDnsForConcept[vIndex].attributes[keyDataName] = v
 
    # Overloaded __setitem method of Dictionary - tracking sensor data and building corresponding data graph
    def __setitem__(self, key, value):
        start = time.time()
        self.__addSetitemCounter()
          
        skey = key.split('/')
        
        if self.__addSensorCounters(skey, value):
            return

        if value is None:
            return dict.__setitem__(self, key, value)
        
        if len(skey) < 3:
            return dict.__setitem__(self, key, value)
        
        usedGraph = dict.__getitem__(self, "graph")
        _conceptName = skey[2]
        concept = self.__findConcept(_conceptName, usedGraph)
        
        if not concept:
            return dict.__setitem__(self, key, value)
        
        conceptInfo = self.__findConceptInfo(usedGraph, concept)
        
        keyDataName = "".join(map(lambda x: '/' + x, skey[3:]))
        keyDataName = keyDataName[1:]

        if conceptInfo['relation']:
            self.__buildRelationLink(key, skey, value, conceptInfo) # Build or update relation link
        else:                       
            self.__buildDataNode(value, conceptInfo, keyDataName)   # Build or update Data node
            
        # Add value to the underling dictionary
        r = dict.__setitem__(self, key, value)
        
        # ------------------- Collect time used for __setitem__
        end = time.time()
        if dict.__contains__(self, "DataNodeTime"):
            currenTime = dict.__getitem__(self, "DataNodeTime")
            currenTime = currenTime + end - start
        else:
            currenTime =  end - start
        dict.__setitem__(self, "DataNodeTime", currenTime)
        # -------------------
        
        if not r:
            pass
        
        return r                
                                             
    def __delitem__(self, key):
        return dict.__delitem__(self, key)

    def __contains__(self, key):
        return dict.__contains__(self, key)
    
    # Method calculating Learned prediction based on data in datanode
    def __calculateLearnedPrediction(self, fun):
        epsilon = 0.00001
        # Check if this is prediction data   
        if dict.__contains__(self, 'dataNode'):
            dataNode = dict.__getitem__(self, 'dataNode')[0] 
            
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
                                    
                            _list = [fun(_it) for _it in _list]
                            
                            wordConcept = dict.__getitem__(self, "graph")['linguistic/word']
                            if wordConcept in dataNode.childInstanceNodes:  
                                for dnChildIndex, dnChild in enumerate(dataNode.childInstanceNodes[wordConcept]):
                                    if DataNode.PredictionType["Learned"] in dnChild.attributes:
                                        dnChild.attributes[DataNode.PredictionType["Learned"]][(concept,)] = _list[dnChildIndex]
                                    else:
                                        dnChild.attributes[DataNode.PredictionType["Learned"]] = {(concept,) : _list[dnChildIndex]}
                
        pair_prediction_key = "pair"
        if (pair_prediction_key in key) and (value is not None):
            pattern = "<(.*?)\>"
            match = re.search(pattern, key)
            
            if match: 
                pair = match.group(1)

                if isinstance(value, torch.Tensor):
                    if (len(value.shape) > 1):
                        with torch.no_grad():
                            _list = [fun(_it.cpu().detach().numpy()) for _it in value]
                            
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
                                        _result[_ii][_ii] = fun(np.array([0.999, 0.001]))
                                        
                                    for dnChildIndex, dnChild in enumerate(dataNode.childInstanceNodes[phraseConcept]):
                                        for _ii in range(noPhrases):
                                            if DataNode.PredictionType["Learned"] in dnChild.attributes:
                                                dnChild.attributes[DataNode.PredictionType["Learned"]][(pair, dataNode.childInstanceNodes[phraseConcept][dnChildIndex])] = _result[dnChildIndex][_ii]
                                            else:
                                                dnChild.attributes[DataNode.PredictionType["Learned"]] = {(pair, dataNode.childInstanceNodes[phraseConcept][dnChildIndex]) : _result[dnChildIndex][_ii]}  
        
    # Method returning datanode
    def getDataNode(self):
        if dict.__contains__(self, 'dataNode'):
            _dataNode = dict.__getitem__(self, 'dataNode')
            
            if len(_dataNode) > 0:                
                return _dataNode[0]
        
        return None
