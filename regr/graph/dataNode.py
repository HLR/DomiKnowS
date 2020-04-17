import numpy as np
import torch
from collections import OrderedDict 
import logging
import re
import time
from itertools import product
from regr.solver.ilpConfig import ilpConfig 
from torch.tensor import Tensor

if __package__ is None or __package__ == '':
    from graph import Graph
    from solver import ilpOntSolverFactory
else:
    from .graph import Graph
    from ..solver import ilpOntSolverFactory

# Class representing single data instance with relation  links to other data nodes
class DataNode:
   
    PredictionType = {"Learned" : "Learned", "ILP" : "ILP"}
   
    def __init__(self, instanceID = None, instanceValue = None, ontologyNode = None, relationLinks = {}, attributes = {}):
        self.myLogger = logging.getLogger(ilpConfig['log_name'])

        self.instanceID = instanceID                     # The data instance id (e.g. paragraph number, sentence number, phrase  number, image number, etc.)
        self.instanceValue = instanceValue               # Optional value of the instance (e.g. paragraph text, sentence text, phrase text, image bitmap, etc.)
        self.ontologyNode = ontologyNode                 # Reference to the node in the ontology graph (e.g. Concept) which is the type of this instance (e.g. paragraph, sentence, phrase, etc.)
        if relationLinks:
            self.relationLinks = relationLinks           # Dictionary mapping relation name to the RealtionLinks
        else:
            self.relationLinks = {}
            
        self.impactLinks = {}                            # Dictionary with dataNodes impacting this datanode by having it as a subject of its relation
        
        if attributes:
            self.attributes = attributes                 # Dictionary with node's attributes
        else:
             self.attributes = {}

    def __str__(self):
        if self.instanceValue:
            return self.instanceValue
        else:
            return '{} {}'.format(self.ontologyNode.name, self.instanceID)
        
    def getInstanceID(self):
        return self.instanceID
    
    def getInstanceValue(self):
        return self.instanceValue
    
    def getOntologyNode(self):
        return self.ontologyNode
    
    def getAttributes(self):
        return self.attributes
    
    def getAttribute(self, key = None):
        if key is not None:
            if key in self.attributes:
                return self.attributes[key]
            else:
                return None
        else:
            return None
            
    def getRelationLinks(self, relationName = None, conceptName = None):
        if relationName is None:
            if conceptName is None:
                return self.relationLinks
            else:
                conceptCN = []
                
                for r in self.relationLinks:
                    for dn in self.relationLinks[r]:
                        if dn.ontologyNode.name == conceptName:
                            conceptCN.append(dn)
            
                return conceptCN
        
        if relationName in self.relationLinks:
            relDNs = self.relationLinks[relationName]
            
            if conceptName is None:
                return relDNs
            else:
                conceptCN = []
            
            for dn in relDNs:
                if dn.ontologyNode.name == conceptName:
                    conceptCN.append(dn)
            
            return conceptCN
        else:
            return None
        
    def addRelationLink(self, relationName, dn):
        if relationName is None:
            return
        
        if relationName not in self.relationLinks:
            self.relationLinks[relationName] = []
            
        self.relationLinks[relationName].append(dn)

    def removeRelationLink(self, relationName, dn):
        if relationName is None:
            return
        
        if relationName not in self.relationLinks:
            return
        
        self.relationLinks[relationName].remove(dn)

    def getChildDataNodes(self, conceptName = None):
        containsDNs = self.getRelationLinks('contains')
        
        if conceptName is None:
            return containsDNs

        if containsDNs is None:
            return None

        conceptCN = []
            
        for dn in containsDNs:
            if dn.ontologyNode.name == conceptName:
                conceptCN.append(dn)
        
        return conceptCN
            
    def addChildDataNode(self, dn):
        relationName = 'contains'
        
        if (relationName in self.relationLinks) and (dn in self.relationLinks[relationName]):
            return
        
        self.addRelationLink(relationName, dn)
    
        # Impact
        if relationName not in dn.impactLinks:
            dn.impactLinks[relationName] = []
            
        if self not in dn.impactLinks[relationName]:
            dn.impactLinks[relationName].append(self)

    def removeChildDataNode(self, dn):
        relationName = 'contains'

        self.removeRelationLink(relationName, dn)
        
        # Impact
        dn.impactLinks[relationName].remove(self)

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
    
    # Check if concept is relation
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

    # Get and calculate probability for provided concept and datanodes based on datanodes attributes  - move to concept? - see predict method
    def __getProbability(self, conceptRelation,  *dataNode, fun=None, epsilon = 0.00001):
        # Build probability key
        key = '<' + conceptRelation.name + '>'
        
        # Get probability
        value = dataNode[0].getAttribute(key) # ? What if more DataNodes
        
        if value is None:
            return [0, 1]
        
        # Process probability through function and apply epsilon
        if isinstance(value, torch.Tensor):
            with torch.no_grad():
                value = [_it.cpu().detach().numpy() for _it in value]
        if isinstance(value, (np.ndarray, np.generic)):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
                _list = value
                if _list[0] > 1-epsilon:
                    _list[0] = 1-epsilon
                elif _list[1] > 1-epsilon:
                    _list[1] = 1-epsilon
                    
                if _list[0] < epsilon:
                    _list[0] = epsilon
                elif _list[1] < epsilon:
                    _list[1] = epsilon
                       
                # Apply fun on probabilities 
                if fun is not None:
                    _list = [fun(_it) for _it in _list]
                
                return _list # Return probability

        return [1, 0] # ?
            
    # Calculate ILP prediction for data graph with this instance as a root based on the provided list of concepts and relations
    def inferILPConstrains(self, *_conceptsRelations, fun=None):
        
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
                
            # Get candidates (dataNodes or relation relationName for the concept) from the graph starting from the current data node
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
                    graphResultsForPhraseToken[currentConceptOrRelation.name] = np.zeros((no_candidateds, 2))
                    
                elif len(currentCandidate) == 2: # currentConceptOrRelation is a pair thus candidates tuple has two element
                    graphResultsForPhraseRelation[currentConceptOrRelation.name] = np.zeros((no_candidateds, no_candidateds, 2))
                    
                elif len(currentCandidate) == 3: # currentConceptOrRelation is a triple thus candidates tuple has three element
                    graphResultsForTripleRelations[currentConceptOrRelation.name] = np.zeros((no_candidateds, no_candidateds, no_candidateds, 2))
                        
                else: # No support for more then three candidates yet
                    pass
                
        # Collect probabilities for candidates 
        for currentConceptOrRelation in conceptsRelations:
            currentCandidates = candidates_currentConceptOrRelation[currentConceptOrRelation]
            
            if currentCandidates is None:
                continue
            
            for currentCandidate in currentCandidates:
                if len(currentCandidate) == 1:   # currentConceptOrRelation is a concept thus candidates tuple has only single element
                    currentProbability = self.__getProbability(currentConceptOrRelation, *currentCandidate, fun=fun)
                    if currentProbability:
                        graphResultsForPhraseToken[currentConceptOrRelation.name][currentCandidate[0].instanceID] = currentProbability
                    
                elif len(currentCandidate) == 2: # currentConceptOrRelation is a pair thus candidates tuple has two element
                    currentCandidate1 = currentCandidate[0]
                    currentCandidate2 = currentCandidate[1]
                    currentProbability = currentConceptOrRelation.predict(self, currentCandidate1, currentCandidate2)
                    currentProbability = self.__getProbability(currentConceptOrRelation, *currentCandidate, fun=fun)
                    if currentProbability:
                        graphResultsForPhraseRelation[currentConceptOrRelation.name][currentCandidate1.instanceID][currentCandidate2.instanceID] = currentProbability

                elif len(currentCandidate) == 3: # currentConceptOrRelation is a triple thus candidates tuple has three element     
                    currentCandidate1 = currentCandidate[0]
                    currentCandidate2 = currentCandidate[1]
                    currentCandidate3 = currentCandidate[2]
                    currentProbability = currentConceptOrRelation.predict(self, currentCandidate1, currentCandidate2, currentCandidate3)
                    
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

# Class constructing the data graph based on the sensors data during the model execution
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
    def __findDatanodes(self, dns, conceptName):
        if (dns is None) or (len(dns) == 0):
            return []
         
        returnDns = []
        for dn in dns:
            if dn.ontologyNode.name == conceptName:
               returnDns.append(dn) 
               
        if len(returnDns) > 0:
            return returnDns
        
        # Test if fist dn from dns has link to dn of the given concept type
        dns0CDN = dns[0].getRelationLinks(relationName = None, conceptName = conceptName)
        if (dns0CDN is not None) and (len(dns0CDN) > 0):
            for dn in dns:
                dnCN = dn.getRelationLinks(relationName = None, conceptName = conceptName)
                
                if dnCN is not None:
                    returnDns = returnDns + dnCN
                else:
                    pass
        else:
            for dn in dns:
                cDNs = []
                for r in dn.getRelationLinks().values():
                    for _dn in r:
                        cDNs.append(_dn)
                        
                returnDns = returnDns + self.__findDatanodes(cDNs, conceptName)
    
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
    
    # Build or update relation datanode in the data graph for a given key
    def __buildRelationLink(self, value, conceptInfo, keyDataName):
        # Check if data graph started
        existingRootDns = dict.__getitem__(self, 'dataNode') # Datanodes roots
        
        if not existingRootDns:
            return # No graph yet - information about relation should not be provided yet
        
        # Find if datanodes for relation have been created
        relationName = conceptInfo['concept'].name
        existingDnsForRelation = self.__findDatanodes(existingRootDns, relationName) # Datanodes of the current concept
        
        # Find datanodes connected by this relation
        existingDnsForAttr = []
        for attr in conceptInfo['relationAttrs']:
            _existingDnsForAttr = self.__findDatanodes(existingRootDns, conceptInfo['relationAttrs'][0].name) # Datanodes of the given relations attribute concept
            
            if not _existingDnsForAttr:
                return
            
            existingDnsForAttr.append(_existingDnsForAttr)
    
        # Check the shape of the value
        for i, a in enumerate(existingDnsForAttr):
            if len(a) != value.shape[i]:
                return # ???? Wrong shape not matching relation - !!!
            
        # Create or update relation nodes
        if len(existingDnsForRelation) == 0: # No Datanode of this relation created yet
            for p in product(*existingDnsForAttr):
                instanceValue = ""
                instanceID = ' -> '.join([n.ontologyNode.name + ' ' + str(n.getInstanceID()) for n in p])
                rDn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                
                _p = tuple([__p.getInstanceID() for __p in p])                
                rDn.attributes[keyDataName] = value[_p]
                
                for i, dn in enumerate(p):
                    if i == 0:
                        dn.addRelationLink(relationName, rDn)
                    else:
                        # Impact
                        if relationName not in dn.impactLinks:
                            dn.impactLinks[relationName] = []
                            
                        if rDn not in dn.impactLinks[relationName]:
                            dn.impactLinks[relationName].append(rDn)
                        
                    rDn.addRelationLink("connects", dn)
                    
        else: # Datanode with this relation already created  -update it with new attribute
            for rDn in existingDnsForRelation:
                p = []
                for dn in rDn.getRelationLinks().values():
                    p.append(dn[0].instanceID)
                    
                _p = tuple(p)
                rDn.attributes[keyDataName] = value[_p]
 
    # Build or update data node in the data graph for a given key
    def __buildDataNode(self, value, conceptInfo, keyDataName):
        if not dict.__contains__(self, 'dataNode'): # No datanode yet
            dns = []
            if type(value) is not Tensor: # value is not Tensor
                instanceValue = ""
                instanceID = dict.__getitem__(self, "READER")
                _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                
                _dn.attributes[keyDataName] = value
                
                dns.append(_dn)
            else: # value is Tensor
                for vIndex, v in enumerate(value):
                    instanceValue = ""
                    instanceID = vIndex
                    _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                    
                    _dn.attributes[keyDataName] = v
                    
                    dns.append(_dn)
                        
            dict.__setitem__(self, 'dataNode', dns)
        else: # Datanodes already created
            existingRootDns = dict.__getitem__(self, 'dataNode') # Datanodes roots
            existingDnsForConcept = self.__findDatanodes(existingRootDns, conceptInfo['concept'].name) # Datanodes of the current concept
            l = len(existingDnsForConcept)
            dns = []
            
            if len(existingDnsForConcept) == 0: # No Datanode of this concept created yet
                if (type(value) is not Tensor) and (type(value) is not list) : # Value is not Tensor or list
                    instanceValue = ""
                    instanceID = dict.__getitem__(self, "READER")
                    _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                    
                    _dn.attributes[keyDataName] = value
                    ''
                    if conceptInfo['root'] and len(conceptInfo['contains']) > 0:
                        for _contains in conceptInfo['contains']:
                            _existingDnsForConcept = self.__findDatanodes(existingRootDns, _contains.name) # Datanodes of the current concept
                            
                            for eDN in _existingDnsForConcept:
                                _dn.addChildDataNode(eDN)
                             
                        dns.append(_dn)   
                        dict.__setitem__(self, 'dataNode', dns) # New root
                    else:
                        pass # ????

                else: # Value is Tensor or List
                    if (type(value[0]) is not Tensor) and (type(value[0]) is not list) : # Internal Value is not Tensor or list
                        dns1 = []
                        for vIndex, v in enumerate(value):
                            instanceValue = ""
                            instanceID = vIndex
                            _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                            
                            _dn.attributes[keyDataName] = v
                            
                            dns1.append(_dn)
                            
                        dns.append(dns1)
                    else:
                        for vIndex, v in enumerate(value):
                            dns1 = []
                            for vIndex1, v1 in enumerate(v):
                                instanceValue = ""
                                instanceID = vIndex * len(v) + vIndex1
                                _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                                
                                _dn.attributes[keyDataName] = v1
                                
                                dns1.append(_dn)
                            
                            dns.append(dns1)
           
                    # Forward information in the sensor data
                    if len(conceptInfo['containedIn']) > 0:
                        myContainedIn = self.__findDatanodes(existingRootDns, conceptInfo['containedIn'][0].name) # Assume single containedIn for now
                           
                        for myContainedInIndex, myContainedIn in enumerate(myContainedIn):
                            for dn in dns[myContainedInIndex]:
                                myContainedIn.addChildDataNode(dn)
                                
                    # Backward information in the sensor data
                    if len(conceptInfo['contains']) > 0:
                        myContains = self.__findDatanodes(existingRootDns, conceptInfo['contains'][0].name) # Assume single contains for now
                        
                        if len(myContains) > 0:
                            # sort myContains ? - should be sorted - order based on creation
                            
                            i = 0
                            for _dnsIndex, _dns in enumerate(dns): 
                                for _dnsIndex1, _dns1 in enumerate(_dns):
                                    _i = _dnsIndex * len(_dns) + _dnsIndex1
                                    indexes = value[_i]
                                
                                    for _ in range(indexes[0], indexes[1] + 1):
                                        _dns1.addChildDataNode(myContains[i])
                                        i = i + 1
                            
            else: # Datanode with this concept already created
                if (type(value) is not Tensor) and (type(value) is not list): # Value is not Tensor or list
                    if type(existingDnsForConcept[0]) is DataNode:
                        existingDnsForConcept[0].attributes[keyDataName] = value
                else: # Value is Tensor
                    if len(existingDnsForConcept) > len(value):
                        pass
                    elif len(existingDnsForConcept) == len(value):
                        for vIndex, v in enumerate(value):
                            if type(existingDnsForConcept[vIndex]) is DataNode:
                                existingDnsForConcept[vIndex].attributes[keyDataName] = v
                            else:
                                pass
                    elif len(existingDnsForConcept) < len(value):
                        pass
 
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
            self.__buildRelationLink(value, conceptInfo, keyDataName) # Build or update relation link
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
    
    # Method returning constructed datanode
    def getDataNode(self):
        if dict.__contains__(self, 'dataNode'):
            _dataNode = dict.__getitem__(self, 'dataNode')
            
            if len(_dataNode) > 0:                
                return _dataNode[0]
        
        return None
