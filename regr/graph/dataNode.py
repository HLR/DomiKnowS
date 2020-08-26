import numpy as np
import torch
from collections import OrderedDict, namedtuple
import re
import time
from itertools import product
from .dataNodeConfig import dnConfig 
from torch.tensor import Tensor

from regr.graph import graph
from regr.graph.logicalConstrain import eqL
from regr.solver import ilpOntSolverFactory

import logging
from logging.handlers import RotatingFileHandler

logName = __name__
logLevel = logging.CRITICAL
logFilename='datanode.log'
logFilesize=5*1024*1024*1024
logBackupCount=4
logFileMode='a'

if dnConfig and (isinstance(dnConfig, dict)):
    if 'log_name' in dnConfig:
        logName = dnConfig['log_name']
    if 'log_level' in dnConfig:
        logLevel = dnConfig['log_level']
    if 'log_filename' in dnConfig:
        logFilename = dnConfig['log_filename']
    if 'log_filesize' in dnConfig:
        logFilesize = dnConfig['log_filesize']
    if 'log_backupCount' in dnConfig:
        logBackupCount = dnConfig['log_backupCount']
    if 'log_fileMode' in dnConfig:
        logFileMode = dnConfig['log_fileMode']
        
# Create file handler and set level to info
ch = RotatingFileHandler(logFilename, mode=logFileMode, maxBytes=logFilesize, backupCount=logBackupCount, encoding=None, delay=0)
ch.doRollover()
# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s:%(funcName)s - %(message)s')
# Add formatter to ch
ch.setFormatter(formatter)
print("Log file for %s is in: %s"%(logName,ch.baseFilename))

# --- Create loggers
_DataNode__Logger  = logging.getLogger(logName)
_DataNode__Logger.setLevel(logLevel)
# Add ch to logger
_DataNode__Logger.addHandler(ch)
# Don't propagate
_DataNode__Logger.propagate = False
        
# --- Create loggers
_DataNodeBulder__Logger  = logging.getLogger("dataNodeBuilder")
_DataNodeBulder__Logger.setLevel(logLevel)
# Add ch to logger
_DataNodeBulder__Logger.addHandler(ch)
# Don't propagate
_DataNodeBulder__Logger.propagate = False
        
# Class representing single data instance with relation  links to other data nodes
class DataNode:
    def __init__(self, instanceID = None, instanceValue = None, ontologyNode = None, relationLinks = {}, attributes = {}):

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
        
    def __repr__(self):
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
        
        if not isinstance(relationName, str):
            relationName = relationName.name
            
        if relationName in self.relationLinks:
            relDNs = self.relationLinks[relationName]
            
            if conceptName is None:
                return relDNs
            else:
                conceptCN = []
            
            if not isinstance(conceptName, str):
                conceptName = conceptName.name
            
            for dn in relDNs:
                if dn.ontologyNode.name == conceptName:
                    conceptCN.append(dn)
            
            return conceptCN
        else:
            return []
        
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
            if isinstance(conceptName, str):
                if dn.ontologyNode.name == conceptName:
                    conceptCN.append(dn)
            else:
                 if dn.ontologyNode == conceptName:
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

    def __testDataNode(self, dn, test):
        if test is None:
            return False
            
        if isinstance(test, tuple) or isinstance(test, list): # tuple with at least three elements (concept, key elements, value of attribute)
            _test = []
            for t in test:
                if isinstance(t, tuple):
                    r = self.__testDataNode(dn, t)
                    
                    if not r:
                        return False
                else:
                    _test.append(t)
             
            if len(_test) == 0:
                return True
            else:
                test = _test
               
            if len(test) >= 3:     
                if isinstance(test[0], str):
                    if dn.getOntologyNode().name != test[0]:
                        return False
                else:
                    if dn.getOntologyNode().name != test[0].name:
                        return False
                    
                keys = test[1:-1]
                v = dn.getAttribute(*keys)
                
                last = test[-1]
                if v == last:
                    return True
                else:
                    return False
        else:
            test = [test]
    
        for i, t in enumerate(test):
            if isinstance(t, int):
                if dn.getInstanceID() == t:
                    return True
                else:
                    return False
                
            if t == "instanceID" and i < len(test) - 1:
                if dn.getInstanceID() == test[i+1]:
                    return True
                else:
                    return False
                
            if not isinstance(t, str):
                t = t.name
            
            if t == dn.getOntologyNode().name:
                return True
            else:
                return False
    
    # Find datanodes in data graph of the given concept 
    def findDatanodes(self, dns = None, select = None, indexes = None):
        if dns is None:
            dns = [self]
            
        returnDns = []
        
        if len(dns) == 0:
            return returnDns
        
        if select == None:
            return returnDns
       
        for dn in dns:
            if self.__testDataNode(dn, select):
                if dn not in returnDns:
                    returnDns.append(dn) 
               
            for r, rValue in dn.getRelationLinks().items():
                if r == "contains":
                    continue
                
                for _dn in rValue:
                    if self.__testDataNode(_dn, select):
                        if dn not in returnDns:
                            returnDns.append(_dn) 
                    
        if (indexes != None):
            _returnDns = []
            for dn in returnDns:
                fit = True       
                for indexName, indexValue in indexes.items():
                    if indexName not in dn.relationLinks:
                        fit = False
                        break
                    
                    found = False
                    for _dn in dn.relationLinks[indexName]:
                        if isinstance(indexValue, tuple):
                            _test = []
                            for t in indexValue:
                                if isinstance(t, tuple):
                                    r = self.__testDataNode(_dn, t)
                                    
                                    if r:
                                        found = True
                                        break
                                else:
                                    _test.append(t)
                             
                            if len(_test) == 0:
                                continue
                            else:
                                indexValue = _test
                                
                        if self.__testDataNode(_dn, indexValue):
                            found = True
                            break
                        
                    if not found:
                        fit = False
                        break
                        
                if fit:
                    if dn not in _returnDns:
                        _returnDns.append(dn)
                       
            returnDns = _returnDns
                
        if len(returnDns) > 0:
            return returnDns
        
        # Not found - > call recursively
        for dn in dns:
            _returnDns = self.findDatanodes(dn.getChildDataNodes(), select, indexes)
            
            if _returnDns is not None:
                for _dn in _returnDns:
                    if _dn not in returnDns:
                        returnDns.append(_dn)
    
        return returnDns

    def getAttribute(self, *keys):
        key = ""
        
        for k in keys:
            if key != "":
                key = key + "/"
                
            if isinstance(k, str):
                _k = self.__findConcept(k)
                
                if _k is not None:  
                    key = key + '<' + k +'>'
                else:
                    key = key + k
            else:
                key = key + '<' + k.name +'>'
                    
        if key in self.attributes:
            return self.attributes[key]
        else:
            return None
              
    # Get root of the data node
    def getRootDataNode(self):
        if "contains" in self.impactLinks:
            return self.impactLinks["contains"][0].getRootDataNode()
        else:
            return self
                
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
        if usedGraph is None:
            usedGraph = self.ontologyNode.getOntologyGraph()
            
        if len(conceptRelation.has_a()) > 0:  
            return True
        
        for _isA in conceptRelation.is_a():
            _conceptRelation = _isA.dst
            
            if self.__isRelation(_conceptRelation, usedGraph):
                return True
        
        return False 
    
    def __getRelationAttrNames(self, conceptRelation, usedGraph = None):
        if usedGraph is None:
            usedGraph = self.ontologyNode.getOntologyGraph()
            
        if len(conceptRelation.has_a()) > 0:  
            relationAttrs = OrderedDict()
            for arg_id, rel in enumerate(conceptRelation.has_a()): 
                srcName = rel.src.name
                dstName = rel.dst.name                
                relationAttr = self.__findConcept(dstName, usedGraph)
    
                relationAttrs[rel.name] = relationAttr
                
            return relationAttrs
        
        for _isA in conceptRelation.is_a():
            _conceptRelation = _isA.dst
            
            resultForCurrent = self.__getRelationAttrNames(_conceptRelation, usedGraph)
            
            if bool(resultForCurrent):
                return resultForCurrent
        
        return None 
    
    # Find the root parent of relation of the given relation
    def __findRootConceptOrRelation(self, relationConcept, usedGraph = None):
        if usedGraph is None:
            usedGraph = self.ontologyNode.getOntologyGraph()
        
        if isinstance(relationConcept, str):
            relationConcept = self.__findConcept(relationConcept)
            
        # Does this concept or elation has parent (through _isA)
        for _isA in relationConcept.is_a():
            _relationConcept = _isA.dst
            
            return  self.__findRootConceptOrRelation(_relationConcept, usedGraph)
        
        # If the provided concept or relation is root (has not parents)
        return relationConcept 

    def __isHardConstrains(self, conceptRelation):
        # Check if dedicated DataNodes has been created for this concept or relation
        currentConceptOrRelationDNs = self.findDatanodes(select = conceptRelation)
        if len(currentConceptOrRelationDNs) > 0:
            return True
        
        # Find the root parent  of the given concept or relation
        rootRelation = self.__findRootConceptOrRelation(conceptRelation)
        
        if bool(rootRelation):
            # Find dedicated DataNodes for this rootRelation
            rootRelationDNs = self.findDatanodes(select = rootRelation)
            
            if bool(rootRelationDNs):
                key = str(conceptRelation) # Key is just the name of the concept or relation
               
                # Check if the DataNode has this attribute
                for dn in rootRelationDNs:
                    if key in dn.attributes:
                        return True
                
        return False
            
    def __getHardConstrains(self, conceptRelation, reltationAttrs, currentCandidate):
        key = conceptRelation.name  
        rootRelation = self.__findRootConceptOrRelation(conceptRelation)
        
        if rootRelation == conceptRelation:
            if bool(reltationAttrs):
                indexes = {}
                for i, attr in enumerate(reltationAttrs):
                    if len(currentCandidate) > i:
                        indexes[attr] = ('instanceID', currentCandidate[i].instanceID)
            
                v = self.findDatanodes(select = conceptRelation, indexes = indexes)
                if len(v) == 0:
                    currentProbability = [1, 0]
                else:
                    currentProbability = [0, 1]
                    
                return currentProbability
        else:
            dns = currentCandidate[0].getRelationLinks(relationName = rootRelation, conceptName = None)
            if bool(dns):
                dn = dns[currentCandidate[1].getInstanceID()]
        
                value = None
                if key in dn.attributes:
                    value = dn.attributes[key]
        
                if value is not None:
                    _value = value.item()
                    
                    if _value == 0:
                        currentProbability = [1, 0]
                    else:
                        currentProbability = [0, 1]
                        
                    return currentProbability
        
        return [1, 0]

    def __getAttributeValue(self, key, conceptRelation, *dataNode):
        # Get attribute with probability
        if len(dataNode) == 1:
            value = dataNode[0].getAttribute(key) 
        else:
            rootRelation = self.__findRootConceptOrRelation(conceptRelation)
            rl = dataNode[0].getRelationLinks(relationName = rootRelation, conceptName = None)
            
            if not rl:
                return [float("nan"), float("nan")]
            
            if len(dataNode) == 2:
                 value = dataNode[0].getRelationLinks(relationName = rootRelation, conceptName = None)[dataNode[1].getInstanceID()].getAttribute(key)
            elif len(dataNode) == 3:
                value = dataNode[0].getRelationLinks(relationName = rootRelation, conceptName = None)[dataNode[1].getInstanceID()].getAttribute(key)
            else:
                return [float("nan"), float("nan")] # ?
        
        if value is None: # No probability value - return negative probability 
            return [float("nan"), float("nan")]
        
        # Translate probability to list
        if isinstance(value, torch.Tensor):
            with torch.no_grad():
                value = value.cpu().detach().numpy()
        if isinstance(value, (np.ndarray, np.generic)):
            value = value.tolist()  
        
        return value
    
     # Get and calculate probability for provided concept and datanodes based on datanodes attributes  - move to concept? - see predict method
    def __getLabel(self, conceptRelation,  *dataNode, fun=None, epsilon = None):
        # Build probability key to retrieve attribute
        key = '<' + conceptRelation.name + '>/label'
        
        value = self.__getAttributeValue(key, conceptRelation, *dataNode)
        
        return value
        
    # Get and calculate probability for provided concept and datanodes based on datanodes attributes  - move to concept? - see predict method
    def __getProbability(self, conceptRelation,  *dataNode, fun=None, epsilon = 0.00001):
        # Build probability key to retrieve attribute
        key = '<' + conceptRelation.name + '>'
        
        value = self.__getAttributeValue(key, conceptRelation, *dataNode)
            
        # Process probability through function and apply epsilon
        if isinstance(value, (list, tuple)):
            _list = value
            if epsilon is not None:
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
                _list = fun(_list)
            
            return _list # Return probability

        return [float("nan"), float("nan")]
                    
    def __collectConceptsAndRelations(self, dn):
        conceptsAndRelations = set()
        
        for att in dn.attributes:
            if att[0] == '<' and att[-1] == '>':
                conceptsAndRelations.add(att[1:-1])
        
        for relName, rel in dn.getRelationLinks().items():
            if relName == "contains":
                continue
            
            if len(rel) > 0:
                for att in rel[0].attributes:
                    if att[0] == '<' and att[-1] == '>':
                        conceptsAndRelations.add(att[1:-1])
        
        dnChildren = dn.getChildDataNodes()
        
        if dnChildren != None:
            for child in dnChildren:
                conceptsAndRelations.update(self.__collectConceptsAndRelations(child))

        return conceptsAndRelations

    def __collectEqlsFromLC(self, usedGraph = None):
        if usedGraph is None:
            usedGraph = self.ontologyNode.getOntologyGraph()
            
        lcEqls = {}
        for id, lc in usedGraph._logicalConstrains.items(): 
            if isinstance(lc, eqL):
                if len(lc.e) != 3:
                    continue
                
                if str(lc.e[0]) in lcEqls:
                    lcEqls[str(lc.e[0])].append(lc.e)
                else:
                    lcEqls[str(lc.e[0])] = [lc.e]
                
        return lcEqls
                    
    # Prepare data for ILP based on data graph with this instance as a root based on the provided list of concepts and relations
    def __prepareILPData(self, *_conceptsRelations, dnFun = None, fun=None, epsilon = 0.00001,  minimizeObjective = False):
        # Check if concepts and/or relations have been provided for inference
        if (_conceptsRelations == None) or len(_conceptsRelations) == 0:
            _conceptsRelations = self.__collectConceptsAndRelations(self) # Collect all concepts and relation from graph as default set
        
        conceptsRelations = [] # Will contain concept or relation  - translated to ontological concepts if provided using names
        hardConstrains = []
        _instances = set() # Set of all the candidates across all the concepts to be consider in the ILP constrains
        candidates_currentConceptOrRelation = OrderedDict()
        # Collect all the candidates for concepts and relations in conceptsRelations
        for _currentConceptOrRelation in _conceptsRelations:
            # Convert string to concept if provided as string
            if isinstance(_currentConceptOrRelation, str):
                currentConceptOrRelation = self.__findConcept(_currentConceptOrRelation)
                
                if currentConceptOrRelation is None:
                    continue # String is not a name of concept or relation
            else:
                currentConceptOrRelation = _currentConceptOrRelation
                
            # Check if it is a hard constrain concept or relation - check if dataNode exist of this type
            if self.__isHardConstrains(currentConceptOrRelation):
                hardConstrains.append(str(currentConceptOrRelation)) # Hard Constrain
                            
            # Get candidates (dataNodes or relation relationName for the concept) from the graph starting from the current data node
            currentCandidates = currentConceptOrRelation.candidates(self)
            
            conceptsRelations.append(currentConceptOrRelation)

            if currentCandidates is None:
                continue
            
            if self.__isRelation(currentConceptOrRelation): # Check if relation
                pass
            
            _currentInstances = set()
            for currentCandidate in currentCandidates:
                _currentInstances.add(currentCandidate)
                
                for candidateElement in currentCandidate:
                    _instances.add(candidateElement)

            candidates_currentConceptOrRelation[str(currentConceptOrRelation)] = _currentInstances
          
        if len(_instances) == 0:
            return None, None, None, None, None, None, None
        
        # Get ids of the instances
        infer_candidatesID = []
        for currentCandidate in _instances:
            infer_candidatesID.append(currentCandidate.instanceID)
        
        infer_candidatesID.sort() # Sort the list of instances

        no_candidateds = len(infer_candidatesID)    # Number of candidates
        
        # Get eqls from LCs
        lcEqls = self.__collectEqlsFromLC()
        
        # Create numpy arrays for collected probabilities and zero them
        graphResultsForPhraseToken = dict()
        
        # Make this generic for any number of relation attributes
        graphResultsForPhraseRelation = dict()
        graphResultsForTripleRelations = dict()

        for currentConceptOrRelation in conceptsRelations:
            currentCandidates = candidates_currentConceptOrRelation[str(currentConceptOrRelation)]
            
            if not currentCandidates:
                continue
            
            c = next(iter(currentCandidates or []), None)
            if len(c) == 1:   # currentConceptOrRelation is a concept thus candidates tuple has only single element
                graphResultsForPhraseToken[str(currentConceptOrRelation)] = np.zeros((no_candidateds, 2))
                    
                if str(currentConceptOrRelation) in lcEqls:
                    for e in lcEqls[str(currentConceptOrRelation)]:
                        if isinstance(e[2], set):
                            for e2 in e[2]:
                                key = str(e[0]) + ":" + e[1] + ":" + str(e2)
                                hardConstrains.append(key) # Hard Constrain
        
                                graphResultsForPhraseToken[key] = np.zeros((no_candidateds, 2))
                        else:
                            key = str(e[0]) + ":" + e[1] + ":" + str(e[2])
                            hardConstrains.append(key) # Hard Constrain
    
                            graphResultsForPhraseToken[key] = np.zeros((no_candidateds, 2))
                    
            elif len(c) == 2: # currentConceptOrRelation is a pair thus candidates tuple has two element
                graphResultsForPhraseRelation[str(currentConceptOrRelation)] = np.zeros((no_candidateds, no_candidateds, 2))
                
                if str(currentConceptOrRelation) in lcEqls:
                    for e in lcEqls[str(currentConceptOrRelation)]:
                        if isinstance(e[2], set):
                            for e2 in e[2]:
                                key = str(e[0]) + ":" + e[1] + ":" + str(e2)
                                hardConstrains.append(key) # Hard Constrain
                                
                                graphResultsForPhraseRelation[key] = np.zeros((no_candidateds, no_candidateds, 2))
                        else:
                            key = str(e[0]) + ":" + e[1] + ":" + str(e[2])
                            hardConstrains.append(key) # Hard Constrain
    
                            graphResultsForPhraseRelation[key] = np.zeros((no_candidateds, no_candidateds, 2))

            elif len(c) == 3: # currentConceptOrRelation is a triple thus candidates tuple has three element
                graphResultsForTripleRelations[str(currentConceptOrRelation)] = np.zeros((no_candidateds, no_candidateds, no_candidateds, 2))
                
                if str(currentConceptOrRelation) in lcEqls:
                    for e in lcEqls[str(currentConceptOrRelation)]:
                        if isinstance(e[2], set):
                            for e2 in e[2]:
                                key = str(e[0]) + ":" + e[1] + ":" + str(e2)
                                hardConstrains.append(key) # Hard Constrain
                                
                                graphResultsForTripleRelations[key] = np.zeros((no_candidateds, no_candidateds, no_candidateds, 2))
                        else:
                            key = str(e[0]) + ":" + e[1] + ":" + str(e[2])
                            hardConstrains.append(key) # Hard Constrain
                    
                            graphResultsForTripleRelations[key] = np.zeros((no_candidateds, no_candidateds, no_candidateds, 2))
                    
            else: # No support for more then three candidates yet
                pass
                
        # Collect probabilities for candidates 
        for currentConceptOrRelation in conceptsRelations:
            currentCandidates = candidates_currentConceptOrRelation[str(currentConceptOrRelation)]
            
            if currentCandidates is None:
                continue
            
            reltationAttrs = self.__getRelationAttrNames(currentConceptOrRelation)
            
            for currentCandidate in currentCandidates:
                if len(currentCandidate) == 1:   # currentConceptOrRelation is a concept thus candidates tuple has only single element
                    currentCandidate1 = currentCandidate[0]
                    if str(currentConceptOrRelation) in hardConstrains:
                        currentProbability = self.__getHardConstrains(currentConceptOrRelation, reltationAttrs, currentCandidate)
                    else:
                        currentProbability = dnFun(currentConceptOrRelation, *currentCandidate, fun=fun, epsilon=epsilon)
                        
                    if currentProbability:
                        graphResultsForPhraseToken[str(currentConceptOrRelation)][currentCandidate[0].instanceID] = currentProbability
                    
                elif len(currentCandidate) == 2: # currentConceptOrRelation is a pair thus candidates tuple has two element
                    currentCandidate1 = currentCandidate[0]
                    currentCandidate2 = currentCandidate[1]
                    if str(currentConceptOrRelation) in hardConstrains:
                        currentProbability = self.__getHardConstrains(currentConceptOrRelation, reltationAttrs, currentCandidate)
                    else:
                        currentProbability = dnFun(currentConceptOrRelation, *currentCandidate, fun=fun, epsilon=epsilon)
                    
                    if currentProbability:
                        graphResultsForPhraseRelation[str(currentConceptOrRelation)][currentCandidate1.instanceID][currentCandidate2.instanceID] = currentProbability

                    if str(currentConceptOrRelation) in lcEqls:
                        for e in lcEqls[str(currentConceptOrRelation)]:
                            #dns = self.findDatanodes(select = (e[0], e[1], e[2]))
                            
                            _e2 = currentCandidate[0].relationLinks[str(e[0])][currentCandidate[1].instanceID].attributes[e[1]].item()
                            
                            if isinstance(e[2], set):
                                for e2 in e[2]:
                                    if e2 == _e2:
                                         currentProbability = [0, 1]
                                    else:
                                        currentProbability = [1, 0]
                                    
                                    key = str(e[0])+ ":" + e[1] + ":" + str(e2)
                                    
                                    graphResultsForPhraseRelation[key][currentCandidate1.instanceID][currentCandidate2.instanceID] = currentProbability
                            else:
                                if e[2] == _e2:
                                     currentProbability = [0, 1]
                                else:
                                    currentProbability = [1, 0]
                                
                                key = str(e[0])+ ":" + e[1] + ":" + str(e[2])
                                
                                graphResultsForPhraseRelation[key][currentCandidate1.instanceID][currentCandidate2.instanceID] = currentProbability

                elif len(currentCandidate) == 3: # currentConceptOrRelation is a triple thus candidates tuple has three element     
                    currentCandidate1 = currentCandidate[0]
                    currentCandidate2 = currentCandidate[1]
                    currentCandidate3 = currentCandidate[2]
                    currentProbability = dnFun(currentConceptOrRelation, *currentCandidate, fun=fun, epsilon=epsilon)
                    
                    __myLogger.debug("currentConceptOrRelation is %s for relation %s and tokens %s %s %s - no variable created"%(currentConceptOrRelation,currentCandidate1,currentCandidate2,currentCandidate3,currentProbability))

                    if currentProbability:
                        graphResultsForTripleRelations[str(currentConceptOrRelation)][currentCandidate1.instanceID][currentCandidate2.instanceID][currentCandidate3.instanceID]= currentProbability
                    
                else: # No support for more then three candidates yet
                    pass
        
        # Get ontology graphs and then ilpOntsolver
        myOntologyGraphs = {self.ontologyNode.getOntologyGraph()}
        
        for currentConceptOrRelation in conceptsRelations:
            currentOntologyGraph = currentConceptOrRelation.getOntologyGraph()
            
            if currentOntologyGraph is not None:
                myOntologyGraphs.add(currentOntologyGraph)
                
        myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(myOntologyGraphs)
                        
        return myilpOntSolver, infer_candidatesID, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForTripleRelations, hardConstrains, candidates_currentConceptOrRelation
    
    # Calculate ILP prediction for data graph with this instance as a root based on the provided list of concepts and relations
    def inferILPConstrains(self, *_conceptsRelations, fun=None, epsilon = 0.00001, minimizeObjective = False):
        if not _conceptsRelations:
            _conceptsRelations = ()
            
        myilpOntSolver, infer_candidatesID, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForTripleRelations, hardConstrains, candidates_currentConceptOrRelation= \
            self.__prepareILPData(*_conceptsRelations,  dnFun = self.__getProbability, fun = fun, epsilon = epsilon)
        
        if not myilpOntSolver:
            return
        
        # Call ilpOntsolver with the collected probabilities for chosen candidates
        tokenResult, pairResult, tripleResult = \
            myilpOntSolver.calculateILPSelection(infer_candidatesID, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForTripleRelations, minimizeObjective = minimizeObjective, hardConstrains = hardConstrains)
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for concept_name in tokenResult:
            if concept_name in hardConstrains:
                continue
            
            currentCandidates = candidates_currentConceptOrRelation[concept_name]
            
            key = '<' + concept_name + '>/ILP'
            
            for infer_candidate in currentCandidates:
                infer_candidate[0].attributes[key] = torch.tensor([tokenResult[concept_name][infer_candidate[0].getInstanceID()]], device=device) 
                
        for concept_name in pairResult:
            if concept_name in hardConstrains:
                continue
            
            rootRelation = self.__findRootConceptOrRelation(concept_name)
            currentCandidates = candidates_currentConceptOrRelation[concept_name]
            
            key = '<' + concept_name + '>/ILP'
            
            for infer_candidate in currentCandidates:
                infer_candidate[0].relationLinks[rootRelation.name][infer_candidate[1].getInstanceID()].attributes[key] = \
                    torch.tensor(pairResult[concept_name][infer_candidate[0].getInstanceID()][infer_candidate[1].getInstanceID()], device=device)
                        
        return #tokenResult, pairResult, tripleResult
    
        # Update triple
        for concept_name in tripleResult:
            if concept_name in hardConstrains:
                continue
            
            currentCandidates = candidates_currentConceptOrRelation[concept_name]
            for infer_candidate in currentCandidates:
                if infer_candidate[0] != infer_candidate[1] and infer_candidate[0] != infer_candidate[2] and infer_candidate[1] != infer_candidate[2]:
                    if DataNode.PredictionType["ILP"] not in infer_candidate[0].attributes:
                        infer_candidate[0].attributes[DataNode.PredictionType["ILP"]] = {}
                        
                    infer_candidate[0].attributes[DataNode.PredictionType["ILP"]][concept, (infer_candidate[1], infer_candidate[2])] = \
                        tripleResult[concept_name][infer_candidate[0].instanceID, infer_candidate[1].instanceID, infer_candidate[2].instanceID]
    
    def verifySelection(self, *_conceptsRelations):
        if not _conceptsRelations:
            _conceptsRelations = ()
            
        myilpOntSolver, infer_candidatesID, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForTripleRelations, hardConstrains, candidates_currentConceptOrRelation = \
            self.__prepareILPData(*_conceptsRelations, dnFun = self.__getLabel)
            
        if not myilpOntSolver:
            return False
        
        verifyResult = myilpOntSolver.verifySelection(infer_candidatesID, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForTripleRelations)
        
        return verifyResult

# Class constructing the data graph based on the sensors data during the model execution
class DataNodeBuilder(dict):
    def __init__(self, *args, **kwargs ):
        dict.__init__(self, *args, **kwargs )
        _DataNodeBulder__Logger.info("")
        _DataNodeBulder__Logger.info("Called")

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    # Change elements of value to tuple if they are list - in order to use the value as dictionary key
    def __changToTuple(self, v):
        if isinstance(v, list):
            _v = []
            for v1 in v:
                _v.append(self.__changToTuple(v1))
                
            return tuple(_v)
        else:
            return v
        
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
        for s in skey: # skey[2:]:
            counterNanme = counterNanme + '/' + s
            
        if not dict.__contains__(self, counterNanme):
            try:
               dict.__setitem__(self, counterNanme, {_value : {"counter": 1, "recent" : True}})
            except TypeError:
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
        
    # Cache with found concept infromation    
    __conceptInfoCache = {} 
    # Collect concept information defined in the graph
    def __findConceptInfo(self, usedGraph, concept):
        if concept in self.__conceptInfoCache:
            return self.__conceptInfoCache[concept]
        
        conceptInfo = {}
        
        conceptInfo['concept'] = concept
        
        conceptInfo['relation'] = False
        conceptInfo['relationAttrs'] = {}
        for arg_id, rel in enumerate(concept.has_a()): 
            conceptInfo['relation'] = True
            relationName = rel.src.name
            conceptName = rel.dst.name
                            
            conceptAttr = self.__findConcept(conceptName, usedGraph)

            conceptInfo['relationAttrs'][rel.name] = conceptAttr
            
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
        
        self.__conceptInfoCache[concept] = conceptInfo
        
        return conceptInfo
    
    # Build or update relation datanode in the data graph for a given key
    def __buildRelationLink(self, vInfo, conceptInfo, keyDataName):
        relationName = conceptInfo['concept'].name
         
        # Check if data graph started
        existingRootDns = dict.__getitem__(self, 'dataNode') # Datanodes roots
        
        if not existingRootDns:
            _DataNodeBulder__Logger.error('No dataNode created yet - abandon processing relation link dataNode value for %s and attribute %s'%(relationName,keyDataName))
            return # No graph yet - information about relation should not be provided yet
        
        # Find DataNodes connected by this relation based on graph definition
        existingDnsForAttr = OrderedDict() # DataNodes for Attributes of the relation
        for relationAttributeName, relationAttributeConcept in conceptInfo['relationAttrs'].items():
            _existingDnsForAttr = existingRootDns[0].findDatanodes(existingRootDns, relationAttributeConcept.name) # DataNodes of the given relations attribute concept
             
            if _existingDnsForAttr:
                existingDnsForAttr[relationAttributeName] = _existingDnsForAttr
                _DataNodeBulder__Logger.info('Found %i dataNodes of the attribute %s concept %s'%(len(_existingDnsForAttr),relationAttributeName,relationAttributeConcept.name))
            else:
                existingDnsForAttr[relationAttributeName] = []
                _DataNodeBulder__Logger.warning('Not found dataNodes of the attribute %s concept %s'%(relationAttributeName,relationAttributeConcept.name))

        # Check the shape of the value if it is a Tensor of shape equal to the number of relation attributes
        if hasattr(vInfo.value, 'shape'):
            for i, (k, a) in enumerate(existingDnsForAttr.items()):
                if vInfo.dim <= i: # check if value tensor has enough dimensions
                    _DataNodeBulder__Logger.error('Wrong dimension of value; it is %i which is less then % relation attributes - abandon processing relation link dataNode value for %s'%(vInfo.dim,i,relationName))
                    return # Wrong shape not matching relation attribute number
                
                if len(a) != vInfo.value.shape[i]: # check if the given dimension has a size equal to the number of dataNodes found for the given attribute
                    _DataNodeBulder__Logger.\
                        error('Wrong size of value for dimension %i; it is %i not equal to the number of relation attributes %i - abandon processing relation link dataNode value for %s'%(i,vInfo.value.shape[i],len(a),relationName))
                    return # Wrong shape not matching relation attribute number
        else:
            _DataNodeBulder__Logger.error('The value is not Tensor, it is %s -  - abandon processing relation link dataNode value for %s'%(type(vInfo.value),relationName))
            return # Not Tensor
            
        # --- Create or update relation nodes
        
        # Find if DatnNodes for this relation have been created
        existingDnsForRelation = existingRootDns[0].findDatanodes(existingRootDns, relationName) # DataNodes of the current relation
        
        _DataNodeBulder__Logger.info('Processing relation link dataNode for %s, found %i existing dataNode of this type - provided value has length %i'%(relationName,len(existingDnsForRelation),vInfo.len))

        # -- No DataNode of this relation created yet
        if len(existingDnsForRelation) == 0: 
            attributeNames = [*existingDnsForAttr]
            
            for p in product(*existingDnsForAttr.values()): # Candidates
                _p = tuple([__p.getInstanceID() for __p in p]) # tuple with ids of the hdataNodes
                  
                instanceID = ' -> '.join([n.ontologyNode.name + ' ' + str(n.getInstanceID()) for n in p])
                
                # Check if this relation link is excluded by the provided value (need to have 'Candidate' substring in the tensor column name
                if (vInfo.value.names[0] is not None) and ('Candidate' in vInfo.value.names[0]) and (vInfo.value[_p] == 0):
                    _DataNodeBulder__Logger.debug('DataNode for relation link with id %s is not in the Candidate list - it is not added'%(instanceID))
                    continue
                else:
                    _DataNodeBulder__Logger.debug('DataNode for relation link with id %s created'%(instanceID))

                instanceValue = ""
                
                rDn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept']) # Create dataNode for relation link
                rDn.attributes[keyDataName] = vInfo.value[_p] # Add value of the attribute to this relation link dataNode
                
                # Add this relation link dataNode to all the dataNodes linked by it
                for i, dn in enumerate(p):
                    if i == 0:
                        dn.addRelationLink(relationName, rDn) # First dataNode has relation link dataNode
                    else:
                        # Impact
                        if relationName not in dn.impactLinks: # Next dataNode has relation link dataNode in impact
                            dn.impactLinks[relationName] = []
                            
                        if rDn not in dn.impactLinks[relationName]:
                            dn.impactLinks[relationName].append(rDn)
                        
                    rDn.addRelationLink(attributeNames[i], dn) # Add this dataNode as value for the attribute in the relation link dataNode
        else:    
            # -- DataNode with this relation already created  - update it with new attribute value
            _DataNodeBulder__Logger.info('Updating attribute %s in relation link dataNodes'%(keyDataName))

            for rDn in existingDnsForRelation: # Loop through all relation links dataNodes
                # Collect ids of dataNodes linked by this relation link dataNode
                p = []
                for dn in rDn.getRelationLinks().values():
                    p.append(dn[0].instanceID)
                   
                _p = tuple(p) # Create tuple from ids to access value for this combination of dataNodes
                rDn.attributes[keyDataName] = vInfo.value[_p] # Add / /Update value of the attribute

    def __createInitialdDataNode(self, vInfo, conceptInfo, keyDataName):
        conceptName = conceptInfo['concept'].name

        dns = []
                   
        _DataNodeBulder__Logger.info('Creating initial dataNode - provided value has length %i'%(vInfo.len))

        if vInfo.len == 1:
            instanceValue = ""
            
            if "READER" in self:
                instanceID = dict.__getitem__(self, "READER")
            else:
                instanceID = 0
                
            _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
            
            _dn.attributes[keyDataName] = vInfo.value
            
            _DataNodeBulder__Logger.info('Created single dataNode with id %s of type %s'%(instanceID,conceptName))
            dns.append(_dn)
        elif vInfo.len > 1:
            for vIndex, v in enumerate(vInfo.value):
                instanceValue = ""
                instanceID = vIndex
                _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                
                _dn.attributes[keyDataName] = v
                
                dns.append(_dn)
                        
            _DataNodeBulder__Logger.info('Created %i dataNodes of type %s'%(len(dns),conceptName))
        else:
            pass # ?
                    
        _DataNodeBulder__Logger.info('Updated elements in the root dataNodes list %s'%(dns))
        dict.__setitem__(self, 'dataNode', dns)
        
        return # Done - End the method
    
    def __createSingleDataNode(self, vInfo, conceptInfo, keyDataName):
        conceptName = conceptInfo['concept'].name
        existingRootDns = dict.__getitem__(self, 'dataNode') # Get DataNodes roots

        # -- Create the new dataNode 
        instanceValue = ""
        instanceID = 0
        _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
        _dn.attributes[keyDataName] = vInfo.value
        
        _dnLinked = False # If True then this new dataNode got linked with existing dataNodes
        
        _DataNodeBulder__Logger.info('Created single new dataNode with id %i of type %s'%(instanceID,conceptName))

        # Add it as parent to existing dataNodes
        if len(conceptInfo['contains']) > 0:
            for _contains in conceptInfo['contains']:
                _existingDnsForConcept = existingRootDns[0].findDatanodes(existingRootDns, _contains.name) # DataNodes of the current concept
                
                if _existingDnsForConcept:
                    _DataNodeBulder__Logger.info('Adding this dataNode as a parent to %i dataNodes of type %s'%(len(_existingDnsForConcept),_contains.name))
                else:
                    pass
                    
                # Adding the new dataNode as parent to the dataNodes of type _contains
                for eDN in _existingDnsForConcept:
                    _dn.addChildDataNode(eDN)
                    _dnLinked == True

        # Add it as child to existing datanodes
        if len(conceptInfo['containedIn']) > 0:                
            for _containedIn in conceptInfo['containedIn']:
                myContainedInDns = existingRootDns[0].findDatanodes(existingRootDns, _containedIn.name)
                   
                if myContainedInDns:
                    if myContainedInDns == 1:
                        _DataNodeBulder__Logger.info('Adding this dataNode as child to single dataNode of type %s'%(_containedIn.name))
                    else:
                        _DataNodeBulder__Logger.warning('Adding the same dataNode as child to %i dataNodes of type %s'%(len(myContainedInDns),_containedIn.name))
                else:
                    _DataNodeBulder__Logger.error('Number of dataNodesets %i different the number of %i dataNodes of type %s - abandon the update'%(len(dns),len(myContainedInDns),_containedIn.name))
                
                for myContainedIn in myContainedInDns:
                    myContainedIn.addChildDataNode(_dn)   
                    _dnLinked = True
    
        # Checking if is root
        if conceptInfo['root']:  # Root concept
            _DataNodeBulder__Logger.info('This dataNode is a root datanode')

            # Needs to update this to support batches
            if "READER" in self:
                _dn.instanceID = dict.__getitem__(self, "READER")
                _DataNodeBulder__Logger.debug('Using key \"READER\"  - %s, as a id for the dataNode'%(_dn.instanceID))
            else:
                _dn.instanceID = 0
                _DataNodeBulder__Logger.debug('Setting id for the dataNode to 0')

            # Update the list of root datanodes 
            _dns = dict.__getitem__(self, 'dataNode')
            _DataNodeBulder__Logger.debug('Existing elements in the root dataNodes list %s'%(_dns))

            dns = []
            for dnE in _dns:
                if 'contains' not in dnE.impactLinks:
                    dns.append(dnE)
                    
            dns.append(_dn) # Add the root to the list
            _dnLinked = True
            
            _DataNodeBulder__Logger.info('Updated elements in the root dataNodes list %s'%(dns))
            dict.__setitem__(self, 'dataNode', dns) # Updated the dict
            
        # Check if the new dataNode is connected to existing dataNodes, if not add it to the list of root dataNodes
        if not _dnLinked:
            _DataNodeBulder__Logger.info('The new dataNode has not been linked with existing dataNodes - adding it to the list of root dataNodes')
            dns = dict.__getitem__(self, 'dataNode')
            _DataNodeBulder__Logger.info('Updated elements in the root dataNodes list %s'%(dns))
            dns.append(_dn)
        
    def __createMultiplyDataNode(self, vInfo, conceptInfo, keyDataName):
        conceptName = conceptInfo['concept'].name
        existingRootDns = dict.__getitem__(self, 'dataNode') # Get DataNodes roots
        
        dns = [] # List of lists of created dataNodes
                
        if vInfo.dim == 1: # Internal Value is simple; it is not Tensor or list
            _DataNodeBulder__Logger.info('Adding single set of dataNodes of type %s'%(conceptName))

            dns1 = []
            for vIndex, v in enumerate(vInfo.value):
                instanceValue = ""
                instanceID = vIndex
                _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                
                _dn.attributes[keyDataName] = v
                
                dns1.append(_dn)
                                    
            _DataNodeBulder__Logger.info('Added %i new dataNodes %s'%(len(dns1),dns1))
            dns.append(dns1)
        elif vInfo.dim == 2:
            _DataNodeBulder__Logger.info('Adding %i sets of dataNodes of type %s'%(vInfo.len,conceptName))

            for vIndex, v in enumerate(vInfo.value):
                dns1 = []
                for vIndex1, v1 in enumerate(v):
                    instanceValue = ""
                    instanceID = vIndex * len(v) + vIndex1
                    _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                    
                    _dn.attributes[keyDataName] = v1
                    
                    dns1.append(_dn)
                                    
                _DataNodeBulder__Logger.info('Added %i new dataNodes %s'%(len(dns1),dns1))
                dns.append(dns1)
        else: # vInfo.dim > 2
            _DataNodeBulder__Logger.warning('Dimension of value %i is larger then 2 not supported'%(vInfo.dim))

        _dnLinked = False
            
        # ---------- This section still needs work
        v0Info = self.__processAttributeValue(vInfo.value[0], keyDataName) # Get internal structure of the value - test first element of the value
        if v0Info.len == 2: # Add it as parent to existing dataNodes - Backward information in the sensor data
            if len(conceptInfo['contains']) > 0:
                myContains = existingRootDns[0].findDatanodes(existingRootDns, conceptInfo['contains'][0].name) # Assume single contains for now
                
                if len(myContains) > 0: 
                                       
                    i = 0
            
                    for _dnsIndex, _dns in enumerate(dns): # Set of dataNodes
                        for _dnsIndex1, _dns1 in enumerate(_dns): # dataNode in the current set
                            _i = _dnsIndex * len(_dns) + _dnsIndex1
                            
                            indexes = value[_i]
                        
                            for _ in range(indexes[0], indexes[1] + 1):
                                _dns1.addChildDataNode(myContains[i])
                                i = i + 1
        # ---------- 
        
        # Add them as children to existing dataNodes - Forward information in the sensor data
        if len(conceptInfo['containedIn']) > 0: # Number of parent concepts to the new dataNodes
            for currentParentConcept in conceptInfo['containedIn']:
                currentParentConceptName = currentParentConcept.name
                currentParentDns = existingRootDns[0].findDatanodes(existingRootDns, currentParentConceptName)
                
                if currentParentDns:
                    if len(currentParentDns) == len(dns):
                        _DataNodeBulder__Logger.info('Adding dataNodes as children to %i dataNodes of type %s'%(len(currentParentDns),currentParentConceptName))
                    else:
                        _DataNodeBulder__Logger.error('Number of dataNodesets %i different the number of %i dataNodes of type %s - abandon the update'%(len(dns),len(currentParentDns),currentParentConceptName))
                        continue
                else:
                    _DataNodeBulder__Logger.info('Not found any dataNode type %s - this type is in the list of types of potential children of the current concept'%(currentParentConceptName))
                
                for currentParentDnIndex, currentParentDn in enumerate(currentParentDns):
                    for curentChildDn in dns[currentParentDnIndex]: # Set of new dataNodes for the current parent dataNode
                        currentParentDn.addChildDataNode(curentChildDn)    
                                               
                    _DataNodeBulder__Logger.info('Added %i dataNodes %s as children to the dataNode with id %s'%(len(dns[currentParentDnIndex]),dns[currentParentDnIndex],currentParentDn.instanceID))
                    _dnLinked = True # New dataNodes are linked with existing dataNodes
                    
    def __updateDataNodes(self, vInfo, conceptInfo, keyDataName):
        conceptName = conceptInfo['concept'].name
        
        existingRootDns = dict.__getitem__(self, 'dataNode') # Get DataNodes roots
        existingDnsForConcept = existingRootDns[0].findDatanodes(existingRootDns, conceptName) # Try to get DataNodes of the current concept
            

        if keyDataName in existingDnsForConcept[0].attributes:
            _DataNodeBulder__Logger.info('Updating attribute %s in existing dataNodes - found %i dataNodes of type %s'%(keyDataName, len(existingDnsForConcept),conceptName))
        else:
            _DataNodeBulder__Logger.info('Adding attribute %s in existing dataNodes - found %i dataNodes of type %s'%(keyDataName, len(existingDnsForConcept),conceptName))

        if len(existingDnsForConcept) == 1: # Single dataNode
            existingDnsForConcept[0].attributes[keyDataName] = vInfo.value
            if vInfo.len > 1:
                _DataNodeBulder__Logger.warning('Provided value has length %i but found only a single existing dataNode - the value  as whole is a new value of the attribute %s'%(vInfo.len,keyDataName))
        else: # Multiple dataNodes
            if len(existingDnsForConcept) > vInfo.len: # Not enough elements in the value 
                _DataNodeBulder__Logger.warning('Provided value has length %i but found %i existing dataNode - abandon the update'%(vInfo.len,len(existingDnsForConcept)))
            elif len(existingDnsForConcept) == vInfo.len: # Number of  value elements matches the number of found dataNodes
                for vIndex, v in enumerate(vInfo.value):
                    if isinstance(existingDnsForConcept[vIndex], DataNode): # Check if dataNode
                       existingDnsForConcept[vIndex].attributes[keyDataName] = v
                    else:
                        _DataNodeBulder__Logger.error('Element %i in the list is not a dataNode - skipping it'%(vIndex))
            elif len(existingDnsForConcept) < vInfo.len: # Too many elements in the value
                _DataNodeBulder__Logger.warning('Provided value has length %i but found %i existing dataNode - abandon the update'%(vInfo.len,len(existingDnsForConcept)))
                        
    # Build or update dataNode in the data graph for a given relationAttributeConcept
    def __buildDataNode(self, vInfo, conceptInfo, keyDataName):
        conceptName = conceptInfo['concept'].name
       
        if not dict.__contains__(self, 'dataNode'):   # ------ No DataNode yet
            self.__createInitialdDataNode( vInfo, conceptInfo, keyDataName)
            return # Done - End the method
        else:
            # ---------- DataNodes already created
            
            existingRootDns = dict.__getitem__(self, 'dataNode') # Get DataNodes roots
            existingDnsForConcept = existingRootDns[0].findDatanodes(existingRootDns, conceptName) # Try to get DataNodes of the current concept
            
            if len(existingDnsForConcept) == 0:# Check if datannote for this concept already created                    
                # No Datanode of this concept created yet
    
                # If attribute value is a single element - will create a single new dataNode
                if vInfo.len == 1: 
                    self.__createSingleDataNode(vInfo, conceptInfo, keyDataName)
                else: # -- Value is multiple elements
                    self.__createMultiplyDataNode(vInfo, conceptInfo, keyDataName)
            else: # DataNode with this concept already created - update it
                self.__updateDataNodes(vInfo, conceptInfo, keyDataName)
                
    # Method processing value of for the attribute - determining it it should be treated as a single element. 
    # It returns a tuple with elements specifying the length of the first dimension of the value, the number of dimensions of the value and the original value itself
    def __processAttributeValue(self, value, keyDataName):
        ValueInfo = namedtuple('ValueInfo', ["len", "value", 'dim'])

        if not isinstance(value, (Tensor, list)): # It is scalar value
            return ValueInfo(len = 1, value = value, dim=0) 
            
        if isinstance(value, Tensor) and value.dim() == 0: # It is a Tensor but also scalar value
            return ValueInfo(len = 1, value = value.item(), dim=0)
        
        if (len(value) == 1): # It is Tensor or list with length 1 - treat it as scalar
            if isinstance(value, list): # Unpack the value
                return ValueInfo(len = 1, value = value[0], dim=0)
            elif isinstance(value, Tensor):
                return ValueInfo(len = 1, value = value.item(), dim=0)

        #  If it is Tensor or list with length 2 but it is for attribute providing probabilities - assume it is a scalar value
        if len(value) ==  2 and keyDataName[0] == '<': 
            return ValueInfo(len = 1, value = value, dim=0)
        
        if isinstance(value, list): 
            if not isinstance(value[0], (Tensor, list)):
                return ValueInfo(len = len(value), value = value, dim=1)
            elif not isinstance(value[0][0], (Tensor, list)):
                return ValueInfo(len = len(value), value = value, dim=2)
            elif not isinstance(value[0][0][0], (Tensor, list)):
                return ValueInfo(len = len(value), value = value, dim=3)
            else:
                _DataNodeBulder__Logger.warning('Dimension of nested list value for key %s is more then 3 returning dimension 4'%(key))
                return ValueInfo(len = len(value), value = value, dim=4)

        elif isinstance(value, Tensor):
            return ValueInfo(len = len(value), value = value, dim=value.dim())
    
    # Overloaded __setitem method of Dictionary - tracking sensor data and building corresponding data graph
    def __setitem__(self, key, value):
        start = time.time()
        self.__addSetitemCounter()
          
        skey = key.split('/')
        
        # Check if the key with this value has been set recently
        # If not create a new sensor for it
        # If yes stop __setitem__ and return - the same value for the key was added last time that key was set
        if self.__addSensorCounters(skey, value):
            return # Stop __setitem__ for repeated key value combination
        
        if isinstance(value, Tensor):
            _DataNodeBulder__Logger.info('key - %s,  value - %s, shape %s'%(key,type(value),value.shape))
        elif isinstance(value, list):
            _DataNodeBulder__Logger.info('key - %s,  value - %s, length %s'%(key,type(value),len(value)))
        else:
            _DataNodeBulder__Logger.info('key - %s,  value - %s'%(key,type(value)))

        if value is None:
            _DataNodeBulder__Logger.error('The value for the key %s is None - abandon the update'%(key))
            return dict.__setitem__(self, key, value)
                
        if len(skey) < 2:            
            _DataNodeBulder__Logger.error('The key %s has only two elements, needs at least three - abandon the update'%(key))
            return dict.__setitem__(self, key, value)
        
        usedGraph = dict.__getitem__(self, "graph")

        # Find if the key include concept from graph
        
        keyWithoutGraphName = usedGraph.cutGraphName(skey)
       
        # Check if found concept in the key
        if not keyWithoutGraphName:
            _DataNodeBulder__Logger.warning('key - %s has not concept part - returning'%(key))
            return dict.__setitem__(self, key, value)
 
        # Find description of the concept in the grpah
        _conceptName = keyWithoutGraphName[0]
        concept = self.__findConcept(_conceptName, usedGraph)
        
        if not concept:
            _DataNodeBulder__Logger.warning('_conceptName - %s has not been found in the used graph %s - returning'%(_conceptName,usedGraph.fullname))
            return dict.__setitem__(self, key, value)
        
        conceptInfo = self.__findConceptInfo(usedGraph, concept)
        
        # Create key for datanonde construction
        keyDataName = "".join(map(lambda x: '/' + x, keyWithoutGraphName[1:]))
        keyDataName = keyDataName[1:] # cut first '/' from the string
        
        vInfo = self.__processAttributeValue(value, keyDataName)

        # Decide if this is datanode for concept or relation link
        if conceptInfo['relation']:
            _DataNodeBulder__Logger.debug('%s found in the graph; it is a relation'%(_conceptName))
            self.__buildRelationLink(vInfo, conceptInfo, keyDataName) # Build or update relation link
        else:                       
            _DataNodeBulder__Logger.debug('%s found in the graph; it is a concept'%(_conceptName))
            self.__buildDataNode(vInfo, conceptInfo, keyDataName)   # Build or update Data node
            
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
            pass # Error when adding entry to dictionary ?
        
        return r                
                                             
    def __delitem__(self, key):
        return dict.__delitem__(self, key)

    def __contains__(self, key):
        return dict.__contains__(self, key)
    
    # Add or increase generic counter counting number of setitem calls
    def __addGetDataNodeCounter(self):
        counterName = 'Counter' + 'GetDataNode'
        if not dict.__contains__(self, counterName):
            dict.__setitem__(self, counterName, 1)
        else:
            currentCounter =  dict.__getitem__(self, counterName)
            dict.__setitem__(self, counterName, currentCounter + 1)
            
    # Method returning constructed dataNode - the fist in the list
    def getDataNode(self):
        self.__addGetDataNodeCounter()
        
        if dict.__contains__(self, 'dataNode'):
            _dataNode = dict.__getitem__(self, 'dataNode')
            
            if len(_dataNode) > 0:  
                if len(_dataNode) == 1:
                    _DataNodeBulder__Logger.info('Returning dataNode with id %s of type %s'%(_dataNode[0].instanceID,_dataNode[0].getOntologyNode()))
                else:
                    _DataNodeBulder__Logger.warning('Returning first dataNode with id %s of type %s - there are total %i dataNodes'%(_dataNode[0].instanceID,_dataNode[0].getOntologyNode(),len(_dataNode)))

                return _dataNode[0]
        
        _DataNodeBulder__Logger.error('Returning None - there are no dataNode')
        return None
    
    # Method returning all constructed dataNodes 
    def getBatchDataNodes(self):
        self.__addGetDataNodeCounter()
        
        if dict.__contains__(self, 'dataNode'):
            _dataNode = dict.__getitem__(self, 'dataNode')
            
            if len(_dataNode) > 0:  
                
                _DataNodeBulder__Logger.info('Returning %i dataNodes - %s'%(len(_dataNode),_dataNode))

                return _dataNode[0]
        
        _DataNodeBulder__Logger.error('Returning None - there are no dataNodes')
        return None