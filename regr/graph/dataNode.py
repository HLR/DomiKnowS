import numpy as np
import torch.cuda
from collections import OrderedDict, namedtuple
import time
from itertools import product
from .dataNodeConfig import dnConfig 
from torch.tensor import Tensor

from regr.graph.logicalConstrain import eqL
from regr.solver import ilpOntSolverFactory

import logging
from logging.handlers import RotatingFileHandler
from .property import Property
from .concept import Concept

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
            
        self.impactLinks = {}                            # Dictionary with dataNodes impacting this dataNode by having it as a subject of its relation
        
        if attributes:
            self.attributes = attributes                 # Dictionary with node's attributes
        else:
            self.attributes = {}
                     
    class DataNodeError(Exception):
        pass
    
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
        
    def __reprDeep__(self,  strRep = ""):
        
        rel = [*self.getRelationLinks().keys()]
        if 'contains' in rel:
            rel.remove('contains')
            
        relSting = None
        if len(rel) > 0:
            relSting = ' (' + rel + ')'
            
        if relSting:
            strRep += self.ontologyNode.name + str(rel)
        else:
            strRep += self.ontologyNode.name 
            
        childrenDns = {}
        for cDn in self.getChildDataNodes():
            if cDn.getOntologyNode().name not in childrenDns:
                childrenDns[cDn.getOntologyNode().name] = []

            childrenDns[cDn.getOntologyNode().name].append(cDn)
                
        strRep += '\n'
        for childType in childrenDns:
            strRep += '\t' + childrenDns[childType][0].__repr__(strRep)
        
        return strRep
    
    def getInstanceID(self):
        return self.instanceID
    
    def getInstanceValue(self):
        return self.instanceValue
    
    def getOntologyNode(self):
        return self.ontologyNode
    
    # --- Attributes methods
    
    def getAttributes(self):
        return self.attributes     
    
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
           
    # --- Relation Link methods
     
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
            
        if dn in self.relationLinks[relationName]:
            return 
        
        self.relationLinks[relationName].append(dn)
        
        # Impact
        if relationName not in dn.impactLinks:
            dn.impactLinks[relationName] = []
            
        if self not in dn.impactLinks[relationName]:
            dn.impactLinks[relationName].append(self)

    def removeRelationLink(self, relationName, dn):
        if relationName is None:
            return
        
        if relationName not in self.relationLinks:
            return
        
        self.relationLinks[relationName].remove(dn)
        
        # Impact
        if relationName in  dn.impactLinks:
            dn.impactLinks[relationName].remove(self)

    # --- Contains (children) relation methods
    
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

    def removeChildDataNode(self, dn):
        relationName = 'contains'

        self.removeRelationLink(relationName, dn)
        
    def resetChildDataNode(self):
        relationName = 'contains'

        self.relationLinks[relationName] = []
        
    # --- Equality methods
    
    def getEqualTo(self, equalName = "equalTo", conceptName = None):
        if conceptName:
            dns = self.getRelationLinks(relationName=equalName)
            
            filteredDns = []
            for dn in dns:
                if dn.getOntologyNode().name == conceptName:
                    filteredDns.append(dn)
                    
            return filteredDns
        else:
            return self.getRelationLinks(relationName=equalName)
    
    def addEqualTo(self, equalDn, equalName = "equalTo"):
        self.addRelationLink(equalName, equalDn)
        
    def removeEqualTo(self, equalDn, equalName = "equalTo"):
        self.removeRelationLink(equalName, equalDn)

    # --- Query methods
    
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
    
    # Find concept and relation types of DataNodes
    def findConceptsNamesInDatanodes(self, dns = None, conceptNames = set(), relationNames = set()):
        if dns is None:
            dns = [self]
            
        for dn in dns:
            conceptNames.add(dn.getOntologyNode().name)
            for relName, _ in dn.getRelationLinks().items():
                if relName != 'contains':
                    relationNames.add(relName)
                
            self.findConceptsNamesInDatanodes(dns=dn.getChildDataNodes(), conceptNames = conceptNames, relationNames = relationNames)
            
        return conceptNames, relationNames
    
    # Find dataNodes in data graph of the given concept 
    def findDatanodes(self, dns = None, select = None, indexes = None, depth = 0):
        if dns is None:
            dns = [self]
            
        returnDns = []
        
        if select is None:
            if depth == 0 and not returnDns:
                _DataNode__Logger.warning('Not found any DataNode - no value for select provided')
                
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
        newDepth = depth + 1
        for dn in dns:
            _returnDns = self.findDatanodes(dn.getChildDataNodes(), select = select, indexes = indexes, depth = newDepth)
            
            if _returnDns is not None:
                for _dn in _returnDns:
                    if _dn not in returnDns:
                        returnDns.append(_dn)
    
        if depth == 0 and not returnDns:
            _DataNode__Logger.debug('Not found any DataNode for - %s -'%(select))
    
        if returnDns:
            returnDnsNotSorted = OrderedDict()
            for dn in returnDns:
                returnDnsNotSorted[dn.getInstanceID()] = dn
                    
            returnDnsSorted = OrderedDict(sorted(returnDnsNotSorted.items()))
        
            returnDns = [*returnDnsSorted.values()]
        
        return returnDns
          
    # Get root of the dataNode
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
            for _, rel in enumerate(conceptRelation.has_a()): 
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
                attrNames = []
                for _, rel in enumerate(rootRelation.has_a()): 
                    attrNames.append(rel.name)
                        
                _rDN = dataNode[0].findDatanodes(select = rootRelation, indexes = {attrNames[0] : dataNode[0].getInstanceID(), attrNames[1]: dataNode[1].getInstanceID()})
                
                if _rDN:
                    value = _rDN[0].getAttribute(key)
                else:
                    return [1, float("nan")] # ?
                #value = dataNode[0].getRelationLinks(relationName = rootRelation, conceptName = None)[dataNode[1].getInstanceID()].getAttribute(key)
            elif len(dataNode) == 3:
                value = dataNode[0].getRelationLinks(relationName = rootRelation, conceptName = None)[dataNode[1].getInstanceID()].getAttribute(key)
            else:
                return [1, float("nan")] # ?
        
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
                    
    def __collectConceptsAndRelations(self, dn, conceptsAndRelations = set()):
        
        # Find concepts in dataNode - concept are in attributes from learning sensors
        for att in dn.attributes:
            if att[0] == '<' and att[-1] == '>':
                if att[1:-1] not in conceptsAndRelations:
                    conceptsAndRelations.add(att[1:-1])
                    _DataNode__Logger.info('Found concept %s in dataNode %s'%(att[1:-1],dn))

        # Find relations in dataNode - relation are in attributes from learning sensors
        for relName, rel in dn.getRelationLinks().items():
            if relName == "contains":
                continue
            
            if len(rel) > 0:
                for att in rel[0].attributes:
                    if att[0] == '<' and att[-1] == '>':
                        if att[1:-1] not in conceptsAndRelations:
                            conceptsAndRelations.add(att[1:-1])
                            _DataNode__Logger.info('Found relation %s in dataNode %s'%(att[1:-1],dn))

        dnChildren = dn.getChildDataNodes()
        
        # Recursively find concepts and relations in children dataNodes 
        if dnChildren != None:
            for child in dnChildren:
                self.__collectConceptsAndRelations(child, conceptsAndRelations = conceptsAndRelations)

        return conceptsAndRelations

    def __collectEqlsFromLC(self, usedGraph = None):
        if usedGraph is None:
            usedGraph = self.ontologyNode.getOntologyGraph()
            
        lcEqls = {}
        for _, lc in usedGraph._logicalConstrains.items(): 
            if isinstance(lc, eqL):
                if len(lc.e) != 3:
                    continue
                
                if str(lc.e[0]) in lcEqls:
                    lcEqls[str(lc.e[0])].append(lc.e)
                else:
                    lcEqls[str(lc.e[0])] = [lc.e]
                
        return lcEqls
                    
    # Prepare data for ILP based on data graph with this instance as a root based on the provided list of concepts and relations
    def __prepareILPData(self, *_conceptsRelations, dnFun = None, fun=None, epsilon = 0.00001):
        # Check if concepts and/or relations have been provided for inference
        if not _conceptsRelations:
            _conceptsRelations = self.__collectConceptsAndRelations(self) # Collect all concepts and relation from graph as default set

            if len(_conceptsRelations) == 0:
                _DataNode__Logger.error('Not found any concepts or relations for inference in provided DataNode %s'%(self))
                raise DataNode.DataNodeError('Not found any concepts or relations for inference in provided DataNode %s'%(self))
            else:        
                _DataNode__Logger.info('Found - %s - as a set of concepts and relations for inference'%(_conceptsRelations))
        else:
            pass

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
                    _DataNode__Logger.warning('Concept or relation name %s not found in the graph'%(currentConceptOrRelation))
                    continue # String is not a name of concept or relation
            else:
                currentConceptOrRelation = _currentConceptOrRelation
                
            # Check if it is a hard constrain concept or relation - check if dataNode exist of this type
            if self.__isHardConstrains(currentConceptOrRelation):
                hardConstrains.append(str(currentConceptOrRelation)) # Hard Constrain
                            
            # Get candidates (dataNodes or relation relationName for the concept) from the graph starting from the current data node
            currentCandidates = [*currentConceptOrRelation.candidates(self, logger = _DataNode__Logger)]
            
            conceptsRelations.append(currentConceptOrRelation)

            if not currentCandidates:
                _DataNode__Logger.warning('Not found any candidates for %s'%(currentConceptOrRelation))
                continue
            else:
                _DataNode__Logger.info('Found candidates - %s - for %s'%(currentCandidates,currentConceptOrRelation))

            if self.__isRelation(currentConceptOrRelation): # Check if relation
                pass
            
            _currentInstances = set()
            for currentCandidate in currentCandidates:
                _currentInstances.add(currentCandidate)
                
                for candidateElement in currentCandidate:
                    _instances.add(candidateElement)

            candidates_currentConceptOrRelation[str(currentConceptOrRelation)] = _currentInstances
          
        if len(_instances) == 0:
            _DataNode__Logger.error('Not found any candidates in DataNode %s for concepts and relations - %s'%(self,_conceptsRelations))
            raise DataNode.DataNodeError('Not found any candidates in DataNode %s for concepts and relations - %s'%(self,_conceptsRelations))
        
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
                _DataNode__Logger.warning('No candidates for %s'%(currentConceptOrRelation))
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
                _DataNode__Logger.warning('Not found any candidates for %s'%(currentConceptOrRelation))
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
                    
                    _DataNode__Logger.debug("currentConceptOrRelation is %s for relation %s and tokens %s %s %s - no variable created"%(currentConceptOrRelation,currentCandidate1,currentCandidate2,currentCandidate3,currentProbability))

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
        if len(_conceptsRelations) == 0:
            _DataNode__Logger.info('Called with empty list of concepts and relations for inference')
        else:
            from regr.graph import Concept
            _DataNode__Logger.info('Called with - %s - list of concepts and relations for inference'%([x.name if isinstance(x, Concept) else x for x in _conceptsRelations]))
            
        if not _conceptsRelations:
            _conceptsRelations = ()
            
        myilpOntSolver, infer_candidatesID, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForTripleRelations, hardConstrains, candidates_currentConceptOrRelation= \
            self.__prepareILPData(*_conceptsRelations,  dnFun = self.__getProbability, fun = fun, epsilon = epsilon)
        
        if not myilpOntSolver:
            _DataNode__Logger.error("ILPSolver not initialized")
            raise DataNode.DataNodeError("ILPSolver not initialized")
        
        # Call ilpOntsolver with the collected probabilities for chosen candidates
        _DataNode__Logger.info("Calling ILP solver with infer_candidatesID %s graphResultsForPhraseToken %s"%(infer_candidatesID,graphResultsForPhraseToken))

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
            attrNames = []
            for _, rel in enumerate(rootRelation.has_a()): 
                attrNames.append(rel.name)
                
            currentCandidates  = candidates_currentConceptOrRelation[concept_name]
            
            key = '<' + concept_name + '>/ILP'
            
            for infer_candidate in currentCandidates:  
                rDN = infer_candidate[0].findDatanodes(select = rootRelation, indexes = {attrNames[0] : infer_candidate[0].getInstanceID(), attrNames[1]: infer_candidate[1].getInstanceID()})[0]

                rDN.attributes[key] = torch.tensor(pairResult[concept_name][infer_candidate[0].getInstanceID()][infer_candidate[1].getInstanceID()], device=device)
                        
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
                        
                    infer_candidate[0].attributes[DataNode.PredictionType["ILP"]][concept_name, (infer_candidate[1], infer_candidate[2])] = \
                        tripleResult[concept_name][infer_candidate[0].instanceID, infer_candidate[1].instanceID, infer_candidate[2].instanceID]
    
    def verifySelection(self, *_conceptsRelations):
        if not _conceptsRelations:
            _conceptsRelations = ()
            
        myilpOntSolver, infer_candidatesID, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForTripleRelations, hardConstrains, candidates_currentConceptOrRelation = \
            self.__prepareILPData(*_conceptsRelations, dnFun = self.__getLabel)
            
        if not myilpOntSolver:
            return False
        
        verifyResult = myilpOntSolver.verifySelection(infer_candidatesID, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForTripleRelations, hardConstrains=hardConstrains)
        
        return verifyResult
    
    def calculateLcLoss(self):
        _conceptsRelations = ()
            
        myilpOntSolver, infer_candidatesID, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForTripleRelations, hardConstrains, candidates_currentConceptOrRelation = \
            self.__prepareILPData(*_conceptsRelations, dnFun = self.__getProbability)
            
        if not myilpOntSolver:
            return False
        
        lcResult = myilpOntSolver.calculateLcLoss(graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForTripleRelations, hardConstrains=hardConstrains)
        
        return lcResult

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
        
    # Cache with found concept information    
    __conceptInfoCache = {} 
    # Collect concept information defined in the graph
    def __findConceptInfo(self, usedGraph, concept):
        conceptInfo = {}
        
        conceptInfo['concept'] = concept
        
        conceptInfo['relation'] = False
        conceptInfo['relationAttrs'] = {}
        for _, rel in enumerate(concept.has_a()): 
            conceptInfo['relation'] = True
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
            
    def __updateConceptInfo(self,  usedGraph, conceptInfo, sensor):
        from regr.sensor.pytorch.relation_sensors import EdgeSensor
        conceptInfo["relationAttrData"] = False

        if (isinstance(sensor, EdgeSensor)):
            relationType = None
            
            if "contains" in sensor.relation.name:
                relationType = "contains"
            elif "is_a" in sensor.relation.name:
                relationType  = "is_a"
            elif "has_a" in sensor.relation.name:
                relationType = "has_a"
            elif "equal" in sensor.relation.name:
                relationType = "equal"
        
            conceptInfo['relationType'] = relationType
            
            #conceptInfo['relation'] = True

            if 'relationAttrs' in conceptInfo:
                conceptInfo['relationAttrsGraph'] = conceptInfo['relationAttrs']
                
            conceptInfo['relationAttrs'] = {}
            
            conceptInfo['relationMode'] = sensor.mode
            conceptInfo['relationAttrs']["src"] = self.__findConcept(sensor.src.name, usedGraph)  
            conceptInfo['relationAttrs']["dst"] = self.__findConcept(sensor.dst.name, usedGraph)  
            
            if conceptInfo['relationAttrs']["dst"] == conceptInfo['concept']:
                conceptInfo['relationAttrData'] = True

    def __updateRootDataNodeList(self, *dns):
        if not dns:
            return
    
        # Get existing roots dataNodes
        if dict.__contains__(self, 'dataNode'):
            dnsRoots = dict.__getitem__(self, 'dataNode')
            _DataNodeBulder__Logger.debug('Existing elements in the root dataNodes list - %s'%(dnsRoots))
        else:
            dnsRoots = []
         
        newDnsRoots = []
        
        # Update list of existing root dataNotes     
        for dnE in dnsRoots: # review them if they got connected
            if 'contains' not in dnE.impactLinks: 
                newDnsRoots.append(dnE)    

        # Add new dataNodes which are not connected to other dataNodes to the root list
        flattenDns = [] # first flatten the list of new dataNodes
        if isinstance(dns[0], list): 
            for dList in dns:
                flattenDns.extend(dList)
                
            if isinstance(flattenDns[0], list): 
                _flattenDns = []
                
                for dList in flattenDns:
                    _flattenDns.extend(dList)
                    
                flattenDns = _flattenDns  
        else:
            flattenDns = dns
            
        for dnE in flattenDns: # review them if they got connected
            if 'contains' not in dnE.impactLinks: 
                newDnsRoots.append(dnE)   
                         
        # Set the updated root list 
        _DataNodeBulder__Logger.info('Updated elements in the root dataNodes list - %s'%(newDnsRoots))
        dict.__setitem__(self, 'dataNode', newDnsRoots) # Updated the dict
    

    # Build or update relation dataNode in the data graph for a given key
    def __buildRelationLink(self, vInfo, conceptInfo, keyDataName):
        relationName = conceptInfo['concept'].name
         
        # Check if data graph started
        existingRootDns = dict.__getitem__(self, 'dataNode') # Datanodes roots
        
        if not existingRootDns:
            _DataNodeBulder__Logger.error('No dataNode created yet - abandon processing relation link dataNode value for %s and attribute %s'%(relationName,keyDataName))
            return # No graph yet - information about relation should not be provided yet
        
        # Find if DatnNodes for this relation have been created
        existingDnsForRelation = self.findDataNodesInBuilder(select = relationName)
        
        if conceptInfo['relationAttrData']:
            index = keyDataName.index('.')
            attrName = keyDataName[0:index]
            
            relationAttrsCacheName = conceptInfo['concept'].name + "RelationAttrsCache"
            
            if not dict.__contains__(self, relationAttrsCacheName):
                dict.__setitem__(self, relationAttrsCacheName, {})
        
            relationAttrsCache =  dict.__getitem__(self, relationAttrsCacheName)
            relationAttrsCache[attrName] = vInfo.value
                
            _DataNodeBulder__Logger.info('Received data for %s for relation link dataNode for %s cached, found %i existing dataNode of this type - provided value has length %i'%(keyDataName,relationName,len(existingDnsForRelation),vInfo.len))
            return 
        
        _DataNodeBulder__Logger.info('Processing relation link dataNode for %s, found %i existing dataNode of this type - provided value has length %i'%(relationName,len(existingDnsForRelation),vInfo.len))

        # Find DataNodes connected by this relation based on graph definition
        existingDnsForAttr = OrderedDict() # DataNodes for Attributes of the relation
        for relationAttributeName, relationAttributeConcept in conceptInfo['relationAttrs'].items():
            _existingDnsForAttr = self.findDataNodesInBuilder(select = relationAttributeConcept.name)
             
            if _existingDnsForAttr:
                existingDnsForAttr[relationAttributeName] = _existingDnsForAttr
                _DataNodeBulder__Logger.info('Found %i dataNodes of the attribute %s concept %s'%(len(_existingDnsForAttr),relationAttributeName,relationAttributeConcept.name))
            else:
                existingDnsForAttr[relationAttributeName] = []
                _DataNodeBulder__Logger.warning('Not found dataNodes of the attribute %s concept %s'%(relationAttributeName,relationAttributeConcept.name))
            
        # -------- Create or update relation nodes
        
        # -- No DataNode of this relation created yet
        if len(existingDnsForRelation) == 0: 
            attributeNames = [*existingDnsForAttr]
            
            relationAttrsCacheName = conceptInfo['concept'].name + "RelationAttrsCache"
            relationAttrsCache = dict.__getitem__(self, relationAttrsCacheName)

            rDns = []
            for i in range(0, vInfo.len):
            
                #_p = tuple([__p.getInstanceID() for __p in p]) # tuple with ids of the dataNodes
                #instanceID = ' -> '.join([n.ontologyNode.name + ' ' + str(n.getInstanceID()) for n in p])
                
                instanceID = i
                
                # Check if this relation link is excluded by the provided value (need to have 'Candidate' substring in the tensor column name
                #if (keyDataName.find("_Candidate_") > 0) and (vInfo.value[_p] == 0):
                #   _DataNodeBulder__Logger.warning('DataNode for relation link with id %s is not in the Candidate list - it is not added'%(instanceID))
                #   continue
                #else:
                    

                instanceValue = ""
                
                rDn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept']) # Create dataNode for relation link
                rDn.attributes[keyDataName] = vInfo.value[i] 
                                
                rDns.append(rDn)
                _DataNodeBulder__Logger.debug('DataNode for relation link with id %s created'%(instanceID))

            for i in range(0, len(rDns)):
                  
                currentRdn = rDns[i]  
                
                for aIndex, a in enumerate(attributeNames):
                      
                    aValue = relationAttrsCache[a][i]
                    for j, av in enumerate(aValue):
                        isInRelation = av.item()
                        if isInRelation == 0:
                            continue
                        
                        dn = existingDnsForAttr[a][j]
                        
                        dn.addRelationLink(relationName, currentRdn)
                        currentRdn.addRelationLink(a, dn)   
            return rDns
        else:    
            # -- DataNode with this relation already created  - update it with new attribute value
            _DataNodeBulder__Logger.info('Updating attribute %s in relation link dataNodes'%(keyDataName))

            existingDnsForRelationNotSorted = OrderedDict()
            for dn in existingDnsForRelation:
                existingDnsForRelationNotSorted[dn.getInstanceID()] = dn
                
            existingDnsForRelationSorted = OrderedDict(sorted(existingDnsForRelationNotSorted.items()))
            
            if len(existingDnsForRelation) != vInfo.len:
                _DataNodeBulder__Logger.error('Number of relation is %i and is different then the length of the provided tensor %i'%(len(existingDnsForRelation),vInfo.len))
                return
 
            for i, rDn in existingDnsForRelationSorted.items(): # Loop through all relation links dataNodes
                rDn.attributes[keyDataName] = vInfo.value[i] # Add / /Update value of the attribute

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
                    
        self.__updateRootDataNodeList(dns)
        
        return dns
    
    def __createSingleDataNode(self, vInfo, conceptInfo, keyDataName):
        # -- Create the new dataNode 
        instanceValue = ""
        instanceID = 0
        _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
        _dn.attributes[keyDataName] = vInfo.value
                
        _DataNodeBulder__Logger.info('Single new dataNode %s created'%(_dn))

        # Add it as parent to existing dataNodes
        if len(conceptInfo['contains']) > 0:
            for _contains in conceptInfo['contains']:
                _existingDnsForConcept = self.findDataNodesInBuilder(select = _contains.name)
                
                if _existingDnsForConcept:
                    _DataNodeBulder__Logger.info('Adding this dataNode as a parent to %i dataNodes of type %s'%(len(_existingDnsForConcept),_contains.name))
                else:
                    pass
                    
                # Adding the new dataNode as parent to the dataNodes of type _contains
                for eDN in _existingDnsForConcept:
                    _dn.addChildDataNode(eDN)

        # Add it as child to existing datanodes
        if len(conceptInfo['containedIn']) > 0:                
            for _containedIn in conceptInfo['containedIn']:
                myContainedInDns = self.findDataNodesInBuilder(select = _containedIn.name)
                   
                if myContainedInDns:
                    if len(myContainedInDns) == 1:
                        _DataNodeBulder__Logger.info('Adding this dataNode as a child to single existing dataNode of type %s'%(_containedIn.name))
                    else:
                        _DataNodeBulder__Logger.warning('Adding the new dataNode as child to %i dataNodes of type %s'%(len(myContainedInDns),_containedIn.name))
                else:
                    _DataNodeBulder__Logger.info('Not found any dataNode of type %s to use as parent for the new dataNode'%(_containedIn.name))
                
                for myContainedIn in myContainedInDns:
                    myContainedIn.addChildDataNode(_dn)   
                    _dnLinked = True
    
        # Checking if is root
        if conceptInfo['root']:  # Root concept
            _DataNodeBulder__Logger.info('This dataNode is a root dataNode')

            # Needs to update this to support batches
            if "READER" in self:
                _dn.instanceID = dict.__getitem__(self, "READER")
                _DataNodeBulder__Logger.debug('Using key \"READER\"  - %s, as a id for the dataNode'%(_dn.instanceID))
            else:
                _dn.instanceID = 0
                _DataNodeBulder__Logger.debug('Setting id for the dataNode to 0')

        self.__updateRootDataNodeList(_dn)
                
        return [_dn]
        
    def __createMultiplyDataNode(self, vInfo, conceptInfo, keyDataName):
        conceptName = conceptInfo['concept'].name
        
        dns = [] # Master List of lists of created dataNodes - each list in the master list represent set of new dataNodes connected to the same parent dataNode (identified by the index in the master list)
                
        if vInfo.dim == 1: # Internal Value is simple; it is not Tensor or list
            _DataNodeBulder__Logger.info('Adding single set of dataNodes (linked to single parent concept) of type %s'%(conceptName))

            dns1 = []
            for vIndex, v in enumerate(vInfo.value):
                instanceValue = ""
                instanceID = vIndex
                _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                
                _dn.attributes[keyDataName] = v
                
                dns1.append(_dn)
                                    
            _DataNodeBulder__Logger.info('Added %i new dataNodes - %s'%(len(dns1),dns1))
            dns.append(dns1)

            # Check if value has information about children - need to be tuples with two indexes - start index of children  - end index of children for the given new dataNode
            if isinstance(vInfo.value[0], tuple): 
                # Add it as parent to existing dataNodes - Backward information in the sensor data
                _DataNodeBulder__Logger.info('The value is build of tuples - providing information about children of the new dataNodes ')

                if len(conceptInfo['contains']) > 0:
                    childType = conceptInfo['contains'][0].name # Assume single contains for now
                    childrenDns = self.findDataNodesInBuilder(select = childType)
                    
                    if len(childrenDns) > 0:       
                        for i, v in enumerate(vInfo.value):
                            _DataNodeBulder__Logger.info('Added dataNodes with indexes from %i to %i of type %s to dataNode %s'%(v[0],v[1],childType,dns1[i]))

                            for _i in range(v[0],v[1]+1):
                                dns1[i].addChildDataNode(childrenDns[_i])
                                           
        elif vInfo.dim == 2:
            _DataNodeBulder__Logger.info('Adding %i dataNodes of type %s'%(vInfo.len,conceptName))
            
            if "relationMode" in conceptInfo:
                relatedDnsType = conceptInfo["relationAttrs"]['src']
                relatedDns = self.findDataNodesInBuilder(select = relatedDnsType)
    
                requiredLenOFReltedDns = len(vInfo.value[0])
                
                if requiredLenOFReltedDns != len(relatedDns):
                    _DataNodeBulder__Logger.error('Number of expected related dataNode %i is different then the number of %i dataNodes of type %s - abandon the update'%(requiredLenOFReltedDns,len(relatedDns),relatedDnsType))
                    return

            for i in range(0,vInfo.len):
                instanceValue = ""
                instanceID = i
                _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                    
                _dn.attributes[keyDataName] = vInfo.value[i]
                dns.append(_dn)
                
                if conceptInfo["relationMode"] == "forward":
                    for index, isRelated in enumerate(vInfo.value[i]):
                        if isRelated == 1:
                            relatedDns[index].addChildDataNode(_dn)                            
                elif conceptInfo["relationMode"] == "backward":
                    for index, isRelated in enumerate(vInfo.value[i]):
                        if isRelated == 1:
                            _dn.addChildDataNode(relatedDns[index])  
                
            self.__updateRootDataNodeList(dns)   

            return dns
        
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
                
            # Check if value has information about children - need to be tuples with two indexes - start index of children  - end index of children for the given new dataNode
            if isinstance(vInfo.value[0][0], tuple): 
                # Add it as parent to existing dataNodes - Backward information in the sensor data
                _DataNodeBulder__Logger.info('The value is build of tuples - providing information about children of the new dataNodes')

                if len(conceptInfo['contains']) > 0:
                    childType = conceptInfo['contains'][0].name # Assume single contains for now
                    childrenDns = self.findDataNodesInBuilder(select = childType)
                    
                    childOffset = 0
                    if len(childrenDns) > 0:   
                        for j, _v  in enumerate(vInfo.value):
                            for i, v in enumerate(_v):
                                _DataNodeBulder__Logger.info('Added dataNodes with indexes from %i to %i of type %s to dataNode %s'%(v[0],v[1],childType,dns[j][i]))
    
                                for _i in range(v[0],v[1]+1):
                                    dns[j][i].addChildDataNode(childrenDns[_i + childOffset])
                                    
                            childOffset =+ _v[-1][1]
        else: # vInfo.dim > 2
            _DataNodeBulder__Logger.warning('Dimension of value %i is larger then 2 -  not supported'%(vInfo.dim))

        _dnLinked = False
            
        # Add them as children to existing dataNodes - Forward information in the sensor data
        for currentParentConcept in conceptInfo['containedIn']:
            currentParentConceptName = currentParentConcept.name
            currentParentDns = self.findDataNodesInBuilder(select = currentParentConceptName)
            
            if currentParentDns:
                if len(currentParentDns) == len(dns):
                    _DataNodeBulder__Logger.info('Adding dataNodes as children to %i dataNodes of type %s'%(len(currentParentDns),currentParentConceptName))
                else:
                    _DataNodeBulder__Logger.error('Number of dataNode sets %i is different then the number of %i dataNodes of type %s - abandon the update'%(len(dns),len(currentParentDns),currentParentConceptName))
                    continue
            else:
                _DataNodeBulder__Logger.info('Not found any dataNode of type %s - this type is a potential parent of the current concept'%(currentParentConceptName))
            
            for currentParentDnIndex, currentParentDn in enumerate(currentParentDns):
                for curentChildDn in dns[currentParentDnIndex]: # Set of new dataNodes for the current parent dataNode
                    currentParentDn.addChildDataNode(curentChildDn)    
                                           
                _DataNodeBulder__Logger.info('Added %i dataNodes %s as children to the dataNode with id %s'%(len(dns[currentParentDnIndex]),dns[currentParentDnIndex],currentParentDn.instanceID))
                _dnLinked = True # New dataNodes are linked with existing dataNodes
                 
        self.__updateRootDataNodeList(dns)   
        
        return dns
                    
    def __updateDataNodes(self, vInfo, conceptInfo, keyDataName):
        conceptName = conceptInfo['concept'].name
        existingDnsForConcept = self.findDataNodesInBuilder(select = conceptName) # Try to get DataNodes of the current concept
            
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
            return self.__createInitialdDataNode(vInfo, conceptInfo, keyDataName) # Done - End the method
        else:
            # ---------- DataNodes already created
            existingDnsForConcept = self.findDataNodesInBuilder(select = conceptName) # Try to get DataNodes of the current concept
            
            if len(existingDnsForConcept) == 0:# Check if datannote for this concept already created                    
                # No Datanode of this concept created yet
    
                # If attribute value is a single element - will create a single new dataNode
                if vInfo.len == 1 and vInfo.dim < 2: 
                    return self.__createSingleDataNode(vInfo, conceptInfo, keyDataName)
                else: # -- Value is multiple elements
                    return self.__createMultiplyDataNode(vInfo, conceptInfo, keyDataName)
            else: # DataNode with this concept already created - update it
                self.__updateDataNodes(vInfo, conceptInfo, keyDataName)
                
    def __addEquality(self, vInfo, conceptInfo, equalityConceptName, keyDataName):
        conceptName = conceptInfo['concept'].name
        existingDnsForConcept = self.findDataNodesInBuilder(select = conceptName)
        existingDnsForEqualityConcept = self.findDataNodesInBuilder(select = equalityConceptName)
        
        if not existingDnsForConcept and not existingDnsForEqualityConcept:
            _DataNodeBulder__Logger.warning('No datNodes created for concept %s and equality concept %s'%(conceptName,equalityConceptName))
            return
        
        if not existingDnsForConcept:
            _DataNodeBulder__Logger.warning('No datNodes created for concept %s'%(conceptName))
            return
        
        if not existingDnsForEqualityConcept:
            _DataNodeBulder__Logger.warning('No datNodes created for equality concept %s'%(equalityConceptName))
            return
        
        _DataNodeBulder__Logger.info('Added equality between dataNodes of types %s and %s'%(conceptName,equalityConceptName))

        for conceptDn in existingDnsForConcept:
            for equalDn in existingDnsForEqualityConcept:
                
                if conceptDn.getInstanceID() >= vInfo.value.shape[0]:
                    continue
                
                if equalDn.getInstanceID() >= vInfo.value.shape[1]:
                    continue
                
                if vInfo.value[conceptDn.getInstanceID(), equalDn.getInstanceID()]:
                    _DataNodeBulder__Logger.info('DataNodes of %s is equal to %s'%(conceptDn,equalDn))
                    conceptDn.addEqualTo(equalDn)

    # Method processing value of for the attribute - determining it it should be treated as a single element. 
    # It returns a tuple with elements specifying the length of the first dimension of the value, the number of dimensions of the value and the original value itself
    def __processAttributeValue(self, value, keyDataName):
        ValueInfo = namedtuple('ValueInfo', ["len", "value", 'dim'])

        if not isinstance(value, (Tensor, list)): # It is scalar value
            return ValueInfo(len = 1, value = value, dim=0) 
            
        if isinstance(value, Tensor) and value.dim() == 0: # It is a Tensor but also scalar value
            return ValueInfo(len = 1, value = value.item(), dim=0)
        
        if (len(value) == 1): # It is Tensor or list with length 1 - treat it as scalar
            if isinstance(value, list) and not isinstance(value[0], (Tensor, list)) : # Unpack the value
                return ValueInfo(len = 1, value = value[0], dim=0)
            elif isinstance(value, Tensor) and value.dim() < 2:
                return ValueInfo(len = 1, value = torch.squeeze(value, 0), dim=0)

        #  If it is Tensor or list with length 2 but it is for attribute providing probabilities - assume it is a scalar value
        if len(value) ==  2 and keyDataName[0] == '<': 
            return ValueInfo(len = 1, value = value, dim=0)
        
        if isinstance(value, list): 
            if not isinstance(value[0], (Tensor, list)) or (isinstance(value[0], Tensor) and value[0].dim() == 0):
                return ValueInfo(len = len(value), value = value, dim=1)
            elif not isinstance(value[0][0], (Tensor, list)) or (isinstance(value[0][0], Tensor) and value[0][0].dim() == 0):
                return ValueInfo(len = len(value), value = value, dim=2)
            elif not isinstance(value[0][0][0], (Tensor, list)) or (isinstance(value[0][0][0], Tensor) and value[0][0][0].dim() == 0):
                return ValueInfo(len = len(value), value = value, dim=3)
            else:
                _DataNodeBulder__Logger.warning('Dimension of nested list value for key %s is more then 3 returning dimension 4'%(keyDataName))
                return ValueInfo(len = len(value), value = value, dim=4)

        elif isinstance(value, Tensor):
            return ValueInfo(len = len(value), value = value, dim=value.dim())
    
    # Overloaded __setitem method of Dictionary - tracking sensor data and building corresponding data graph
    def __setitem__(self, _key, value):
        from ..sensor import Sensor

        start = time.time()
        self.__addSetitemCounter()
        
        if isinstance(_key, (Sensor, Property, Concept)):
            key = _key.fullname
            if  isinstance(_key, Sensor) and not _key.build:
                if isinstance(value, Tensor):
                    _DataNodeBulder__Logger.info('No processing (because build is set to False) - key - %s, key type - %s, value - %s, shape %s'%(key,type(_key),type(value),value.shape))
                elif isinstance(value, list):
                    _DataNodeBulder__Logger.info('No processing (because build is set to False) - key - %s, key type - %s, value - %s, length %s'%(key,type(_key),type(value),len(value)))
                else:
                    _DataNodeBulder__Logger.info('No processing (because build is set to False) - key - %s, key type - %s, value - %s'%(key,type(_key),type(value)))

                return dict.__setitem__(self, _key, value)
            
            if  isinstance(_key, Property):
                if isinstance(value, Tensor):
                    _DataNodeBulder__Logger.info('No processing Property as key - key - %s, key type - %s, value - %s, shape %s'%(key,type(_key),type(value),value.shape))
                elif isinstance(value, list):
                    _DataNodeBulder__Logger.info('No processing Property as key - key - %s, key type - %s, value - %s, length %s'%(key,type(_key),type(value),len(value)))
                else:
                    _DataNodeBulder__Logger.info('No processing Property as key - key - %s, key type - %s, value - %s'%(key,type(_key),type(value)))

                return dict.__setitem__(self, _key, value)
        elif isinstance(_key, str):
            key = _key
        else:
            _DataNodeBulder__Logger.error('key - %s, type %s is not supported'%(_key,type(_key)))
            return
        
        skey = key.split('/')
        
        # Check if the key with this value has been set recently
        # If not create a new sensor for it
        # If yes stop __setitem__ and return - the same value for the key was added last time that key was set
        if self.__addSensorCounters(skey, value):
            return # Stop __setitem__ for repeated key value combination
        
        if isinstance(value, Tensor):
            _DataNodeBulder__Logger.info('key - %s, key type - %s, value - %s, shape %s'%(key,type(_key),type(value),value.shape))
        elif isinstance(value, list):
            _DataNodeBulder__Logger.info('key - %s, key type - %s, value - %s, length %s'%(key,type(_key),type(value),len(value)))
        else:
            _DataNodeBulder__Logger.info('key - %s, key type - %s, value - %s'%(key,type(_key),type(value)))

        if value is None:
            _DataNodeBulder__Logger.error('The value for the key %s is None - abandon the update'%(key))
            return dict.__setitem__(self, _key, value)
                
        if len(skey) < 2:            
            _DataNodeBulder__Logger.error('The key %s has only two elements, needs at least three - abandon the update'%(key))
            return dict.__setitem__(self, _key, value)
        
        usedGraph = dict.__getitem__(self, "graph")

        # Find if the key include concept from graph
        
        graphPathIndex = usedGraph.cutGraphName(skey)
        keyWithoutGraphName = skey[graphPathIndex:]
        graphPath =  ''.join(map(str, skey[:graphPathIndex])) 
       
        # Check if found concept in the key
        if not keyWithoutGraphName:
            _DataNodeBulder__Logger.warning('key - %s has not concept part - returning'%(key))
            return dict.__setitem__(self, _key, value)
 
        # Find description of the concept in the graph
        if isinstance(_key, Sensor):
            try:
                _conceptName = _key.concept.name 
            except TypeError as _:
                _conceptName = keyWithoutGraphName[0]
        else:
            _conceptName = keyWithoutGraphName[0]
        concept = self.__findConcept(_conceptName, usedGraph)
                
        if not concept:
            _DataNodeBulder__Logger.warning('_conceptName - %s has not been found in the used graph %s - returning'%(_conceptName,usedGraph.fullname))
            return dict.__setitem__(self, _key, value)
        
        conceptInfo = self.__findConceptInfo(usedGraph, concept)
        
        if isinstance(_key, Sensor):
            self.__updateConceptInfo(usedGraph, conceptInfo, _key)

        # Create key for dataNonde construction
        keyDataName = "".join(map(lambda x: '/' + x, keyWithoutGraphName[1:-1]))
        keyDataName = keyDataName[1:] # __cut first '/' from the string
        
        vInfo = self.__processAttributeValue(value, keyDataName)
        
        # Decide if this is equality between concept data, dataNode creation or update for concept or relation link
        if keyDataName.find("_Equality_") > 0:
            equalityConceptName = keyDataName[keyDataName.find("_Equality_") + len("_Equality_"):]
            self.__addEquality(vInfo, conceptInfo, equalityConceptName, keyDataName)
        
        elif conceptInfo['relation']:
            _DataNodeBulder__Logger.debug('%s found in the graph; it is a relation'%(_conceptName))
            self.__buildRelationLink(vInfo, conceptInfo, keyDataName) # Build or update relation link
            
        else:                       
            _DataNodeBulder__Logger.debug('%s found in the graph; it is a concept'%(_conceptName))
            index = self.__buildDataNode(vInfo, conceptInfo, keyDataName)   # Build or update Data node
            
            if index:
                indexKey = graphPath  + '/' +_conceptName + '/index'
                dict.__setitem__(self, indexKey, index)
            
        # Add value to the underling dictionary
        r = dict.__setitem__(self, _key, value)
        
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
            
    def findDataNodesInBuilder(self, select = None, indexes = None):
        existingRootDns = dict.__getitem__(self, 'dataNode') # Datanodes roots
        foundDns = existingRootDns[0].findDatanodes(dns = existingRootDns, select = select, indexes = indexes) 
        
        return foundDns
    
    # Method returning constructed dataNode - the fist in the list
    def getDataNode(self):
        self.__addGetDataNodeCounter()
        
        if dict.__contains__(self, 'dataNode'):
            _dataNode = dict.__getitem__(self, 'dataNode')
            
            if len(_dataNode) > 0:  
                returnDn = _dataNode[0]
                
                if len(_dataNode) == 1:
                    _DataNodeBulder__Logger.info('Returning dataNode with id %s of type %s'%(returnDn.instanceID,returnDn.getOntologyNode()))
                else:
                    _DataNodeBulder__Logger.warning('Returning first dataNode with id %s of type %s - there are total %i dataNodes'%(returnDn.instanceID,returnDn.getOntologyNode(),len(_dataNode)))

                return returnDn
        
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