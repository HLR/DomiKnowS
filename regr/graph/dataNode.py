import torch
from collections import OrderedDict, namedtuple
import time
import re
from .dataNodeConfig import dnConfig 
from torch.tensor import Tensor
import graphviz

from regr.graph.logicalConstrain import eqL
from regr.solver import ilpOntSolverFactory

import logging
from logging.handlers import RotatingFileHandler
from .property import Property
from .concept import Concept, EnumConcept
# from _pytest.reports import _R

import graphviz

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

    def visualize(self, filename: str, inference_mode="ILP", include_legend=False, open_image=False):
        if include_legend:
            # Build Legend subgraph
            legend = graphviz.Digraph(name='cluster_legend',comment='Legend')
            legend.attr('node', shape='rectangle')
            legend.attr(label="Legend")
            legend.node('Attribute')

            legend.attr('node', shape='diamond')
            legend.node('Decision')

            legend.attr('node', shape='oval')
            legend.node('Concept')
        # ----
        g = graphviz.Digraph(name='cluster_main')

        # Root node
        root_id = self.ontologyNode.name
        g.node(root_id)

        g.attr('node', shape = 'rectangle')
        for attribute_name, attribute in self.getAttributes().items():
            # Visualize all attributes which are not a relation
            attr_node_id = str(attribute_name)

            if attribute_name.endswith('.reversed'):
                continue
            elif re.match(r'^<.*>$', attribute_name):
                if attribute.shape[0] != 2:
                    print('WARNING: We currently only support visualization for binary decisions.')
                    continue

                label = self.getAttribute(f'{attribute_name}/label').item()
                if inference_mode.lower() == "ilp":
                    prediction = self.getAttribute(f"{attribute_name}/ILP")
                    if prediction is None:
                        raise Exception(f'inference_mode=\"{inference_mode}\" could not be found in the DataNode')
                else:
                    # Extract decision
                    decisions = self.getAttribute(f"{attribute_name}/local/{inference_mode}")
                    if decisions is None:
                        raise Exception(f'inference_mode=\"{inference_mode}\" could not be found in the DataNode')
                    prediction = decisions[1]

                g.attr('node', shape='diamond')
                g.node(attr_node_id, f'{attribute_name[1:-1]}\nlabel={label}\nprediction={prediction.item():.2f}')
                g.edge(root_id, attr_node_id)
                g.attr('node', color='black')
            elif re.match(r'^<.*>(/.*)+', attribute_name):
                #print(f'Filtered {attribute_name}')
                continue
            else:
                # Normal nodes
                g.attr('node', shape='rectangle')
                
                # Format attribute
                attr_str = str(attribute)
                if isinstance(attribute, Tensor):
                    attr_str = f'<tensor of shape {list(attribute.shape)}>'

                g.node(attr_node_id, f'{attribute_name}: {attr_str}')
                g.edge(root_id, attr_node_id)

        main_graph = graphviz.Digraph()
        if include_legend:
            main_graph.subgraph(legend)
        main_graph.subgraph(g)

        main_graph.render(filename, format='png', view=open_image)
    
    # --- Attributes methods
    
    def getAttributes(self):
        return self.attributes     
    
    def getAttribute(self, *keys):
        key = ""
        keyBis  = ""
        index = None
        
        for i, k in enumerate(keys):
            if key != "":
                key = key + "/"
                keyBis = keyBis + "/"
                
            if isinstance(k, str):
                _k = self.findConcept(k)
                
                if _k is not None:  
                    if isinstance(_k, tuple):
                        key = key + '<' + _k[0].name +'>'
                        index = _k[2]
                        keyBis = keyBis + k
                    else:
                        key = key + '<' + k +'>'
                        keyBis = keyBis + k
                else:
                    key = key + k
                    keyBis = keyBis + k
            elif isinstance(k, tuple):
                key = key + '<' + k[0].name +'>'
                keyBis = keyBis + k[0].name
            elif isinstance(k, Concept):
                key = key + '<' + k.name +'>'
                keyBis = keyBis + k.name
                    
        if key in self.attributes:
            if index is None:
                return self.attributes[key]
            else:
                return self.attributes[key][index]
        elif keyBis in self.attributes:
            if index is None:
                return self.attributes[keyBis]
            else:
                return self.attributes[keyBis][index]
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
    
    # Find dataNodes in data graph for the given concept 
    def findDatanodes(self, dns = None, select = None, indexes = None, visitedDns = None, depth = 0):
        # If no DataNodes provided use self
        if not depth and dns is None:
            dns = [self]
            
        returnDns = []
        
        # If empty list of provided DataNodes then return - it is a recursive call with empty list
        if dns is None or len(dns) == 0:
            return returnDns
        
        # No select provided - query not defined - return
        if select is None:
            if depth == 0 and not returnDns:
                _DataNode__Logger.warning('Not found any DataNode - no value for the select part of query provided')
                
            return returnDns
       
        # Check each provided DataNode if it satisfy the select part of the query  
        for dn in dns:
            # Test current DataNote against the query
            if self.__testDataNode(dn, select):
                if dn not in returnDns:
                    returnDns.append(dn) 
                            
            if not visitedDns:
                visitedDns = set()
                             
            visitedDns.add(dn)
                    
        # Call recursively
        newDepth = depth + 1
        for dn in dns:
            # Visit  DataNodes in relations
            for r, rValue in dn.getRelationLinks().items():            
               
                # Check if the nodes already visited
                dnsToVisit = set()
                for rDn in rValue:
                    if rDn not in visitedDns:
                        dnsToVisit.add(rDn)
                    
                if not dnsToVisit:
                    continue
                
                # Visit DataNodes in the current relation
                _returnDns = self.findDatanodes(dnsToVisit, select = select, indexes = indexes, visitedDns = visitedDns, depth = newDepth)
        
                if _returnDns is not None:
                    for _dn in _returnDns:
                        if _dn not in returnDns:
                            returnDns.append(_dn)

        if depth: # Finish recursion
            return returnDns
        
        # If index provided in query then filter the found results for the select part of query through the index part of query
        if (indexes != None):
            _returnDns = [] # Will contain results from returnDns satisfying the index
            
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
        
        # If not fund any results
        if depth == 0 and not returnDns:
            _DataNode__Logger.debug('Not found any DataNode for - %s -'%(select))
    
        # Sort results according to their ids
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
    def findConcept(self, _conceptName, usedGraph = None):
        if isinstance(_conceptName, Concept):
            _conceptName = _conceptName.name()
            
        if not usedGraph:
            usedGraph = self.ontologyNode.getOntologyGraph()
            
        subGraph_keys = [key for key in usedGraph._objs]
        for subGraphKey in subGraph_keys:
            subGraph = usedGraph._objs[subGraphKey]
            
            for conceptNameItem in subGraph.concepts:
                if _conceptName == conceptNameItem:
                    concept = subGraph.concepts[conceptNameItem]
                    
                    return (concept, concept.name, None, 1)
                elif isinstance(subGraph.concepts[conceptNameItem], EnumConcept):
                    vlen = len(subGraph.concepts[conceptNameItem].values)
                    
                    if _conceptName in subGraph.concepts[conceptNameItem].values:
                        concept = subGraph.concepts[conceptNameItem]
                        
                        return (concept, _conceptName, subGraph.concepts[conceptNameItem].get_index(_conceptName), vlen)
        
        return None 
    
    # Check if concept is relation
    def isRelation(self, conceptRelation, usedGraph = None):
        if usedGraph is None:
            usedGraph = self.ontologyNode.getOntologyGraph()
            
        if len(conceptRelation.has_a()) > 0:  
            return True
        
        for _isA in conceptRelation.is_a():
            _conceptRelation = _isA.dst
            
            if self.__isRelation(_conceptRelation, usedGraph):
                return True
        
        return False 
    
    def getRelationAttrNames(self, conceptRelation, usedGraph = None):
        if usedGraph is None:
            usedGraph = self.ontologyNode.getOntologyGraph()
            
        if len(conceptRelation.has_a()) > 0:  
            relationAttrs = OrderedDict()
            for _, rel in enumerate(conceptRelation.has_a()): 
                dstName = rel.dst.name                
                relationAttr = self.findConcept(dstName, usedGraph)[0]
    
                relationAttrs[rel.name] = relationAttr
                
            return relationAttrs
        
        for _isA in conceptRelation.is_a():
            _conceptRelation = _isA.dst
            
            resultForCurrent = self.__getRelationAttrNames(_conceptRelation, usedGraph)
            
            if bool(resultForCurrent):
                return resultForCurrent
        
        return None 
    
    # Find the root parent of relation of the given relation
    def findRootConceptOrRelation(self, relationConcept, usedGraph = None):
        if usedGraph is None:
            usedGraph = self.ontologyNode.getOntologyGraph()
        
        if isinstance(relationConcept, str):
            relationConcept = self.findConcept(relationConcept)[0]
            
        # Does this concept or relation has parent (through _isA)
        for _isA in relationConcept.is_a():
            _relationConcept = _isA.dst
            
            return  self.findRootConceptOrRelation(_relationConcept, usedGraph)
        
        # If the provided concept or relation is root (has not parents)
        return relationConcept 

    # Find Datanodes starting from the given Datanode following provided path
    # path can contain eqL statement selecting Datanodes from the datanodes collecting on the path
    def getEdgeDataNode(self, path):
        # Path is empty
        if len(path) == 0:
            return [self]

        # Path has single element
        if len(path) == 1:
            if isinstance(path[0], str):
                path0 = path[0]
            else:
                path0 = path[0].name
                
            if path0 in self.relationLinks:
                return self.relationLinks[path0]
            else:
                return [None]
        
        # Path has at least 2 elements - will perfomr recursion
        if isinstance(path[0], eqL): # check if eqL
            concept = path[0].e[0][0]
        elif path[0] in self.relationLinks:
            concept = path[0]
        else:
            return [None]

        if isinstance(concept, Concept):
            concept = concept.name
            
        if concept in self.relationLinks:
            cDns = self.relationLinks[concept]
        else:
            cDns = []
            
        # Filter DataNoded through eqL
        if isinstance(path[0], eqL):
            _cDns = []
            for cDn in cDns:
                if isinstance(path[0].e[1], str):
                    path0e1 = path[0].e[1]
                else:
                    path0e1 = path[0].e[1].name
                    
                if path0e1 in cDn.attributes and cDn.attributes[path0e1].item() in path[0].e[2]:
                    _cDns.append(cDn)
                    
            cDns = _cDns
        
        # recursion
        rDNS = []
        for cDn in cDns:
            rDn = cDn.getEdgeDataNode(path[1:])
            
            if rDn:
                rDNS.extend(rDn)
                
        if rDNS:
            return rDNS
        else:
            return [None]

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
                                
    def collectConceptsAndRelations(self, dn, conceptsAndRelations = set()):
        candR = self.__collectConceptsAndRelations(self)
        
        returnCandR = []
        
        for c in candR:
            _concept = self.findConcept(c)[0]
            
            if _concept is None:
                continue
            
            if isinstance(_concept, tuple):
                _concept = _concept[0]
            
            if isinstance(_concept, EnumConcept):
                for i, a in enumerate(_concept.values):
                    
                    if conceptsAndRelations and a not in conceptsAndRelations:
                        continue
                    
                    returnCandR.append((_concept, a, i, len(_concept.values)))
            else:
                if conceptsAndRelations and c not in conceptsAndRelations and _concept not in conceptsAndRelations:
                    continue
                
                returnCandR.append((_concept, _concept.name, None, 1))
        
        return returnCandR
        
    def __getILPsolver(self, conceptsRelations = []):
        
        _conceptsRelations = []
        
        # Get ontology graphs and then ilpOntsolver
        myOntologyGraphs = {self.ontologyNode.getOntologyGraph()}
        
        for currentConceptOrRelation in conceptsRelations:
            if isinstance(currentConceptOrRelation, str):
                currentConceptOrRelation = self.findConcept(currentConceptOrRelation)
            
            _conceptsRelations.append(currentConceptOrRelation)
            
            if isinstance(currentConceptOrRelation, tuple):
                currentOntologyGraph = currentConceptOrRelation[0].getOntologyGraph()
            else:
                currentOntologyGraph = currentConceptOrRelation.getOntologyGraph()
            
            if currentOntologyGraph is not None:
                myOntologyGraphs.add(currentOntologyGraph)
                
        myilpOntSolver = ilpOntSolverFactory.getOntSolverInstance(myOntologyGraphs)
        
        if not myilpOntSolver:
            _DataNode__Logger.error("ILPSolver not initialized")
            raise DataNode.DataNodeError("ILPSolver not initialized")
        
        return myilpOntSolver, _conceptsRelations
    
    #----------------- Solver methods

    # Collect inferred results of the given type (e.g. ILP, softmax, argmax, etc) from the given concept
    def collectInferedResults(self, concept, inferKey):
        collectAttributeList = []
        
        if not isinstance(concept, tuple):
            if not isinstance(concept, Concept):
                concept = self.findConcept(concept)
                if concept is None:
                    return torch.tensor(collectAttributeList)
                
            if isinstance(concept, EnumConcept):
                concept = (concept, concept.name, None, len(concept.values))
            else:
                concept = (concept, concept.name, None, 1)

            
        rootConcept = self.findRootConceptOrRelation(concept[0])
        
        if not rootConcept:
            return torch.tensor(collectAttributeList)
        
        rootConceptDns = self.findDatanodes(select = rootConcept)
        
        if not rootConceptDns:
            return torch.tensor(collectAttributeList)

        keys = [concept, inferKey]
        
        for dn in rootConceptDns:
            rTensor = dn.getAttribute(*keys)
            if rTensor is None:
                continue
            
            if torch.is_tensor(rTensor):
                if len(rTensor.shape) == 0 or len(rTensor.shape) == 1 and  rTensor.shape[0] == 1:
                    collectAttributeList.append(rTensor.item())
                elif (concept[2] is None) and concept[3] == 1:
                    collectAttributeList.append(rTensor[1])
                elif concept[2] is not None:
                    collectAttributeList.append(rTensor[concept[2]])
                elif (concept[2] is None) and concept[3] > 1:
                    return rTensor
            elif rTensor:
                collectAttributeList.append(1)
            else:
                collectAttributeList.append(0)
        
        return torch.tensor(collectAttributeList)        
    
    # Calculate argMax and softMax
    def infer(self):
        conceptsRelations = self.collectConceptsAndRelations(self) 
        
        for c in conceptsRelations:
            cRoot = self.findRootConceptOrRelation(c[0])
            dns = self.findDatanodes(select = cRoot)
            
            if not dns:
                continue
            
            vs = []
            
            for dn in dns:
                v = dn.getAttribute(c[0])
                
                if v is None:
                    vs = []
                    break
                elif not torch.is_tensor(v):
                    vs = []
                    break
                else:
                    if c[2] is not None:
                        vs.append(v[c[2]])
                    else:
                        if len(v.size()) != 1 or v.size()[0] != 2:
                            vs = []
                            break
                        else:
                            vs.append(v[1])
            
            if not vs:
                continue
            
            t = torch.tensor(vs)
            t[torch.isnan(t)] = 0 # NAN  -> 0
            
            vM = torch.argmax(t).item() # argmax
            
            # Elements for softmax
            tExp = torch.exp(t)
            tExpSum = torch.sum(tExp).item()
            
            keyArgmax = "<" + c[0].name + ">/argmax"
            keySoftMax = "<" + c[0].name + ">/softmax"
            
            # Add argmax and softmax to DataNodes
            for dn in dns:    
                if keyArgmax not in dn.attributes:
                    dn.attributes[keyArgmax] = torch.empty(c[3], dtype=torch.float)
                    
                if dn.getInstanceID() == vM:
                    dn.attributes[keyArgmax][c[2]] = 1
                else:
                    dn.attributes[keyArgmax][c[2]] = 0
                
                if keySoftMax not in dn.attributes:
                    dn.attributes[keySoftMax] = torch.empty(c[3], dtype=torch.float)
                    
                dnSoftmax = tExp[dn.getInstanceID()]/tExpSum
                dn.attributes[keySoftMax][c[2]] = dnSoftmax.item()

    # Calculate local for datanote argMax and softMax
    def inferLocal(self):
        conceptsRelations = self.collectConceptsAndRelations(self) 
        
        for c in conceptsRelations:
            cRoot = self.findRootConceptOrRelation(c[0])
            dns = self.findDatanodes(select = cRoot)
            
            if not dns:
                continue
            
            vs = []
            
            for dn in dns:
                keySoftmax = "<" + c[0].name + ">/local/softmax"
                if keySoftmax in dn.attributes:
                    continue
                
                v = dn.getAttribute(c[0])
                
                if v is None:
                    continue
                elif not torch.is_tensor(v):
                    continue
                
                vClone = torch.clone(v).double()
                tExp = torch.exp(vClone)
                for i, e in enumerate(tExp):
                    if e == float("inf"):
                        tExp[i] = 1.0
                        
                tExpSum = torch.sum(tExp).item()
            
                vSoftmax = [(tExp[i]/tExpSum).item() for i in range(len(v))]
                    
                vSoftmaxT = torch.as_tensor(vSoftmax) 
                
                # Replace nan with 1/len
                #for i, s in enumerate(vSoftmaxT):
                #   if s != s:
                #       vSoftmaxT[i] = 1/len(v)
                
                dn.attributes[keySoftmax] = vSoftmaxT
                
                keyArgmax  = "<" + c[0].name + ">/local/argmax"
                vArgmax = torch.clone(v)
                vArgmaxIndex = torch.argmax(v).item()
                
                for i, _ in enumerate(v):
                    if i == vArgmaxIndex:
                        vArgmax[i] = 1
                    else:
                        vArgmax[i] = 0
                                
                dn.attributes[keyArgmax] = vArgmax
        
    # Calculate ILP prediction for data graph with this instance as a root based on the provided list of concepts and relations
    def inferILPResults(self, *_conceptsRelations, fun=None, epsilon = 0.00001, minimizeObjective = False):
        if len(_conceptsRelations) == 0:
            _DataNode__Logger.info('Called with empty list of concepts and relations for inference')
        else:
            _DataNode__Logger.info('Called with - %s - list of concepts and relations for inference'%([x.name if isinstance(x, Concept) else x for x in _conceptsRelations]))
            
        # Check if concepts and/or relations have been provided for inference
        _conceptsRelations = self.collectConceptsAndRelations(self, _conceptsRelations) # Collect all concepts and relation from graph as default set

        if len(_conceptsRelations) == 0:
            _DataNode__Logger.error('Not found any concepts or relations for inference in provided DataNode %s'%(self))
            raise DataNode.DataNodeError('Not found any concepts or relations for inference in provided DataNode %s'%(self))
        else:        
            _DataNode__Logger.info('Found - %s - as a set of concepts and relations for inference'%([x[0].name if isinstance(x, tuple) else x for x in _conceptsRelations]))
                
        myilpOntSolver, conceptsRelations = self.__getILPsolver(_conceptsRelations)
        
        # Call ilpOntsolver with the collected probabilities for chosen candidates
        _DataNode__Logger.info("Calling ILP solver")
        
        self.inferLocal()
        myilpOntSolver.calculateILPSelection(self, *conceptsRelations, fun=None, epsilon = 0.00001, minimizeObjective = minimizeObjective)    
        
    def verifySelection(self, *_conceptsRelations):
        
        # --- Update !
        
        if not _conceptsRelations:
            _conceptsRelations = ()
            
        myilpOntSolver, infer_candidatesID, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForTripleRelations, hardConstrains, candidates_currentConceptOrRelation = \
            self.__prepareILPData(*_conceptsRelations, dnFun = self.__getLabel)
            
        if not myilpOntSolver:
            return False
        
        verifyResult = myilpOntSolver.verifySelection(infer_candidatesID, graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForTripleRelations, hardConstrains=hardConstrains)
        
        return verifyResult
    
    def calculateLcLoss(self):
        
        myilpOntSolver, _ = self.__getILPsolver(conceptsRelations = self.collectConceptsAndRelations(self))

        self.inferLocal()
        lcResult = myilpOntSolver.calculateLcLoss(self)
        
        return lcResult

    def getInferMetric(self, *conceptsRelations, inferType='ILP', weight = None):
               
        if weight is None:
            weight = torch.tensor(1)
                 
        if not conceptsRelations:
            conceptsRelations = self.collectConceptsAndRelations(self, conceptsRelations) # Collect all concepts and relation from graph as default set
        
        result = {}
        tp, fp, tn, fn  = [], [], [], []

        for cr in conceptsRelations:
            if not isinstance(cr, tuple):
                if not isinstance(cr, Concept):
                    cr = self.findConcept(cr)
                    
                    if cr is None:
                        continue
                
                if isinstance(cr, EnumConcept):
                    cr = (cr, cr.name, None, len(cr.values))
                else:
                    cr = (cr, cr.name, None, 1)
            
            
            preds = self.collectInferedResults(cr, inferType)
            labelsR = self.collectInferedResults(cr, 'label')
            
            labels = torch.clone(labelsR)
            
            if cr[2] is not None:
                for i, l in enumerate(labelsR):
                    if labelsR[i] == cr[2]:
                        labels[i] = 1
                    else:
                        labels[i] = 0
            elif (cr[2] is None) and cr[3] > 1:
                labels = torch.clone(preds)
                l = labelsR.item()
                for i, _ in enumerate(preds):
                    if i == l:
                        labels[i] = 1
                    else:
                        labels[i] = 0
            
            result[cr[1]] = {'TP': torch.tensor(0.), 'FP': torch.tensor(0.), 'TN': torch.tensor(0.), 'FN': torch.tensor(0.)}
            
            if preds is None or labels is None:
                continue
            
            if not torch.is_tensor(preds) or not torch.is_tensor(labels):
                continue
            
            if preds.dim() != 1 or labels.dim() != 1:
                continue
            
            if  preds.size()[0] != labels.size()[0]:
                continue
            
            labels = labels.long()
            # calculate confusion matrix
            _tp = (preds * labels * weight).sum() # true positive
            tp.append(_tp)
            result[cr[1]]['TP'] = _tp 
            _fp = (preds * (1 - labels) * weight).sum() # false positive
            fp.append(_fp)
            result[cr[1]]['FP'] = _fp
            _tn = ((1 - preds) * (1 - labels) * weight).sum() # true negative
            tn.append(_tn)
            result[cr[1]]['TN'] = _tn
            _fn = ((1 - preds) * labels * weight).sum() # false positive
            fn.append(_fn)
            result[cr[1]]['FN'] = _fn
            
            _p, _r  = 0, 0
            
            if _tp + _fp:
                _p = _tp / (_tp + _fp) # precision or positive predictive value (PPV)
                result[cr[1]]['P'] = _p
            if _tp + _fn:
                _r = _tp / (_tp + _fn) # recall, sensitivity, hit rate, or true positive rate (TPR)
                result[cr[1]]['R'] = _r
                
            if _p + _r:
                _f1 = 2 * _p * _r / (_p + _r) # F1 score is the harmonic mean of precision and recall
                result[cr[1]]['F1'] = _f1
            elif _tp + (_fp + _fn)/2: 
                _f1 = _tp/(_tp + (_fp + _fn)/2)
                result[cr[1]]['F1'] = _f1
                      
        result['Total'] = {}  
        tpT = (torch.tensor(tp)).sum()
        result['Total']['TP'] = tpT 
        fpT = (torch.tensor(fp)).sum() 
        result['Total']['FP'] = fpT
        tnT = (torch.tensor(tn)).sum() 
        result['Total']['TN'] = tnT
        fnT = (torch.tensor(fn)).sum() 
        result['Total']['FN'] = fnT
        
        if tpT + fpT:
            pT = tpT / (tpT + fpT)
            result['Total']['P'] = pT
            rT = tpT / (tpT + fnT)
            result['Total']['R'] = rT
            
            if pT + rT:
                f1T = 2 * pT * rT / (pT + rT)
                result['Total']['F1'] = f1T
            elif tpT + (fpT + fnT)/2:
                f1T = tpT/(tpT + (fpT + fnT)/2)
                result['Total']['F1'] = f1T
                
        return result
    
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
                conceptInfo['contains'].append(_contain.dst)
                
        conceptInfo['containedIn'] = []  
        # Check if concept is contained in other concepts
        if ('contains' in concept._in):
            for _contain in concept._in['contains']:
                conceptInfo['containedIn'].append(_contain.src)
                           
        return conceptInfo
            
    def __updateConceptInfo(self,  usedGraph, conceptInfo, sensor):
        from regr.sensor.pytorch.relation_sensors import EdgeSensor
        conceptInfo["relationAttrData"] = False
        conceptInfo['label'] = False
        if hasattr(sensor, 'label') and sensor.label: 
            conceptInfo['label'] = True

        if (isinstance(sensor, EdgeSensor)):
            
            conceptInfo['relationName'] = sensor.relation.name
            conceptInfo['relationTypeName'] = str(type(sensor.relation))
                    
            if 'relationAttrs' in conceptInfo:
                conceptInfo['relationAttrsGraph'] = conceptInfo['relationAttrs']
                
            conceptInfo['relationAttrs'] = {}
          
            conceptInfo['relationMode'] = sensor.relation.mode
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
            if not dnE.impactLinks: 
                if dnE not in newDnsRoots:
                    newDnsRoots.append(dnE)    

        # Add new dataNodes which are not connected to other dataNodes to the root list
        flattenDns = [] # first flatten the list of new dataNodes
        if isinstance(dns[0], list): 
            for dList in dns:
                flattenDns.extend(dList)
                
            if flattenDns and isinstance(flattenDns[0], list): 
                _flattenDns = []
                
                for dList in flattenDns:
                    _flattenDns.extend(dList)
                    
                flattenDns = _flattenDns  
        else:
            flattenDns = dns
            
        for dnE in flattenDns: # review them if they got connected
            if not dnE.impactLinks: 
                if dnE not in newDnsRoots:
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
        
        existingDnsForRelationNotSorted = OrderedDict()
        for dn in existingDnsForRelation:
            existingDnsForRelationNotSorted[dn.getInstanceID()] = dn
                
        existingDnsForRelationSorted = OrderedDict(sorted(existingDnsForRelationNotSorted.items()))
            
        # This is an infromation about relation attributes
        if conceptInfo['relationAttrData']:
            index = keyDataName.index('.')
            attrName = keyDataName[0:index]
            
            relationAttrsCacheName = conceptInfo['concept'].name + "RelationAttrsCache"
            
            if not dict.__contains__(self, relationAttrsCacheName):
                dict.__setitem__(self, relationAttrsCacheName, {})
        
            relationAttrsCache =  dict.__getitem__(self, relationAttrsCacheName)
            relationAttrsCache[attrName] = vInfo.value
                
            _DataNodeBulder__Logger.info('Caching received data for %s related to relation %s dataNode, found %i existing dataNode of this type - provided value has length %i'%(keyDataName,relationName,len(existingDnsForRelation),vInfo.len))
            
            # Find if all the needed attribute were initialized
            allAttrInit = True
            for relationAttributeName, _ in conceptInfo['relationAttrsGraph'].items():
                if relationAttributeName not in relationAttrsCache:
                    allAttrInit = False
                    break
            
            if allAttrInit: #Create links for the ralation DataNode
                # Find DataNodes connected by this relation based on graph definition
                existingDnsForAttr = OrderedDict() # DataNodes for Attributes of the relation
                for relationAttributeName, relationAttributeConcept in conceptInfo['relationAttrsGraph'].items():
                    _existingDnsForAttr = self.findDataNodesInBuilder(select = relationAttributeConcept.name)
                     
                    if _existingDnsForAttr:
                        existingDnsForAttr[relationAttributeName] = _existingDnsForAttr
                        _DataNodeBulder__Logger.info('Found %i dataNodes of the attribute %s for concept %s'%(len(_existingDnsForAttr),relationAttributeName,relationAttributeConcept.name))
                    else:
                        existingDnsForAttr[relationAttributeName] = []
                        _DataNodeBulder__Logger.warning('Not found dataNodes of the attribute %s for concept %s'%(relationAttributeName,relationAttributeConcept.name))
                                    
                attributeNames = [*existingDnsForAttr]
                
                # Create links between this relation and instance dataNode based on the candidate information provided by sensor for each relation attribute
                for relationDnIndex, relationDn in existingDnsForRelationSorted.items():
                    for attributeIndex, attribute in enumerate(attributeNames):
                          
                        candidatesForRelation = relationAttrsCache[attribute][relationDnIndex]
                        
                        for candidateIndex, candidate in enumerate(candidatesForRelation):
                            isInRelation = candidate.item()
                            if isInRelation == 0:
                                continue
                            
                            candidateDn = existingDnsForAttr[attribute][candidateIndex]
                            
                            if attributeIndex == 0:
                                candidateDn.addRelationLink(relationName, relationDn)
                            
                            relationDn.addRelationLink(attribute, candidateDn)  
                            relationDn.attributes[keyDataName] = vInfo.value[relationDnIndex] # Add / /Update value of the attribute
            else:
                # Just add the sensor value to relation DataNodes
                for i, rDn in existingDnsForRelationSorted.items(): # Loop through all relation links dataNodes
                    rDn.attributes[keyDataName] = vInfo.value[i] # Add / /Update value of the attribute

            self.__updateRootDataNodeList(list(existingDnsForRelationSorted.values()))
        else:    
            # -- DataNode with this relation already created  - update it with new attribute value
            _DataNodeBulder__Logger.info('Updating attribute %s in relation link dataNodes %s'%(keyDataName,conceptInfo['concept'].name))
 
            if len(existingDnsForRelation) != vInfo.len:
                _DataNodeBulder__Logger.error('Number of relation is %i and is different then the length of the provided tensor %i'%(len(existingDnsForRelation),vInfo.len))
                return
 
            if len(existingDnsForRelationSorted) == 1:
                if vInfo.dim == 0:
                    existingDnsForRelationSorted[0].attributes[keyDataName] = vInfo.value.item() # Add / /Update value of the attribute
            elif vInfo.dim > 0:
                for i, rDn in existingDnsForRelationSorted.items(): # Loop through all relation links dataNodes
                    rDn.attributes[keyDataName] = vInfo.value[i] # Add / /Update value of the attribute
            else:
                pass

    def __createInitialdDataNode(self, vInfo, conceptInfo, keyDataName):
        conceptName = conceptInfo['concept'].name

        dns = []
                   
        _DataNodeBulder__Logger.info('Creating initial dataNode - provided value has length %i'%(vInfo.len))

        if vInfo.len == 1: # Will use "READER" key as an id of the root dataNode
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
        conceptName = conceptInfo['concept'].name
        _DataNodeBulder__Logger.info('Received information about dataNodes of type %s - value dim is %i and length is %i'%(conceptName,vInfo.dim,vInfo.len))

        # -- Create a single the new dataNode 
        instanceValue = ""
        instanceID = 0
        _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
        _dn.attributes[keyDataName] = vInfo.value
                
        _DataNodeBulder__Logger.info('Single new dataNode %s created'%(_dn))

        self.__updateRootDataNodeList(_dn)
                
        return [_dn]
        
    def __createMultiplyDataNode(self, vInfo, conceptInfo, keyDataName):
        conceptName = conceptInfo['concept'].name
        
        dns = [] # Master List of lists of created dataNodes - each list in the master list represent set of new dataNodes connected to the same parent dataNode (identified by the index in the master list)
                
        _DataNodeBulder__Logger.info('Received information about dataNodes of type %s - value dim is %i and length is %i'%(conceptName,vInfo.dim,vInfo.len))

        if vInfo.dim == 0: 
            _DataNodeBulder__Logger.warning('Provided value is empty %s - abandon the update'%(vInfo.value))
            return
        elif vInfo.dim == 1: # Internal Value is simple; it is not Tensor or list
            _DataNodeBulder__Logger.info('Adding %i new dataNodes of type %s'%(vInfo.len,conceptName))

            dns1 = []
            for vIndex, v in enumerate(vInfo.value):
                instanceValue = ""
                instanceID = vIndex
                _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                
                _dn.attributes[keyDataName] = v
                
                dns1.append(_dn)
                                    
            dns.append(dns1)              
        elif vInfo.dim == 2:
            if "relationMode" in conceptInfo:
                relatedDnsType = conceptInfo["relationAttrs"]['src']
                relatedDns = self.findDataNodesInBuilder(select = relatedDnsType)
                
                if len(vInfo.value) > 0:
                    try:
                        requiredLenOFReltedDns = len(vInfo.value[0])
                    except IndexError:
                        requiredLenOFReltedDns = 0
                else:
                    requiredLenOFReltedDns = 0
                    
                if requiredLenOFReltedDns != len(relatedDns):
                    _DataNodeBulder__Logger.warning('Provided value expects %i related dataNode of type %s but the number of existing dataNodes is %i - abandon the update'%(requiredLenOFReltedDns,relatedDnsType,len(relatedDns)))
                    return
           
                _DataNodeBulder__Logger.info('Create %i new dataNodes of type %s and link them with %i existing dataNodes of type %s with contain relation %s'%(vInfo.len,conceptName,requiredLenOFReltedDns,relatedDnsType,conceptInfo["relationMode"]))

                for i in range(0,vInfo.len):
                    instanceValue = ""
                    instanceID = i
                    _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                        
                    _dn.attributes[keyDataName] = vInfo.value[i]
                    dns.append(_dn)
                    
                    # Create contain relation between the new datanode and existing datanodes
                    if not conceptInfo['relation']:
                        if conceptInfo["relationMode"] == "forward":
                            for index, isRelated in enumerate(vInfo.value[i]):
                                if isRelated == 1:
                                    relatedDns[index].addChildDataNode(_dn)                            
                        elif conceptInfo["relationMode"] == "backward":
                            for index, isRelated in enumerate(vInfo.value[i]):
                                if isRelated == 1:
                                    _dn.addChildDataNode(relatedDns[index])  
            else:
                _DataNodeBulder__Logger.info('Create %i new dataNodes of type %s'%(vInfo.len,conceptName))
                for i in range(0,vInfo.len):
                    instanceValue = ""
                    instanceID = i
                    _dn = DataNode(instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                        
                    dns.append(_dn)
        else:
            _DataNodeBulder__Logger.warning('It is a unsupported sensor ipnut - %s'%(vInfo))
                
        self.__updateRootDataNodeList(dns)   
        return dns
            
    def __updateDataNodes(self, vInfo, conceptInfo, keyDataName):
        conceptName = conceptInfo['concept'].name
        existingDnsForConcept = self.findDataNodesInBuilder(select = conceptName) # Try to get DataNodes of the current concept

        # Check if this is the contain relation update or attribute update
        if "relationMode" in conceptInfo:
            _DataNodeBulder__Logger.info('It is a contain update of type - %s'%(conceptInfo["relationMode"]))

            relatedDnsType = conceptInfo["relationAttrs"]['src']
            relatedDns = self.findDataNodesInBuilder(select = relatedDnsType)

            if vInfo.dim:
                requiredLenOFReltedDns = len(vInfo.value[0])
            else:
                requiredLenOFReltedDns = len(vInfo.item())
            
            if requiredLenOFReltedDns != len(relatedDns):
                _DataNodeBulder__Logger.error('Provided value expected %i related dataNode of type %s but the number of existing dataNodes is %i - abandon the update'%(requiredLenOFReltedDns,relatedDnsType,len(relatedDns)))
                return
                
            for i in range(0,vInfo.len):
                _dn = existingDnsForConcept[i]
                    
                if vInfo.dim == 0:
                    if i == 0:
                        if isinstance(vInfo.value, Tensor):
                            _dn.attributes[keyDataName] =  vInfo.value.item()
                        else:
                            _dn.attributes[keyDataName] =  vInfo.value
                    else:
                        _DataNodeBulder__Logger.error('Provided value %s is a single element value (its dim is 0) but its length is %i'%(vInfo.value, vInfo.len))
                else:
                    _dn.attributes[keyDataName] = vInfo.value[i]
                
                # Create contain relation between existings datanodes
                if not conceptInfo["relation"]:
                    if conceptInfo["relationMode"] == "forward":
                        for index, isRelated in enumerate(vInfo.value[i]):
                            if isRelated == 1:
                                relatedDns[index].addChildDataNode(_dn)                            
                    elif conceptInfo["relationMode"] == "backward":
                        for index, isRelated in enumerate(vInfo.value[i]):
                            if isRelated == 1:
                                _dn.addChildDataNode(relatedDns[index])  
                    
            self.__updateRootDataNodeList(existingDnsForConcept)   
        else: # Attribute update
            if not existingDnsForConcept:
                existingDnsForConcept = self.findDataNodesInBuilder(select = conceptName)
                                                                    
            if keyDataName in existingDnsForConcept[0].attributes:
                _DataNodeBulder__Logger.info('Updating attribute %s in existing dataNodes - found %i dataNodes of type %s'%(keyDataName, len(existingDnsForConcept),conceptName))
            else:
                _DataNodeBulder__Logger.info('Adding attribute %s in existing dataNodes - found %i dataNodes of type %s'%(keyDataName, len(existingDnsForConcept),conceptName))
            
            if len(existingDnsForConcept) > vInfo.len: # Not enough elements in the value 
                _DataNodeBulder__Logger.warning('Provided value has length %i but found %i existing dataNode - abandon the update'%(vInfo.len,len(existingDnsForConcept)))
            elif len(existingDnsForConcept) == vInfo.len: # Number of  value elements matches the number of found dataNodes
                if len(existingDnsForConcept) == 0:
                    return
                elif vInfo.dim == 0:
                    if isinstance(vInfo.value, Tensor):
                        if keyDataName[0] == '<' and keyDataName[-1] == '>':
                            existingDnsForConcept[0].attributes[keyDataName] = [1-vInfo.value.item(), vInfo.value.item()]
                        else:
                            existingDnsForConcept[0].attributes[keyDataName] = vInfo.value
                    else:
                        existingDnsForConcept[0].attributes[keyDataName] = [vInfo.value]
                else:
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

        if isinstance(value, Tensor):
            dimV = value.dim()
            if dimV:
                lenV = len(value)
            else:
                lenV = 1
        else:
            lenV = len(value)
            
        if not isinstance(value, (Tensor, list)): # It is scalar value
            return ValueInfo(len = 1, value = value, dim=0) 
            
        if isinstance(value, Tensor) and dimV == 0: # It is a Tensor but also scalar value
            return ValueInfo(len = 1, value = value.item(), dim=0)
        
        if (lenV == 1): # It is Tensor or list with length 1 - treat it as scalar
            if isinstance(value, list) and not isinstance(value[0], (Tensor, list)) : # Unpack the value
                return ValueInfo(len = 1, value = value[0], dim=0)
            elif isinstance(value, Tensor) and dimV < 2:
                return ValueInfo(len = 1, value = torch.squeeze(value, 0), dim=0)

        #  If it is Tensor or list with length 2 but it is for attribute providing probabilities - assume it is a scalar value
        if isinstance(value, list) and lenV ==  2 and keyDataName[0] == '<': 
            return ValueInfo(lenV = 1, value = value, dim=0)
        elif isinstance(value, Tensor) and lenV ==  2 and dimV  == 0 and keyDataName[0] == '<':
            return ValueInfo(len = 1, value = value, dim=0)

        if isinstance(value, list): 
            if not isinstance(value[0], (Tensor, list)) or (isinstance(value[0], Tensor) and value[0].dim() == 0):
                return ValueInfo(len = lenV, value = value, dim=1)
            elif not isinstance(value[0][0], (Tensor, list)) or (isinstance(value[0][0], Tensor) and value[0][0].dim() == 0):
                return ValueInfo(len = lenV, value = value, dim=2)
            elif not isinstance(value[0][0][0], (Tensor, list)) or (isinstance(value[0][0][0], Tensor) and value[0][0][0].dim() == 0):
                return ValueInfo(len = lenV, value = value, dim=3)
            else:
                _DataNodeBulder__Logger.warning('Dimension of nested list value for key %s is more then 3 returning dimension 4'%(keyDataName))
                return ValueInfo(len = lenV, value = value, dim=4)

        elif isinstance(value, Tensor):
            return ValueInfo(len = lenV, value = value, dim=dimV)
    
    # Overloaded __setitem method of Dictionary - tracking sensor data and building corresponding data graph
    def __setitem__(self, _key, value):
        from ..sensor import Sensor

        start = time.time()
        self.__addSetitemCounter()
        
        if isinstance(_key, (Sensor, Property, Concept)):
            key = _key.fullname
            if  isinstance(_key, Sensor) and not _key.build:
                if isinstance(value, Tensor):
                    _DataNodeBulder__Logger.debug('No processing (because build is set to False) - key - %s, key type - %s, value - %s, shape %s'%(key,type(_key),type(value),value.shape))
                elif isinstance(value, list):
                    _DataNodeBulder__Logger.debug('No processing (because build is set to False) - key - %s, key type - %s, value - %s, length %s'%(key,type(_key),type(value),len(value)))
                else:
                    _DataNodeBulder__Logger.debug('No processing (because build is set to False) - key - %s, key type - %s, value - %s'%(key,type(_key),type(value)))

                return dict.__setitem__(self, _key, value)
            
            if  isinstance(_key, Property):
                if isinstance(value, Tensor):
                    _DataNodeBulder__Logger.debug('No processing Property as key - key - %s, key type - %s, value - %s, shape %s'%(key,type(_key),type(value),value.shape))
                elif isinstance(value, list):
                    _DataNodeBulder__Logger.debug('No processing Property as key - key - %s, key type - %s, value - %s, length %s'%(key,type(_key),type(value),len(value)))
                else:
                    _DataNodeBulder__Logger.debug('No processing Property as key - key - %s, key type - %s, value - %s'%(key,type(_key),type(value)))

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
        
        if conceptInfo['label']:
            keyDataName += '/label'
            
        vInfo = self.__processAttributeValue(value, keyDataName)
        
        # Decide if this is equality between concept data, dataNode creation or update for concept or relation link
        if keyDataName.find("_Equality_") > 0:
            equalityConceptName = keyDataName[keyDataName.find("_Equality_") + len("_Equality_"):]
            self.__addEquality(vInfo, conceptInfo, equalityConceptName, keyDataName)
        
        else:                       
            _DataNodeBulder__Logger.debug('%s found in the graph; it is a concept'%(_conceptName))
            index = self.__buildDataNode(vInfo, conceptInfo, keyDataName)   # Build or update Data node
            
            if index:
                indexKey = graphPath  + '/' +_conceptName + '/index'
                dict.__setitem__(self, indexKey, index)
            
            if conceptInfo['relation']:
                _DataNodeBulder__Logger.debug('%s is a relation'%(_conceptName))
                self.__buildRelationLink(vInfo, conceptInfo, keyDataName) # Build or update relation link
            
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
                    _DataNodeBulder__Logger.info('Returning dataNode with id %s of type %s'%(returnDn.instanceID,returnDn.getOntologyNode().name))
                else:
                    typesInDNs = set()
                    for d in _dataNode:
                        typesInDNs.add(returnDn.getOntologyNode().name)
                        
                    _DataNodeBulder__Logger.warning('Returning first dataNode with id %s of type %s - there are total %i dataNodes of types %s'%(returnDn.instanceID,returnDn.getOntologyNode(),len(_dataNode),typesInDNs))

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

                return _dataNode
        
        _DataNodeBulder__Logger.error('Returning None - there are no dataNodes')
        return None
