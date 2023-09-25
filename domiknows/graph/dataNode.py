import torch
from collections import OrderedDict, namedtuple
from time import perf_counter, perf_counter_ns
import re

from .dataNodeConfig import dnConfig 

from ordered_set import OrderedSet 

from domiknows import getRegrTimer_logger, getProductionModeStatus
from domiknows.solver import ilpOntSolverFactory
from domiknows.utils import getDnSkeletonMode

import logging
from logging.handlers import RotatingFileHandler
from .property import Property
from .concept import Concept, EnumConcept

import graphviz

from sklearn import metrics

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
import pathlib
pathlib.Path("logs").mkdir(parents=True, exist_ok=True)
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
_DataNodeBuilder__Logger  = logging.getLogger("dataNodeBuilder")
_DataNodeBuilder__Logger.setLevel(logLevel)
# Add ch to logger
_DataNodeBuilder__Logger.addHandler(ch)
# Don't propagate
_DataNodeBuilder__Logger.propagate = False

_DataNodeBuilder__Logger.info('--- Starting new run ---')

# Class representing single data instance with relation links to other data nodes
class DataNode:
    def __init__(self, myBuilder = None, instanceID = None, instanceValue = None, ontologyNode = None, graph = None, relationLinks = {}, attributes = {}):

        self.myBuilder = myBuilder                       # DatanodeBuilder used to construct this datanode
        self.instanceID = instanceID                     # The data instance id (e.g. paragraph number, sentence number, phrase  number, image number, etc.)
        self.instanceValue = instanceValue               # Optional value of the instance (e.g. paragraph text, sentence text, phrase text, image bitmap, etc.)
        self.ontologyNode = ontologyNode                 # Reference to the node in the ontology graph (e.g. Concept) which is the type of this instance (e.g. paragraph, sentence, phrase, etc.)
        
        if ontologyNode is not None:
            self.graph = self.ontologyNode.sup
            if graph is not None:
                self.graph = graph
                
        if relationLinks:
            self.relationLinks = relationLinks           # Dictionary mapping relation name to RelationLinks
        else:
            self.relationLinks = {}
            
        self.impactLinks = {}                            # Dictionary with dataNodes impacting this dataNode by having it as a subject of its relation
        
        if attributes:
            self.attributes = attributes                 # Dictionary with node's attributes
        else:
            self.attributes = {}
            
        self.current_device = 'cpu'
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.current_device = 'cuda:1'
            else:
                self.current_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
        self.gurobiModel = None
        
        self.myLoggerTime = getRegrTimer_logger()
                     
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
                if isinstance(attribute, torch.Tensor):
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
    
    def hasAttribute(self, key):
        if key in self.attributes:
            return True
        elif "rootDataNode" in self.attributes:
            rootDataNode = self.attributes["rootDataNode"]
            if "variableSet" in rootDataNode.attributes:
                keyInVariableSet = self.ontologyNode.name + "/" + key
                if keyInVariableSet in rootDataNode.attributes["variableSet"]:
                    return True
                elif "propertySet" in rootDataNode.attributes and keyInVariableSet in rootDataNode.attributes["propertySet"]:
                    return True
        elif "variableSet" in self.attributes:
            if key in self.attributes["variableSet"]:
                return True
            elif "propertySet" in self and key in self.attributes["propertySet"]:
                return True
        else:
            return False
        
    def getAttribute(self, *keys):
        key = ""
        keyBis  = ""
        index = None
        
        conceptFound = False
        for _, kConcept in enumerate(keys):
            if key != "":
                key = key + "/"
                keyBis = keyBis + "/"
                
            # Handle different way of representing concept in the key list
            if isinstance(kConcept, str): # Concept name
                conceptForK = None
                if not conceptFound:
                    conceptForK = self.findConcept(kConcept) # Find concept
                
                if not conceptFound and conceptForK is not None:  
                    conceptFound = True
                    if isinstance(conceptForK, tuple):
                        key = key + '<' + conceptForK[0].name +'>'
                        index = conceptForK[2]
                        keyBis = keyBis + kConcept
                    else:
                        key = key + '<' + kConcept +'>'
                        keyBis = keyBis + kConcept
                else:
                    key = key + kConcept
                    keyBis = keyBis + kConcept
            elif isinstance(kConcept, tuple): # Concept represented as tuple
                conceptFound = True
                key = key + '<' + kConcept[0].name +'>'
                keyBis = keyBis + kConcept[0].name
            elif isinstance(kConcept, Concept): # Just concept
                conceptFound = True
                key = key + '<' + kConcept.name +'>'
                keyBis = keyBis + kConcept.name
            
        # Use key and keyBis to get the dn attribute     
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
        elif "rootDataNode" in self.attributes or "variableSet" in self.attributes:
            if "rootDataNode" in self.attributes:
                rootDataNode = self.attributes["rootDataNode"]
                keyInVariableSet = self.ontologyNode.name + "/" + key
                
                if "variableSet" in rootDataNode.attributes:
                    if keyInVariableSet in rootDataNode.attributes["variableSet"]:
                        return rootDataNode.attributes["variableSet"][keyInVariableSet][self.instanceID]
                    elif keyInVariableSet in rootDataNode.attributes["propertySet"]:
                        return rootDataNode.attributes["propertySet"][keyInVariableSet][self.instanceID]
            elif "variableSet" in self.attributes:
                if key in self.attributes["variableSet"]:
                    return self.attributes["variableSet"][key]
                elif key in self.attributes["propertySet"]:
                    return self.attributes["propertySet"][key]
                
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
            
    def getLinks(self, relationName = None, conceptName = None):
        keys = self.relationLinks.keys() | self.impactLinks.keys()
        
        links = {}
        for k in keys:
            if k not in self.relationLinks:
                links[k] = self.impactLinks[k]
                continue
            
            if k not in self.impactLinks:
                links[k] = self.relationLinks[k]
                continue
            
            links[k] = self.impactLinks[k] + self.relationLinks[k]
            
        if relationName is None:
            if conceptName is None:
                return links
            else:
                conceptCN = []
                
                for r in links:
                    for dn in links[r]:
                        if dn.ontologyNode.name == conceptName:
                            conceptCN.append(dn)
            
                return conceptCN
        
        if not isinstance(relationName, str):
            relationName = relationName.name
            
        if relationName in links:
            relDNs = links[relationName]
            
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
    
    # Recursively search for concepts and relations in the data graph
    def findConceptsAndRelations(self, dn, conceptsAndRelations = None, visitedDns = None):
        if 'variableSet' in self.attributes:
            conceptsAndRelations = set()
            for key in self.attributes['variableSet']:
                if "/label" in key:
                    continue
                conceptsAndRelations.add(key[key.index('<')+1:key.index('>')])
            
            return conceptsAndRelations
        else: 
            if conceptsAndRelations is None:
                conceptsAndRelations = set()
            if visitedDns is None:
                visitedDns = set()
                
            # Find concepts in dataNode - concept are in attributes from learning sensors
            for att in dn.attributes:
                if att[0] == '<' and att[-1] == '>':  
                    if att[1:-1] not in conceptsAndRelations:
                        conceptsAndRelations.add(att[1:-1])
                        _DataNode__Logger.info('Found concept %s in dataNode %s'%(att[1:-1],dn))
                        
            # Recursively find concepts and relations in linked dataNodes 
            links = dn.getLinks()
            if links:
                for link in links:
                    for lDn in links[link]:
                        if lDn in visitedDns:
                            continue
                        
                        visitedDns.add(lDn)
                        self.findConceptsAndRelations(lDn, conceptsAndRelations = conceptsAndRelations, visitedDns = visitedDns)
    
            return conceptsAndRelations

    # Find concept and relation names of DataNodes - used in concept.py
    def findConceptsNamesInDatanodes(self, dns = None, conceptNames = None, relationNames = None):
        if conceptNames is None:
            conceptNames=set()
        if relationNames is None:
            relationNames=set()

        if dns is None:
            dns = [self]
            
        for dn in dns:
            conceptNames.add(dn.getOntologyNode().name)
            for relName, _ in dn.getRelationLinks().items():
                if relName != 'contains':
                    relationNames.add(relName)
                
            self.findConceptsNamesInDatanodes(dns=dn.getChildDataNodes(), conceptNames = conceptNames, relationNames = relationNames)
            
        return conceptNames, relationNames
    
    # Find the root parent of relation of the given relation
    def findRootConceptOrRelation(self, relationConcept, usedGraph = None):
        if usedGraph is None:
            usedGraph = self.ontologyNode.getOntologyGraph()
        
        if isinstance(relationConcept, str):
            _relationConcepts = self.findConcept(relationConcept)
            
            if _relationConcepts:
                relationConcept = _relationConcepts[0]
            else:
                return relationConcept 

        # Does this concept or relation has parent (through _isA)
        
        if isinstance(relationConcept, tuple):
            relationConcept = relationConcept[0]
            
        try:
            isAs = relationConcept.is_a()
        except (AttributeError, TypeError):
            isAs = []
        
        for _isA in isAs:
            _relationConcept = _isA.dst
            
            return  self.findRootConceptOrRelation(_relationConcept, usedGraph)
        
        # If the provided concept or relation is root (has not parents)
        return relationConcept 

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
    
    def getDnsForRelation(self, rel):
        relRoot = self.findRootConceptOrRelation(rel)
            
        if relRoot is None:
            return [None]
        
        if not isinstance(relRoot, str):
            relRoot = relRoot.name     
        
        if relRoot.endswith(".reversed"):
            relRoot = relRoot[:-len(".reversed")]
            if relRoot in self.impactLinks: 
                return self.impactLinks[relRoot]
            else:
                return [None]
        elif relRoot in self.relationLinks:
            return self.relationLinks[relRoot]
        else:
            return [None]
            
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
                visitedDns = OrderedSet()
                             
            visitedDns.add(dn)
                    
        # Call recursively
        newDepth = depth + 1
        for dn in dns:
            # Visit  DataNodes in links
            for r, rValue in dn.getLinks().items():            
               
                # Check if the nodes already visited
                dnsToVisit = OrderedSet()
                for rDn in rValue:
                    if rDn not in visitedDns:
                        dnsToVisit.add(rDn)
                    
                if not dnsToVisit:
                    continue
                
                # Visit DataNodes in the current relation
                currentRelationDns = self.findDatanodes(dnsToVisit, select = select, indexes = indexes, visitedDns = visitedDns, depth = newDepth)
        
                if currentRelationDns is not None:
                    for currentRDn in currentRelationDns:
                        if currentRDn not in returnDns:
                            returnDns.append(currentRDn)

        if depth: # Finish recursion
            return returnDns
        
        # If index provided in query then filter the found results for the select part of query through the index part of query
        if (indexes != None):
            currentReturnDns = [] # Will contain results from returnDns satisfying the index
            
            for dn in returnDns:
                fit = True       
                for indexName, indexValue in indexes.items():
                    
                    relDns = dn.getDnsForRelation(indexName)
                    
                    if relDns is None or len(relDns) == 0 or relDns[0] is None:
                        fit = False
                        break
                    
                    found = False
                    for rDn in relDns:
                        if isinstance(indexValue, tuple):
                            _test = []
                            for t in indexValue:
                                if isinstance(t, tuple):
                                    r = self.__testDataNode(rDn, t)
                                    
                                    if r:
                                        found = True
                                        break
                                else:
                                    _test.append(t)
                             
                            if len(_test) == 0:
                                continue
                            else:
                                indexValue = _test
                                
                        if self.__testDataNode(rDn, indexValue):
                            found = True
                            break
                        
                    if not found:
                        fit = False
                        break
                        
                if fit:
                    if dn not in currentReturnDns:
                        currentReturnDns.append(dn)
                       
            returnDns = currentReturnDns
        
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
    
    # Keeps hashMap of concept name queries in findConcept to results
    conceptsMap = {}
    
    # Find concept in the graph based on concept name
    def findConcept(self, conceptName, usedGraph = None):
        if '<' in conceptName:
            conceptName = conceptName[1:-1]
            
        if not usedGraph:
            usedGraph = self.ontologyNode.getOntologyGraph()
            
        if usedGraph not in self.conceptsMap:
            self.conceptsMap[usedGraph] = {}
            
        usedGraphConceptsMap = self.conceptsMap[usedGraph]
        
        if isinstance(conceptName, Concept):
            conceptName = conceptName.name()
            
        if conceptName in usedGraphConceptsMap:
            return usedGraphConceptsMap[conceptName]
        
        subGraph_keys = [key for key in usedGraph._objs]
        for subGraphKey in subGraph_keys:
            subGraph = usedGraph._objs[subGraphKey]
            
            for conceptNameItem in subGraph.concepts:
                if conceptName == conceptNameItem:
                    concept = subGraph.concepts[conceptNameItem]
                    
                    usedGraphConceptsMap[conceptName] =  (concept, concept.name, None, 1)
                    return usedGraphConceptsMap[conceptName]
                
                elif isinstance(subGraph.concepts[conceptNameItem], EnumConcept):
                    vlen = len(subGraph.concepts[conceptNameItem].enum)
                    
                    if conceptName in subGraph.concepts[conceptNameItem].enum:
                        concept = subGraph.concepts[conceptNameItem]
                        
                        usedGraphConceptsMap[conceptName] = (concept, conceptName, subGraph.concepts[conceptNameItem].get_index(conceptName), vlen)
                        return usedGraphConceptsMap[conceptName]

        usedGraphConceptsMap[conceptName] = None
        
        return usedGraphConceptsMap[conceptName]

    # Check if concept is relation
    def isRelation(self, conceptRelation, usedGraph = None):
        if usedGraph is None:
            usedGraph = self.ontologyNode.getOntologyGraph()
        
        if isinstance(conceptRelation, str):
            conceptRelation = self.findConcept(conceptRelation)
            
            if conceptRelation == None:
                return False
            
            conceptRelation = conceptRelation[0]
            
        from  domiknows.graph.relation import Relation
        if isinstance(conceptRelation, Relation):
            return True
        
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

    # cache
    collectedConceptsAndRelations = None
    
    # Collect all the concepts and relation from the data graph and translate them to tuple form
    def collectConceptsAndRelations(self, conceptsAndRelations = None):
        if conceptsAndRelations is None:
            conceptsAndRelations = set()
            
        if self.collectedConceptsAndRelations:
            return self.collectedConceptsAndRelations
        
        # Search the graph starting from self for concepts and relations
        candR = self.findConceptsAndRelations(self) 
        self.rootConcepts = []
        
        returnCandR = []
        
        # Process founded concepts - translate them to tuple form with more information needed for logical constraints and metrics
        for c in candR:
            _concept = self.findConcept(c)[0]
                        
            if _concept is None:
                continue
            
            if isinstance(_concept, tuple):
                _concept = _concept[0]
            
            # Check if this is multiclass concept
            if isinstance(_concept, EnumConcept):
                self.rootConcepts.append((_concept, len(_concept.enum)))
                
                for i, a in enumerate(_concept.enum):
                    
                    if conceptsAndRelations and a not in conceptsAndRelations:
                        # continue
                        pass
                    
                    returnCandR.append((_concept, a, i, len(_concept.enum))) # Create tuple representation for multiclass concept
            else:
                self.rootConcepts.append((_concept, 1))

                if conceptsAndRelations and c not in conceptsAndRelations and _concept not in conceptsAndRelations:
                    continue
                
                returnCandR.append((_concept, _concept.name, None, 1)) # Create tuple representation for binary concept
        
        self.collectedConceptsAndRelations = returnCandR
        return self.collectedConceptsAndRelations
        
    def __getILPSolver(self, conceptsRelations = None):
        if conceptsRelations is None:
            conceptsRelations = []

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
    def collectInferredResults(self, concept, inferKey):
        collectAttributeList = []
        
        if not isinstance(concept, tuple):
            if not isinstance(concept, Concept):
                concept = self.findConcept(concept)
                if concept is None:
                    return torch.tensor(collectAttributeList)
                
            if isinstance(concept, EnumConcept):
                concept = (concept, concept.name, None, len(concept.enum))
            else:
                concept = (concept, concept.name, None, 1)
    
        rootConcept = self.findRootConceptOrRelation(concept[0])
        
        if not rootConcept:
            return torch.tensor(collectAttributeList)
        
        rootConceptDns = self.findDatanodes(select = rootConcept)
        
        if not rootConceptDns:
            return torch.tensor(collectAttributeList)
        
        if getDnSkeletonMode() and "variableSet" in self.attributes:
            vKeyInVariableSet = rootConcept.name + "/<" + concept[0].name +">"
            
            # inferKey
            inferKeyInVariableSet = vKeyInVariableSet + "/" + inferKey
            
            if self.hasAttribute(inferKeyInVariableSet):
                return self.getAttribute(inferKeyInVariableSet)

        keys = [concept, inferKey]
        
        for dn in rootConceptDns:
            rTensor = dn.getAttribute(*keys)
            if rTensor is None:
                continue
            
            if torch.is_tensor(rTensor):
                if len(rTensor.shape) == 0 or len(rTensor.shape) == 1 and rTensor.shape[0] == 1:
                    collectAttributeList.append(rTensor.item())
                elif (concept[2] is None) and concept[3] == 1: # local/argmax, rTensor.shape[0] == 2
                    collectAttributeList.append(rTensor[1])
                elif concept[2] is not None: # multiclass given index(concept[2]) of the multiclass category
                    collectAttributeList.append(rTensor[concept[2]])
                elif (concept[2] is None) and concept[3] > 1: # multiclass as whole thus no index
                    collectAttributeList.append(rTensor)
            elif isinstance(rTensor, (list,tuple)) and len(rTensor) == 1:
                collectAttributeList.append(rTensor[0])
            elif rTensor:
                collectAttributeList.append(1)
            else:
                collectAttributeList.append(0)
                
        if collectAttributeList and torch.is_tensor(collectAttributeList[0]):
            return torch.stack(tuple(collectAttributeList), dim=0)
        
        return torch.as_tensor(collectAttributeList)        
    
    # Calculate argMax and softMax
    def infer(self):
        conceptsRelations = self.collectConceptsAndRelations() 
        
        for c in conceptsRelations:
            cRoot = self.findRootConceptOrRelation(c[0])
            
            # ----- skeleton - tensor
            if getDnSkeletonMode() and "variableSet" in self.attributes:
                vKeyInVariableSet = cRoot.name + "/<" + c[0].name +">"
                
                # softmax
                softmaxKeyInVariableSet = vKeyInVariableSet + "/softmax"
                
                if not self.hasAttribute(softmaxKeyInVariableSet):
                    vKeyInVariableSetValues = self.attributes["variableSet"][vKeyInVariableSet]
                    if c[2] is not None:
                        v = vKeyInVariableSetValues[:, c[2]]
                    else:
                        v = vKeyInVariableSetValues[:, 1]
                    
                    # check if v is None or not a tensor
                    if v is None or not torch.is_tensor(v):
                        continue
                    
                    if not(isinstance(v, torch.FloatTensor) or isinstance(v, torch.cuda.FloatTensor)):
                        v = v.float()
                        
                    vSoftmaxT = torch.nn.functional.softmax(v, dim=-1)
                    self.attributes["variableSet"][softmaxKeyInVariableSet] = vSoftmaxT
                        
                # argmax
                argmaxKeyInVariableSet = vKeyInVariableSet + "/argmax"
                if not self.hasAttribute(argmaxKeyInVariableSet):
                    vKeyInVariableSetValues = self.attributes["variableSet"][vKeyInVariableSet]
                    if c[2] is not None:
                        v = vKeyInVariableSetValues[:, c[2]]
                    else:
                        v = vKeyInVariableSetValues[:, 1]
                     
                    vArgmaxTInxexes = torch.argmax(v, dim=-1)
                    vArgmax = torch.zeros_like(v).scatter_(-1, vArgmaxTInxexes.unsqueeze(-1), 1.)
                    
                    self.attributes["variableSet"][argmaxKeyInVariableSet] = vArgmax
                  
                # This is test 
                if False:
                    dns = self.findDatanodes(select = cRoot)  
                    for dn in dns:
                        keyArgmax = "<" + c[0].name + ">/argmax"
                        keySoftMax = "<" + c[0].name + ">/softmax"
                        
                        index = c[2]
                        if index is None:
                            index = 1
                            
                        s = dn.getAttribute(keySoftMax)[c[2]]
                        a = dn.getAttribute(keyArgmax)[c[2]]
                        continue
                        
            else:      
                # ---- loop through dns  
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
    def inferLocal(self, keys=("softmax", "argmax"), Acc=None):
        startInferLocal = perf_counter()

        # Go through keys and remove anything from each of them which is before slash, including slash        
        keys = [key[key.rfind('/')+1:] for key in keys]
        
        conceptsRelations = self.collectConceptsAndRelations() 
        
        normalized_keys = set([
                    "normalizedProb", "meanNormalizedProb", 
                    "normalizedProbAll", "meanNormalizedProbStd",
                    "normalizedProbAcc", "entropyNormalizedProbAcc",
                    "normalizedJustAcc",
                    ])
        
        if "softmax" in keys or normalized_keys.intersection(set(keys)):
            needSoftmax = True
        else:
            needSoftmax = False
            
        for c in conceptsRelations:
            cRoot = self.findRootConceptOrRelation(c[0])
            inferLocalKeys = list(keys) # used to check if all keys are calculated
            
            # ----- skeleton - tensor
            if getDnSkeletonMode() and "variableSet" in self.attributes:
                
                vKeyInVariableSet = cRoot.name + "/<" + c[0].name +">"
                
                if needSoftmax:
                    localSoftmaxKeyInVariableSet = vKeyInVariableSet + "/local/softmax"
                    
                    if "softmax" in inferLocalKeys:
                        inferLocalKeys.remove("softmax")
                    
                    if not self.hasAttribute(localSoftmaxKeyInVariableSet):
                        v = self.attributes["variableSet"][vKeyInVariableSet]
                        
                        # check if v is None or not a tensor
                        if v is None or not torch.is_tensor(v):
                            continue
                        
                        if not(isinstance(v, torch.FloatTensor) or isinstance(v, torch.cuda.FloatTensor)):
                            v = v.float()
                            
                        vSoftmaxT = torch.nn.functional.softmax(v, dim=-1)
                        self.attributes["variableSet"][localSoftmaxKeyInVariableSet] = vSoftmaxT
                        
                if "argmax" in keys:
                    localArgmaxKeyInVariableSet = vKeyInVariableSet + "/local/argmax"
                    inferLocalKeys.remove("argmax")
                    
                    if not self.hasAttribute(localArgmaxKeyInVariableSet):
                        v = self.attributes["variableSet"][vKeyInVariableSet]
                         
                        vArgmaxTInxexes = torch.argmax(v, dim=1)
                        vArgmax = torch.zeros_like(v).scatter_(1, vArgmaxTInxexes.unsqueeze(1), 1.)
                        
                        self.attributes["variableSet"][localArgmaxKeyInVariableSet] = vArgmax
            
            # check if we already processed all keys using skeleton
            if not inferLocalKeys:
                continue
            
            # ---- loop through dns
            dns = self.findDatanodes(select = cRoot)
            if not dns:
                continue
            
            vs = []
            
            for dn in dns:
                if needSoftmax:
                    keySoftmax = "<" + c[0].name + ">/local/softmax"

                    if not dn.hasAttribute(keySoftmax):                        
                        v = dn.getAttribute(c[0])
                        
                        # check if v is None or not a tensor
                        if v is None or not torch.is_tensor(v):
                            continue
                        
                        if not(isinstance(v, torch.FloatTensor) or isinstance(v, torch.cuda.FloatTensor)):
                            v = v.float()
                            
                        vSoftmaxT = torch.nn.functional.softmax(v, dim=-1)
                        dn.attributes[keySoftmax] = vSoftmaxT
                
                if "normalizedProb" in keys:
                    keyNormalizedProb = "<" + c[0].name + ">/local/normalizedProb"
                    if not dn.hasAttribute(keyNormalizedProb): # Already calculated ?   
                        vSoftmaxT = dn.getAttribute(keySoftmax)
                        
                        # Clamps the softmax probabilities
                        vector = torch.clamp(vSoftmaxT, min=1e-18, max=1 - 1e-18) 
                        
                        # Calculates their entropy;
                        entropy = torch.distributions.Categorical(torch.log(vector)).entropy() / vector.shape[0]
                        
                        # Multiplies the reverse of entropy to the vector divided by its mean value. P
                        vNormalizedProbT = (1/entropy.item()) * (vector/torch.mean(vector))
                        
                        dn.attributes[keyNormalizedProb] = vNormalizedProbT

                if "normalizedProbAcc" in keys:
                    keyNormalizedProb = "<" + c[0].name + ">/local/normalizedProbAcc"
                    if not dn.hasAttribute(keyNormalizedProb): # Already calculated ?   
                        vSoftmaxT = dn.getAttribute(keySoftmax)

                        # Clamps the softmax probabilities
                        vector = torch.clamp(vSoftmaxT, min=1e-18, max=1 - 1e-18) 
                        
                        ### Calculate the multiplier factor
                        if Acc and c[0].name in Acc:
                            multiplier = pow(Acc[c[0].name], 4)
                        else:
                            multiplier = 1
                        
                        # Calculates their entropy;
                        entropy = torch.distributions.Categorical(torch.log(vector)).entropy() / vector.shape[0]
                        
                        # Multiplies the reverse of entropy to the vector divided by its mean value. P
                        vNormalizedProbT = (1/entropy.item()) * (vector/torch.mean(vector))

                        if multiplier != 1:
                            vNormalizedProbT = vNormalizedProbT * multiplier
                        
                        dn.attributes[keyNormalizedProb] = vNormalizedProbT

                if "entropyNormalizedProbAcc" in keys:
                    keyNormalizedProb = "<" + c[0].name + ">/local/entropyNormalizedProbAcc"
                    if not dn.hasAttribute(keyNormalizedProb): # Already calculated ?   
                        vSoftmaxT = dn.getAttribute(keySoftmax)

                        # Clamps the softmax probabilities
                        vector = torch.clamp(vSoftmaxT, min=1e-18, max=1 - 1e-18) 
                        
                        ### Calculate the multiplier factor
                        if Acc and c[0].name in Acc:
                            multiplier = pow(Acc[c[0].name], 4)
                        else:
                            multiplier = 1
                        
                        # Calculates their entropy;
                        entropy = torch.distributions.Categorical(torch.log(vector)).entropy() / vector.shape[0]
                        
                        # Multiplies the reverse of entropy to the vector divided by its mean value. P
                        vNormalizedProbT = (1/entropy.item()) * vector

                        if multiplier != 1:
                            vNormalizedProbT = vNormalizedProbT * multiplier
                        
                        dn.attributes[keyNormalizedProb] = vNormalizedProbT

                if "normalizedJustAcc" in keys:
                    keyNormalizedProb = "<" + c[0].name + ">/local/normalizedJustAcc"
                    if not dn.hasAttribute(keyNormalizedProb): # Already calculated ?   
                        vSoftmaxT = dn.getAttribute(keySoftmax)
                        
                        ### Calculate the multiplier factor
                        if Acc and c[0].name in Acc:
                            multiplier = pow(Acc[c[0].name], 8)
                        else:
                            multiplier = 1
                        
                        # Calculates their entropy;
                        
                        # Multiplies the reverse of entropy to the vector divided by its mean value. P
                        vNormalizedProbT = vSoftmaxT

                        if multiplier != 1:
                            vNormalizedProbT = vNormalizedProbT * multiplier
                        
                        dn.attributes[keyNormalizedProb] = vNormalizedProbT

                if "meanNormalizedProb" in keys:
                    keyNormalizedProb = "<" + c[0].name + ">/local/meanNormalizedProb"
                    if not dn.hasAttribute(keyNormalizedProb): # Already calculated ?   
                        vSoftmaxT = dn.getAttribute(keySoftmax)

                        vector = vSoftmaxT
                        
                        # Multiplies the reverse of entropy to the vector divided by its mean value. P
                        vNormalizedProbT = vector/torch.mean(vector)
                        
                        dn.attributes[keyNormalizedProb] = vNormalizedProbT

                if "normalizedProbAll" in keys:
                    keyNormalizedProb = "<" + c[0].name + ">/local/normalizedProbAll"
                    if not dn.hasAttribute(keyNormalizedProb): # Already calculated ?   
                        vSoftmaxT = dn.getAttribute(keySoftmax)

                        # Clamps the softmax probabilities
                        vector = torch.clamp(vSoftmaxT, min=1e-18, max=1 - 1e-18) 
                        
                        # Calculates their entropy;
                        entropy = torch.distributions.Categorical(torch.log(vector)).entropy() / vector.shape[0]
                        
                        signs = vector - torch.mean(vector)
                        signs[signs < 0] = -1
                        signs[signs >= 0] = +1
                        adjustment = signs * torch.pow(vector - torch.mean(vector), 4)
                        
                        # Multiplies the reverse of entropy to the vector divided by its mean value. P
                        vNormalizedProbT = (1/entropy.item()) * (vector/torch.mean(vector)) + adjustment
                        
                        dn.attributes[keyNormalizedProb] = vNormalizedProbT

                if "meanNormalizedProbStd" in keys:
                    keyNormalizedProb = "<" + c[0].name + ">/local/meanNormalizedProbStd"
                    if not dn.hasAttribute(keyNormalizedProb): # Already calculated ?   
                        vSoftmaxT = dn.getAttribute(keySoftmax)

                        vector = vSoftmaxT

                        signs = vector - torch.mean(vector)
                        signs[signs < 0] = -1
                        signs[signs >= 0] = +1
                        adjustment = signs * torch.pow(vector - torch.mean(vector), 2)
                        
                        # Multiplies the reverse of entropy to the vector divided by its mean value. P
                        vNormalizedProbT = (adjustment/torch.pow(torch.mean(vector), 2))
                        
                        dn.attributes[keyNormalizedProb] = vNormalizedProbT
                
                if "argmax" in keys:
                    keyArgmax  = "<" + c[0].name + ">/local/argmax"
                    if not dn.hasAttribute(keyArgmax):
                        v = dn.getAttribute(c[0])
                        vArgmax = torch.zeros(v.shape)
                        vArgmaxCalculated = torch.argmax(v, keepdim=True)
                        vArgmaxIndex = torch.argmax(v).item()
                        vArgmax[vArgmaxIndex] = 1
                                        
                        dn.attributes[keyArgmax] = vArgmax
                        
        endInferLocal = perf_counter()
        elapsedInferLocalInMs = (endInferLocal - startInferLocal) * 1000
        self.myLoggerTime.info('Infer Local Probabilities - keys: %s, time: %dms', keys, elapsedInferLocalInMs)
        
    # Calculate ILP prediction for data graph with this instance as a root based on the provided list of concepts and relations
    def inferILPResults(self, *_conceptsRelations, key = ("local" , "softmax"), fun=None, epsilon = 0.00001, minimizeObjective = False, ignorePinLCs = False, Acc=None):
        if len(_conceptsRelations) == 0:
            _DataNode__Logger.info('Called with empty list of concepts and relations for inference')
        else:
            _DataNode__Logger.info('Called with - %s - list of concepts and relations for inference'%([x.name if isinstance(x, Concept) else x for x in _conceptsRelations]))
            
        # Check if full data node is created and create it if not -it is needed for ILP inference
        if self.myBuilder:
            self.myBuilder.createFullDataNode(self)
        
        # Check if concepts and/or relations have been provided for inference, if provide translate then to tuple concept info form
        _conceptsRelations = self.collectConceptsAndRelations(_conceptsRelations) # Collect all concepts and relations from data graph as default set
        if len(_conceptsRelations) == 0:
            _DataNode__Logger.error('Not found any concepts or relations for inference in provided DataNode %s'%(self))
            raise DataNode.DataNodeError('Not found any concepts or relations for inference in provided DataNode %s'%(self))
        else:        
            _DataNode__Logger.info('Found - %s - as a set of concepts and relations for inference'%([x[1] if isinstance(x, tuple) else x for x in _conceptsRelations]))
        myILPOntSolver, conceptsRelations = self.__getILPSolver(_conceptsRelations)
        
        # Call ilpOntSolver with the collected probabilities for chosen candidates
        _DataNode__Logger.info("Calling ILP solver")
        
        if "local" in key:
            keys = (key[1],)
            self.inferLocal(keys=keys, Acc=Acc)
          
        startILPInfer = perf_counter()
        if self.graph.batch and self.ontologyNode == self.graph.batch and 'contains' in self.relationLinks:
            batchConcept = self.graph.batch
            self.myLoggerTime.info('ILP Batch processing for %s'%(batchConcept))

            for batchIndex, dn in enumerate(self.relationLinks['contains']):
                startILPBatchStepInfer = perf_counter()
                myILPOntSolver.calculateILPSelection(dn, *conceptsRelations, key=key, fun=fun, epsilon = epsilon, minimizeObjective = minimizeObjective, ignorePinLCs = ignorePinLCs)
                endILPBatchStepInfer = perf_counter()    
                
                elapsedInS = (endILPBatchStepInfer - startILPBatchStepInfer)
                if elapsedInS > 1:
                    self.myLoggerTime.info('Finish step %i for batch ILP Inference - time: %fs'%(batchIndex,elapsedInS))
                else:
                    elapsedInMs = elapsedInS *1000
                    self.myLoggerTime.info('Finish step %i for batch ILP Inference - time: %ims'%(batchIndex,elapsedInMs))
        else:
            myILPOntSolver.calculateILPSelection(self, *conceptsRelations, key=key, fun=fun, epsilon = epsilon, minimizeObjective = minimizeObjective, ignorePinLCs = ignorePinLCs)    
        endILPInfer = perf_counter()
        
        elapsedInS = (endILPInfer - startILPInfer)
        if elapsedInS > 1:
            self.myLoggerTime.info('End ILP Inference - total time: %fs'%(elapsedInS))
        else:
            elapsedInMs = elapsedInS *1000
            self.myLoggerTime.info('End ILP Inference - total time: %ims'%(elapsedInMs))
                
        self.myLoggerTime.info('')
        
    def inferGBIResults(self, *_conceptsRelations, model):
        if len(_conceptsRelations) == 0:
            _DataNode__Logger.info('Called with empty list of concepts and relations for inference')
        else:
            _DataNode__Logger.info('Called with - %s - list of concepts and relations for inference'%([x.name if isinstance(x, Concept) else x for x in _conceptsRelations]))
            
        # Check if concepts and/or relations have been provided for inference, if provide translate then to tuple concept info form
        _conceptsRelations = self.collectConceptsAndRelations(_conceptsRelations) # Collect all concepts and relations from graph as default set

        from domiknows.program.model.gbi import GBIModel
        myGBIModel = GBIModel(self.graph, solver_model=model)
        myGBIModel.calculateGBISelection(self, _conceptsRelations)
        
    # Calculate the percentage of results satisfying each logical constraint 
    def verifyResultsLC(self, key = "/local/argmax"):
                
        myilpOntSolver, _ = self.__getILPSolver(conceptsRelations = self.collectConceptsAndRelations())

        # Check if full data node is created and create it if not
        self.myBuilder.createFullDataNode(self)
        
        if "local" in key:
            self.inferLocal(keys=[key])            
        elif "ILP" in key:
            self.infer()
        else:
            _DataNode__Logger.error("Not supported key %s for verifyResultsLC"%(key))
            
        verifyResult = myilpOntSolver.verifyResultsLC(self, key = key)
        
        return verifyResult
    
    # T-norms: L - Lukasiewicz, G - Godel, P - Product
    #tnorms = ['L', 'G', 'P']
    tnormsDefault = 'P'
    # sampleSize = -1 means Semantic Sample
    def calculateLcLoss(self, tnorm=tnormsDefault, sample = False, sampleSize = 0, sampleGlobalLoss = False):
        # Check if full data node is created and create it if not
        self.myBuilder.createFullDataNode(self)
        
        myilpOntSolver, conceptsRelations = self.__getILPSolver(conceptsRelations = self.collectConceptsAndRelations())

        self.inferLocal()
        lcResult = myilpOntSolver.calculateLcLoss(self, tnorm = tnorm, sample = sample, 
                                                  sampleSize = sampleSize, sampleGlobalLoss = sampleGlobalLoss, conceptsRelations = conceptsRelations)
        
        return lcResult

    def getInferMetrics(self, *conceptsRelations, inferType='ILP', weight = None, average='binary'):
        if not conceptsRelations:
            _DataNode__Logger.info("Calling %s metrics with empty conceptsRelations"%(inferType))
            conceptsRelations = self.collectConceptsAndRelations(conceptsRelations) # Collect all concepts and relations from graph as default set
            _DataNode__Logger.info("Found conceptsRelations in DataNode- %s"%(conceptsRelations))
        else:
            _DataNode__Logger.info("Calling %s metrics with conceptsRelations - %s"%(inferType, conceptsRelations))
                
        weightOriginal = weight
        if weight is None:
            weight = torch.tensor(1)
        else:
            _DataNode__Logger.info("Using weight %s"%(weight))
         
        # Will store calculated metrics an related data   
        result = {}   
        tp, fp, tn, fn  = [], [], [], []  
        isBinary = False  
        isMulticlass = False
        isMulticlassLabel = False
        
        # Calculate metrics for each provided concept
        for cr in conceptsRelations:
            # Check format of concepts and translate them to tuple in order to accommodate multiclass concepts
            if not isinstance(cr, tuple): # Not tuple concept form yet
                if not isinstance(cr, Concept): # If string find the corresponding concept
                    cr = self.findConcept(cr)
                    
                    if cr is None: # Sting mapping to concept is not found
                        _DataNode__Logger.error("% string is not a concept - not able to calculate metrics"%(cr))
                        continue
                elif isinstance(cr, EnumConcept): # Multiclass mapping to concept tuple form
                    cr = (cr, cr.name, None, len(cr.enum))
                elif isinstance(cr, Concept): # Binary mapping to tuple concept form
                    cr = (cr, cr.name, None, 1)
                else:
                    _DataNode__Logger.error("% string is not a concept - not able to calculate metrics"%(cr))
                    continue
            
            _DataNode__Logger.info("Calculating metrics for concept %s"%(cr[0]))

            # Collect date for metrics from DataNode
            preds = self.collectInferredResults(cr, inferType)
            labelsR = self.collectInferredResults(cr, 'label')

            # Check if not empty
            if preds is None:
                _DataNode__Logger.warning("Concept %s has predictions data None - not able to calculate metrics"%(cr[1]))
                continue
            else:
                _DataNode__Logger.info("Concept %s predictions from DataNode %s"%(cr[1], preds))

            if labelsR is None:
                _DataNode__Logger.warning("Concept %s has labels None - not able to calculate metrics"%(cr[1]))
                continue
            else:
                _DataNode__Logger.info("Concept %s labels from DataNode %s"%(cr[1], labelsR))
            
            if not torch.is_tensor(preds):
                _DataNode__Logger.error("Concept %s labels is not a Tensor - not able to calculate metrics"%(cr[1]))
                continue
            
            if not torch.is_tensor(labelsR):
                _DataNode__Logger.error("Concept %s predictions is not a Tensor - not able to calculate metrics"%(cr[1]))
                continue
            
            # Move to CPU
            if preds.is_cuda: preds = preds.cpu()
            if labelsR.is_cuda: labelsR = labelsR.cpu()
            
            # Translate labels - if provided as True/False to long
            labels = torch.clone(labelsR)
            labels = labels.long()
            preds = preds.long()
           
            # -- Multiclass processing
            
            # Check if concept is a label from Multiclass
            if cr[2] is not None: # Multiclass label given multiclass index (cr[2]) 
                isMulticlassLabel = True
                average = None
                labelsList = [i for i in range(cr[3])]
                _DataNode__Logger.info("Index of class Labels %s is %s"%(cr[1], cr[2]))
            # Check if concept is a  Multiclass
            elif (cr[2] is None) and cr[3] > 1: # Multiclass general without index (cr[2]) - called by the IML model forward method
                isMulticlass = True
                average = "micro"
                labelsList = [i for i in range(cr[3])]
                if preds.shape[0] == len(labelsR):
                    predsOriginal = preds
                    preds = torch.nonzero(preds, as_tuple=True)[1]
                    
                    if preds.shape[0] != len(labelsR):
                        _DataNode__Logger.warning("Concept %s predictions tensor has some predictions not calculated - %s"%(cr[1], predsOriginal))
                    
                    _DataNode__Logger.info("Concept %s is Multiclass "%(cr[1]))
                    _DataNode__Logger.info("Using average %s for Multiclass metrics calculation"%(average))

                else:
                    _DataNode__Logger.error("Incompatible lengths for %s between inferred results %s and labels %s"%(cr[1], len(preds), len(labelsR)))
                    continue
                
                _DataNode__Logger.info("Calculating metrics for all class Labels of  %s "%(cr[1]))
                multiclassLabels = cr[0].enum
                result = self.getInferMetrics(*multiclassLabels, inferType=inferType, weight = weightOriginal, average=average)
            else:
                isBinary = True
                labelsList = None

            # ---
            
            # Check if date prepared correctly
            if preds.dim() != 1:
                _DataNode__Logger.error("Concept %s predictions is Tensor with dimension %s > 1- not able to calculate metrics"%(cr[1], preds.dim()))
                continue
            
            if labels.dim() != 1:
                _DataNode__Logger.error("Concept %s labels is Tensor with dimension %s > 1- not able to calculate metrics"%(cr[1], labels.dim()))
                continue
            
            if  preds.size()[0] != labels.size()[0]:
                _DataNode__Logger.error("Concept %s labels size %s is not equal to prediction size %s - not able to calculate metrics"%(cr[1], labels.size()[0], preds.size()[0]))
                continue
            
            # Prepare the metrics result storage
            result[cr[1]] = {'cr': cr, 'inferType' : inferType, 'TP': torch.tensor(0.), 'FP': torch.tensor(0.), 'TN': torch.tensor(0.), 'FN': torch.tensor(0.)}
            
            # To numpy for sklearn
            labels = labels.numpy() 
            preds = preds.numpy()
            
            import numpy as np
            if np.sum(labels) == 0:
                _DataNode__Logger.warning("Concept %s - found all zero labels %s"%(cr[1], labels))
            else:
                _DataNode__Logger.info("Concept %s - labels used for metrics calculation %s"%(cr[1], labels))
            result[cr[1]]['labels'] = labels
            
            if np.sum(preds) == 0:
                _DataNode__Logger.warning("Concept %s - found all zero predictions %s"%(cr[1], preds))
            else:
                _DataNode__Logger.info("Concept %s - Predictions used for metrics calculation %s"%(cr[1], preds))
            result[cr[1]]['preds'] = preds

            # Calculate confusion matrix
            try:
                if isMulticlass:
                    cm = metrics.confusion_matrix(labels, preds)
                elif isMulticlassLabel:
                    cm = metrics.multilabel_confusion_matrix(labels, preds, labels=labelsList)
                    cm = cm[cr[2]]
                elif isBinary:
                    cm = metrics.confusion_matrix(labels, preds)
                    _tn, _fp, _fn, _tp = cm.ravel()
        
                    tp.append(_tp) 
                    result[cr[1]]['TP'] = _tp # true positive 
        
                    fp.append(_fp)
                    result[cr[1]]['FP'] = _fp # false positive
        
                    tn.append(_tn)
                    result[cr[1]]['TN'] = _tn # true negative
        
                    fn.append(_fn)
                    result[cr[1]]['FN'] = _fn # false positive
                else:
                    pass
                    
                result[cr[1]]['confusion_matrix'] = cm
                _DataNode__Logger.info("Concept %s confusion matrix %s"%(cr[1], result[cr[1]]['confusion_matrix']))
            except ValueError as ve: # Error when both labels and preds as zeros
                _DataNode__Logger.warning("Concept %s - both labels and predictions are all zeros - not able to calculate confusion metrics"%(cr[1]))
            
            # Calculate precision P - tp/(tp + fp)
            _p = metrics.precision_score(labels, preds, average=average, labels=labelsList, zero_division=0) # precision or positive predictive value (PPV)
            if isMulticlassLabel:
                _p = _p[cr[2]]
            result[cr[1]]['P'] = _p
            if _p == 0:
                _DataNode__Logger.warning("Concept %s precision %s"%(cr[1], _p))
            else:
                _DataNode__Logger.info("Concept %s precision %s"%(cr[1], _p))

            # Calculate recall R - tp/(tp + fn)
            _r = metrics.recall_score(labels, preds, average=average, labels=labelsList, zero_division=0) # recall, sensitivity, hit rate, or true positive rate (TPR)
            if isMulticlassLabel:
                _r = _r[cr[2]]
            result[cr[1]]['R'] = _r
            if _r == 0:
                _DataNode__Logger.warning("Concept %s recall %s"%(cr[1], _r))
            else:
                _DataNode__Logger.info("Concept %s recall %s"%(cr[1], _r))
             
            # Calculate F1 score - (P X R)/(P + R)
            _f1 = metrics.f1_score(labels, preds, average=average, labels=labelsList, zero_division=0) # f1
            if isMulticlassLabel:
                _f1 = _f1[cr[2]]
            result[cr[1]]['F1'] = _f1
            if _f1 == 0:
                _DataNode__Logger.warn("Concept %s f1 %s"%(cr[1], _f1))
            else:
                _DataNode__Logger.info("Concept %s f1 %s"%(cr[1], _f1))

        # --- Calculate Total metrics for binary concept
        if isBinary:
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
                if pT == 0:
                    _DataNode__Logger.warning("Total precision is %s"%(pT))
                else:
                    _DataNode__Logger.info("Total precision is %s"%(pT))
                    
                rT = tpT / (tpT + fnT)
                result['Total']['R'] = rT
                if rT == 0:
                    _DataNode__Logger.warning("Total recall is %s"%(rT))
                else:
                    _DataNode__Logger.info("Total recall is %s"%(rT))
                
                if pT + rT:
                    f1T = 2 * pT * rT / (pT + rT)
                    result['Total']['F1'] = f1T
                    if f1T == 0:
                        _DataNode__Logger.warning("Total F1 is %s"%(f1T))
                    else:
                        _DataNode__Logger.info("Total F1 is %s"%(f1T))
                        
                elif tpT + (fpT + fnT)/2:
                    f1T = tpT/(tpT + (fpT + fnT)/2)
                    result['Total']['F1'] = f1T
                    if f1T == 0:
                        _DataNode__Logger.warning("Total F1 is %s"%(f1T))
                    else:
                        _DataNode__Logger.info("Total F1 is %s"%(f1T))
                else:
                    _DataNode__Logger.warning("No able to calculate F1 for Total") 
        else:
            result['Total'] = {"No Total metrics for multiclass concept"}

        return result
    
# Class constructing the data graph based on the sensors data during the model execution
class DataNodeBuilder(dict):
    
    context = "build"
    def __init__(self, *args, **kwargs ):
        dict.__init__(self, *args, **kwargs )
        _DataNodeBuilder__Logger.info("")
        _DataNodeBuilder__Logger.info("Called")
        self.myLoggerTime = getRegrTimer_logger()
        
        from domiknows.utils import getDnSkeletonMode, getDnSkeletonModeFull
        self.skeletonDataNode = getDnSkeletonMode()
        self.skeletonDataNodeFull = getDnSkeletonModeFull()
        
        dict.__setitem__(self, "DataNodesConcepts", {})
        dict.__setitem__(self, "KeysInOrder", [])
        
        if args:
            dict.__setitem__(self, "data_item", args[0])

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
        
    # Add variable name to set
    def __addVariableNameToSet(self, vName):
        variableSetName = 'variableSet'
        if not dict.__contains__(self, variableSetName):
            dict.__setitem__(self, variableSetName, set())
        
        variableSet = dict.__getitem__(self, variableSetName)
        variableSet.add(vName)
        
    # Add property name to set
    def __addPropertyNameToSet(self, pName):
        propertySetName = 'propertySet'
        if not dict.__contains__(self, propertySetName):
            dict.__setitem__(self, propertySetName, set())
        
        variableSet = dict.__getitem__(self, propertySetName)
        variableSet.add(pName)
            
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
            
        counterName = 'Counter'
        for s in skey: # skey[2:]:
            counterName = counterName + '/' + s
            
        if not dict.__contains__(self, counterName):
            try:
                dict.__setitem__(self, counterName, {_value : {"counter": 1, "recent" : True}})
            except TypeError:
                return False
            
            return False
        else:
            currentCounter =  dict.__getitem__(self, counterName)
            
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
    def __findConcept(self, conceptName, usedGraph):
        subGraph_keys = [key for key in usedGraph._objs]
        for subGraphKey in subGraph_keys:
            subGraph = usedGraph._objs[subGraphKey]
            
            for conceptNameItem in subGraph.concepts:
                if conceptName == conceptNameItem:
                    concept = subGraph.concepts[conceptNameItem]
                    
                    return concept
        return None 
        
    # Collect concept information defined in the graph
    def __findConceptInfo(self, usedGraph, concept):
        conceptInfo = {
            'concept': concept,
            'relation': bool(concept.has_a()),
            'relationAttrs': {rel.name: self.__findConcept(rel.dst.name, usedGraph) for _, rel in enumerate(concept.has_a())},
            'root': not ('contains' in concept._in),
            'contains': [contain.dst for contain in concept._out.get('contains', [])],
            'containedIn': [contain.src for contain in concept._in.get('contains', [])]
        }

        return conceptInfo
            
    def __updateConceptInfo(self,  usedGraph, conceptInfo, sensor):
        from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
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

    def __isRootDn(self, testedDn, checkedDns, visitedDns):
        if visitedDns == None:
            visitedDns = set()
            
        visitedDns.add(testedDn)
        
        if not testedDn.impactLinks and testedDn in checkedDns:
            return False
        
        isRoot = True    
        for _, iDnList in testedDn.impactLinks.items(): # Check if its impacts are connected to Dn in the new Root list
            if iDnList:
                for iDn in iDnList:
                    if iDn in visitedDns:
                        continue
                    
                    if self.__isRootDn(iDn, checkedDns, visitedDns):
                        isRoot = False
                        break
                    
            if not isRoot:
                break
            
        return isRoot
    
    # Update root dataNodes list
    def __updateRootDataNodeList(self, *dns):
        if not dns:
            return
    
        # Get existing roots dataNodes
        if dict.__contains__(self, 'dataNode'):
            dnsRoots = dict.__getitem__(self, 'dataNode')
            _DataNodeBuilder__Logger.debug('Existing elements in the root dataNodes list - %s'%(dnsRoots))
        else:
            dnsRoots = []
        
        # First flatten the list of new dataNodes
        def flatten(dns):
            for dn in dns:
                if isinstance(dn, list):
                    yield from flatten(dn)
                else:
                    yield dn

        # Flatten the list of new dataNodes
        flattenDns = list(flatten(dns))
            
        # Create a set of all unique dataNodes in dnsRoots and flattenDns
        allDns = set(dnsRoots)
        allDns.update(flattenDns)
                
        # -- Update list of existing root dataNotes  
        
        # Will be used to store new root dataNodes
        newDnsRoots = []
        
        # Loop over all known unique dataNodes
        #for dnE in allDns:
        # Check if the dataNode is a root dataNode because it has no impact link
        # if not dnE.impactLinks: # Has no impact link
        #     if dnE not in newDnsRoots: # Not yet in the new Root list
        #         # Add it to the new Root list
        #         newDnsRoots.append(dnE)
        # else:
        #     # Check if the current dataNode is still a root dataNode
        #     if self.__isRootDn(dnE, dnsRoots, visitedDns = None):
        #         newDnsRoots.append(dnE)
        
        # Count the number of incoming links for each dataNode
        incomingLinks = {dn: 0 for dn in allDns}
        dnTypes = {}
        for dn in allDns:
            if dn.ontologyNode in dnTypes:
                dnTypes[dn.ontologyNode].append(dn)
            else:
                dnTypes[dn.ontologyNode] = [dn]
                
            for il in dn.impactLinks:
                if il in incomingLinks:
                    incomingLinks[dn] += 1
                else:
                    incomingLinks[dn] = 1
        
        # Find the root dataNodes which have no incoming links
        newDnsRoots = [dn for dn in allDns if incomingLinks[dn] == 0 or not dn.impactLinks]
        newDnsRoots = sorted(newDnsRoots, key=lambda dn: len(dnTypes[dn.ontologyNode]), reverse=False)

        # if newDnsRoots is empty
        if not newDnsRoots:
            newDnsRoots = allDns
            #newDnsRoots = sorted(newDnsRoots, key=lambda dn: incomingLinks[dn], reverse=True)
            newDnsRoots = sorted(newDnsRoots, key=lambda dn: len(dnTypes[dn.ontologyNode]), reverse=False)
            
        # Set the updated root list 
        if not getProductionModeStatus():
            _DataNodeBuilder__Logger.info('Updated elements in the root dataNodes list - %s'%(newDnsRoots))
        dict.__setitem__(self, 'dataNode', newDnsRoots) # Updated the dict 
    
        return
    
    # Build or update relation dataNode in the data graph for a given key
    def __buildRelationLink(self, vInfo, conceptInfo, keyDataName):
        relationName = conceptInfo['concept'].name
         
        # Check if data graph started
        existingRootDns = dict.__getitem__(self, 'dataNode') # DataNodes roots
        
        if not existingRootDns:
            _DataNodeBuilder__Logger.error('No dataNode created yet - abandon processing relation link dataNode value for %s and attribute %s'%(relationName,keyDataName))
            return # No graph yet - information about relation should not be provided yet
        
        # Find if DataNodes for this relation have been created
        existingDnsForRelation = self.findDataNodesInBuilder(select = relationName)
        
        existingDnsForRelationNotSorted = OrderedDict()
        for dn in existingDnsForRelation:
            existingDnsForRelationNotSorted[dn.getInstanceID()] = dn
                
        existingDnsForRelationSorted = OrderedDict(sorted(existingDnsForRelationNotSorted.items()))
            
        # This is an information about relation attributes
        if conceptInfo['relationAttrData']:
            index = keyDataName.index('.')
            attrName = keyDataName[0:index]
            
            relationAttrsCacheName = conceptInfo['concept'].name + "RelationAttrsCache"
            
            if not dict.__contains__(self, relationAttrsCacheName):
                dict.__setitem__(self, relationAttrsCacheName, {})
        
            relationAttrsCache =  dict.__getitem__(self, relationAttrsCacheName)
            relationAttrsCache[attrName] = vInfo.value
                
            if not getProductionModeStatus():
                _DataNodeBuilder__Logger.info('Caching received data for %s related to relation %s dataNode, found %i existing dataNode of this type - provided value has length %i'
                                         %(keyDataName,relationName,len(existingDnsForRelation),vInfo.len))
            
            # Find if all the needed attribute were initialized
            allAttrInit = True
            for relationAttributeName, _ in conceptInfo['relationAttrsGraph'].items():
                if relationAttributeName not in relationAttrsCache:
                    allAttrInit = False
                    break
            
            if allAttrInit: #Create links for the relation DataNode
                # Find DataNodes connected by this relation based on graph definition
                existingDnsForAttr = OrderedDict() # DataNodes for Attributes of the relation
                for relationAttributeName, relationAttributeConcept in conceptInfo['relationAttrsGraph'].items():
                    _existingDnsForAttr = self.findDataNodesInBuilder(select = relationAttributeConcept.name)
                     
                    if _existingDnsForAttr:
                        existingDnsForAttr[relationAttributeName] = _existingDnsForAttr
                        if not getProductionModeStatus():
                            _DataNodeBuilder__Logger.info('Found %i dataNodes of the attribute %s for concept %s'%(len(_existingDnsForAttr),relationAttributeName,relationAttributeConcept.name))
                    else:
                        existingDnsForAttr[relationAttributeName] = []
                        _DataNodeBuilder__Logger.warning('Not found dataNodes of the attribute %s for concept %s'%(relationAttributeName,relationAttributeConcept.name))
                                    
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
                            
                            #if attributeIndex == 0:
                            #   candidateDn.addRelationLink(attribute, relationDn)
                            
                            relationDn.addRelationLink(attribute, candidateDn)  
                            if (not self.skeletonDataNode):
                                relationDn.attributes[keyDataName] = vInfo.value[relationDnIndex] # Add / /Update value of the attribute
                
                if not getProductionModeStatus():
                    _DataNodeBuilder__Logger.info('Create links between the relation %s and instance dataNode of types'%(conceptInfo['concept'].name))
            else:
                # Just add the sensor value to relation DataNodes
                if not getProductionModeStatus():
                    if keyDataName in self:
                        _DataNodeBuilder__Logger.info('Updating attribute %s in relation link dataNodes %s'%(keyDataName,conceptInfo['concept'].name))
                    else:
                        _DataNodeBuilder__Logger.info('Adding attribute %s to relation link dataNodes %s'%(keyDataName,conceptInfo['concept'].name))
                        
                if (not self.skeletonDataNode):
                    for i, rDn in existingDnsForRelationSorted.items(): # Loop through all relation links dataNodes
                        rDn.attributes[keyDataName] = vInfo.value[i] # Add / /Update value of the attribute

            self.__updateRootDataNodeList(list(existingDnsForRelationSorted.values()))
        else:    
            # -- DataNode with this relation already created  - update it with new attribute value
            if not getProductionModeStatus():
                if keyDataName in self:
                    _DataNodeBuilder__Logger.info('Updating attribute %s in relation link dataNodes %s'%(keyDataName,conceptInfo['concept'].name))
                else:
                    _DataNodeBuilder__Logger.info('Adding attribute %s to relation link dataNodes %s'%(keyDataName,conceptInfo['concept'].name))
    
            if len(existingDnsForRelation) != vInfo.len:
                _DataNodeBuilder__Logger.error('Number of relations is %i and is different then the length of the provided tensor %i'%(len(existingDnsForRelation),vInfo.len))
                raise ValueError('Number of relations is %i and is different then the length of the provided tensor %i'%(len(existingDnsForRelation),vInfo.len))
 
            if (not self.skeletonDataNode):
                if len(existingDnsForRelationSorted) == 1:
                    if vInfo.dim == 0:
                        existingDnsForRelationSorted[0].attributes[keyDataName] = vInfo.value # Add / /Update value of the attribute
                elif vInfo.dim > 0:
                    for i, rDn in existingDnsForRelationSorted.items(): # Loop through all relations links dataNodes
                        rDn.attributes[keyDataName] = vInfo.value[i] # Add / /Update value of the attribute
                else:
                    pass

    def __createInitialDataNode(self, vInfo, conceptInfo, keyDataName):
        conceptName = conceptInfo['concept'].name

        dns = []
           
        if not getProductionModeStatus():        
            _DataNodeBuilder__Logger.info('Creating initial dataNode - provided value has length %i'%(vInfo.len))

        if vInfo.len == 1: # Will use "READER" key as an id of the root dataNode
            instanceValue = ""
            
            if "READER" in self:
                instanceID = dict.__getitem__(self, "READER")
            else:
                instanceID = 0
                
            initialDn = DataNode(myBuilder = self, instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
            
            if (not self.skeletonDataNode):
                initialDn.attributes[keyDataName] = vInfo.value
            
            if not getProductionModeStatus():
                _DataNodeBuilder__Logger.info('Created single dataNode with id %s of type %s'%(instanceID,conceptName))
            dns.append(initialDn)
        elif vInfo.len > 1:
            for vIndex, v in enumerate(vInfo.value):
                instanceValue = ""
                instanceID = vIndex
                newInitialDn = DataNode(myBuilder = self, instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                
                if (not self.skeletonDataNode):
                    newInitialDn.attributes[keyDataName] = v
                
                dns.append(newInitialDn)
                        
            if not getProductionModeStatus():
                _DataNodeBuilder__Logger.info('Created %i dataNodes of type %s'%(len(dns),conceptName))
                    
        self.__updateRootDataNodeList(dns)
        
        return dns
    
    def __createSingleDataNode(self, vInfo, conceptInfo, keyDataName):
        conceptName = conceptInfo['concept'].name
        if not getProductionModeStatus():
            _DataNodeBuilder__Logger.info('Received information about dataNodes of type %s - value dim is %i and length is %i'%(conceptName,vInfo.dim,vInfo.len))

        # -- Create a single the new dataNode 
        instanceValue = ""
        instanceID = 0
        newSingleDn = DataNode(myBuilder = self, instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
        if (not self.skeletonDataNode):
            newSingleDn.attributes[keyDataName] = vInfo.value
          
        if not getProductionModeStatus():      
            _DataNodeBuilder__Logger.info('Single new dataNode %s created'%(newSingleDn))

        self.__updateRootDataNodeList(newSingleDn)
                
        return [newSingleDn]
        
    def __createMultiplyDataNode(self, vInfo, conceptInfo, keyDataName):
        conceptName = conceptInfo['concept'].name
        
        # Master List of lists of created dataNodes - each list in the master list represent set of new dataNodes connected to the same parent dataNode 
        # (identified by the index in the master list)
        dns = [] 
                
        if not getProductionModeStatus():
            _DataNodeBuilder__Logger.info('Received information about dataNodes of type %s - value dim is %i and length is %i'%(conceptName,vInfo.dim,vInfo.len))

        # --- Create dataNodes
        
        # Check the type of sensor data
        if vInfo.dim == 0: 
            _DataNodeBuilder__Logger.warning('Provided value is empty %s - abandon the update'%(vInfo.value))
            return
        elif vInfo.dim == 1: # List with indexes for new DataNodes and data for attribute
            if not getProductionModeStatus():
                _DataNodeBuilder__Logger.info('Adding %i new dataNodes of type %s'%(vInfo.len,conceptName))

            for vIndex, v in enumerate(vInfo.value):
                instanceValue = ""
                instanceID = vIndex
                
                # Create new DataNode
                newDn = DataNode(myBuilder = self, instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                
                # add attribute
                if (not self.skeletonDataNode):
                    newDn.attributes[keyDataName] = v
                
                dns.append(newDn)       
        elif vInfo.dim == 2: # Two dimensional relation information
            if "relationMode" in conceptInfo:
                relatedDnsType = conceptInfo["relationAttrs"]['src']
                relatedDns = self.findDataNodesInBuilder(select = relatedDnsType)
                
                if len(vInfo.value) > 0:
                    try:
                        requiredLenOFRelatedDns = len(vInfo.value[0])
                    except IndexError:
                        requiredLenOFRelatedDns = 0
                else:
                    requiredLenOFRelatedDns = 0
                    
                if requiredLenOFRelatedDns != len(relatedDns):
                    _DataNodeBuilder__Logger.warning('Value of %s expects %i related dataNode of type %s but the number of existing dataNodes is %i - abandon the update'
                                                    %(conceptInfo['relationName'],requiredLenOFRelatedDns,relatedDnsType,len(relatedDns)))
                    return
           
                if not getProductionModeStatus():
                    _DataNodeBuilder__Logger.info('Create %i new dataNodes of type %s'%(vInfo.len,conceptName))
                
                    if not conceptInfo['relation']:
                        _DataNodeBuilder__Logger.info('It is a contain update of type - %s'%(conceptInfo["relationMode"]))
                        if conceptInfo["relationMode"] == "forward":
                            _DataNodeBuilder__Logger.info('%s is contain in %s'%(conceptName, relatedDnsType))
                        else:
                            _DataNodeBuilder__Logger.info('%s is contain in %s'%(relatedDnsType, conceptName))

                for i in range(0,vInfo.len):
                    instanceValue = ""
                    instanceID = i
                    newDn = DataNode(myBuilder = self, instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                        
                    if (not self.skeletonDataNode):
                        newDn.attributes[keyDataName] = vInfo.value[i]
                    dns.append(newDn)
                    
                    # If it is not a regular relation but (Create contain relation between the new DataNode and existing DataNodes
                    if not conceptInfo['relation']:
                        if conceptInfo["relationMode"] == "forward":
                            for index, isRelated in enumerate(vInfo.value[i]):
                                if isRelated == 1:
                                    relatedDns[index].addChildDataNode(newDn)                            
                        elif conceptInfo["relationMode"] == "backward":
                            for index, isRelated in enumerate(vInfo.value[i]):
                                if isRelated == 1:
                                    newDn.addChildDataNode(relatedDns[index])  
            else:
                if not getProductionModeStatus():
                    _DataNodeBuilder__Logger.info('Create %i new dataNodes of type %s'%(vInfo.len,conceptName))
                for i in range(0,vInfo.len):
                    instanceValue = ""
                    instanceID = i
                    newDn = DataNode(myBuilder = self, instanceID = instanceID, instanceValue = instanceValue, ontologyNode = conceptInfo['concept'])
                        
                    dns.append(newDn)
        else:
            _DataNodeBuilder__Logger.warning('It is an unsupported sensor input - %s'%(vInfo))
                
        self.__updateRootDataNodeList(dns)   
        return dns
            
    # not called when skeletonDataNode is on
    def __updateDataNodes(self, vInfo, conceptInfo, keyDataName):
        conceptName = conceptInfo['concept'].name
        existingDnsForConcept = self.findDataNodesInBuilder(select = conceptName) # Try to get DataNodes of the current concept

        if not existingDnsForConcept:
            existingDnsForConcept = self.findDataNodesInBuilder(select = conceptName)
            
        if not existingDnsForConcept:
            return
         
        if not getProductionModeStatus():                                                           
            if keyDataName in existingDnsForConcept[0].attributes:
                _DataNodeBuilder__Logger.info('Updating attribute %s in existing dataNodes - found %i dataNodes of type %s'%(keyDataName, len(existingDnsForConcept),conceptName))
            else:
                _DataNodeBuilder__Logger.info('Adding attribute %s in existing dataNodes - found %i dataNodes of type %s'%(keyDataName, len(existingDnsForConcept),conceptName))
            
        if len(existingDnsForConcept) > vInfo.len: # Not enough elements in the value 
            _DataNodeBuilder__Logger.warning('Provided value has length %i but found %i existing dataNode - abandon the update'%(vInfo.len,len(existingDnsForConcept)))
        elif len(existingDnsForConcept) == vInfo.len: # Number of  value elements matches the number of found dataNodes
            if len(existingDnsForConcept) == 0:
                return
            elif vInfo.dim == 0:
                if isinstance(vInfo.value, torch.Tensor):
                    if keyDataName[0] == '<' and keyDataName[-1] == '>':
                        existingDnsForConcept[0].attributes[keyDataName] = [1-vInfo.value.item(), vInfo.value.item()]
                    else:
                        existingDnsForConcept[0].attributes[keyDataName] = vInfo.value
                else:
                    existingDnsForConcept[0].attributes[keyDataName] = [vInfo.value]
            else:
                for vIndex, v in enumerate(vInfo.value):
                    if isinstance(existingDnsForConcept[vIndex], DataNode): # Check if DataNode
                        existingDnsForConcept[vIndex].attributes[keyDataName] = v
                    else:
                        _DataNodeBuilder__Logger.error('Element %i in the list is not a dataNode - skipping it'%(vIndex))
                        raise ValueError('Element %i in the list is not a dataNode - skipping it'%(vIndex))
        
                if keyDataName[0] == '<' and keyDataName[-1] == '>':
                    if "contains" in existingDnsForConcept[0].impactLinks:
                        dnParent = existingDnsForConcept[0].impactLinks["contains"][0]
                        dnParent.attributes[keyDataName] = vInfo.value
        elif len(existingDnsForConcept) < vInfo.len: # Too many elements in the value
            _DataNodeBuilder__Logger.warning('Provided value has length %i but found %i existing dataNode - abandon the update'%(vInfo.len,len(existingDnsForConcept)))
            
        # Check if this is the contain relation update or attribute update
        if "relationMode" in conceptInfo and  not conceptInfo["relation"]:
            relatedDnsType = conceptInfo["relationAttrs"]['src']

            relatedDns = self.findDataNodesInBuilder(select = relatedDnsType)

            if vInfo.dim:
                requiredLenOFRelatedDns = len(vInfo.value[0])
            else:
                requiredLenOFRelatedDns = len(vInfo.item())
            
            if requiredLenOFRelatedDns != len(relatedDns):
                _DataNodeBuilder__Logger.error('Provided value expected %i related dataNode of type %s but the number of existing dataNodes is %i - abandon the update'
                                              %(requiredLenOFRelatedDns,relatedDnsType,len(relatedDns)))
                raise ValueError('Provided value expected %i related dataNode of type %s but the number of existing dataNodes is %i - abandon the update'
                                              %(requiredLenOFRelatedDns,relatedDnsType,len(relatedDns)))

             
            if not getProductionModeStatus():   
                _DataNodeBuilder__Logger.info('It is a contain update of type - %s'%(conceptInfo["relationMode"]))
                if conceptInfo["relationMode"] == "forward":
                    _DataNodeBuilder__Logger.info('%s is contain in %s'%(conceptName, relatedDnsType))
                else:
                    _DataNodeBuilder__Logger.info('%s is contain in %s'%(relatedDnsType, conceptName))
                
            for i in range(0,vInfo.len):
                exitingDn = existingDnsForConcept[i]
                    
                if conceptInfo["relationMode"] == "forward":
                    for index, isRelated in enumerate(vInfo.value[i]):
                        if isRelated == 1:
                            relatedDns[index].addChildDataNode(exitingDn)                            
                elif conceptInfo["relationMode"] == "backward":
                    for index, isRelated in enumerate(vInfo.value[i]):
                        if isRelated == 1:
                            exitingDn.addChildDataNode(relatedDns[index])  
                
            self.__updateRootDataNodeList(existingDnsForConcept)   
                      
    # Build or update dataNode in the data graph for a given relationAttributeConcept
    def __buildDataNode(self, vInfo, conceptInfo, keyDataName):
        conceptName = conceptInfo['concept'].name
       
        if not dict.__contains__(self, 'dataNode'):   # ------ No DataNode yet
            return self.__createInitialDataNode(vInfo, conceptInfo, keyDataName) # Done - End the method
        else:
            # ---------- DataNodes already created
            existingDnsForConcept = self.findDataNodesInBuilder(select = conceptName) # Try to get DataNodes of the current concept
            
            if len(existingDnsForConcept) == 0:# Check if DataNode for this concept already created                    
                # No DataNode of this concept created yet
    
                # If attribute value is a single element - will create a single new DataNode
                if vInfo.len == 1 and vInfo.dim < 2: 
                    return self.__createSingleDataNode(vInfo, conceptInfo, keyDataName)
                else: # -- Value is multiple elements
                    return self.__createMultiplyDataNode(vInfo, conceptInfo, keyDataName)
            else: # DataNode with this concept already created - update it
                if (not self.skeletonDataNode):
                    self.__updateDataNodes(vInfo, conceptInfo, keyDataName)
                
    def __addEquality(self, vInfo, conceptInfo, equalityConceptName, keyDataName):
        conceptName = conceptInfo['concept'].name
        existingDnsForConcept = self.findDataNodesInBuilder(select = conceptName)
        existingDnsForEqualityConcept = self.findDataNodesInBuilder(select = equalityConceptName)
        
        if not existingDnsForConcept and not existingDnsForEqualityConcept:
            _DataNodeBuilder__Logger.warning('No datNodes created for concept %s and equality concept %s'%(conceptName,equalityConceptName))
            return
        
        if not existingDnsForConcept:
            _DataNodeBuilder__Logger.warning('No datNodes created for concept %s'%(conceptName))
            return
        
        if not existingDnsForEqualityConcept:
            _DataNodeBuilder__Logger.warning('No datNodes created for equality concept %s'%(equalityConceptName))
            return
        
        if not getProductionModeStatus():
            _DataNodeBuilder__Logger.info('Added equality between dataNodes of types %s and %s'%(conceptName,equalityConceptName))

        for conceptDn in existingDnsForConcept:
            for equalDn in existingDnsForEqualityConcept:
                
                if conceptDn.getInstanceID() >= vInfo.value.shape[0]:
                    continue
                
                if equalDn.getInstanceID() >= vInfo.value.shape[1]:
                    continue
                
                if vInfo.value[conceptDn.getInstanceID(), equalDn.getInstanceID()]:
                    if not getProductionModeStatus():
                        _DataNodeBuilder__Logger.info('DataNodes of %s is equal to %s'%(conceptDn,equalDn))
                    conceptDn.addEqualTo(equalDn)

    # Method processing value of the attribute - determining it it should be treated as a single element. 
    #     It returns a tuple with elements specifying the length of the first dimension of the value, 
    #     the number of dimensions of the value and the original value itself
    def __processAttributeValue(self, value, keyDataName):
        ValueInfo = namedtuple('ValueInfo', ["len", "value", 'dim'])

        if isinstance(value, torch.Tensor):
            dimV = value.dim()
            if dimV:
                lenV = len(value)
            else:
                lenV = 1
        else:
            lenV = len(value)
            
        if not isinstance(value, (torch.Tensor, list)): # It is scalar value
            return ValueInfo(len = 1, value = value, dim=0) 
            
        if isinstance(value, torch.Tensor) and dimV == 0: # It is a Tensor but also scalar value
            return ValueInfo(len = 1, value = value.item(), dim=0)
        
        if (lenV == 1): # It is Tensor or list with length 1 - treat it as scalar
            if isinstance(value, list) and not isinstance(value[0], (torch.Tensor, list)) : # Unpack the value
                return ValueInfo(len = 1, value = value[0], dim=0)
            elif isinstance(value, torch.Tensor) and dimV < 2:
                return ValueInfo(len = 1, value = torch.squeeze(value, 0), dim=0)

        #  If it is Tensor or list with length 2 but it is for attribute providing probabilities - assume it is a scalar value
        if isinstance(value, list) and lenV ==  2 and keyDataName[0] == '<': 
            return ValueInfo(lenV = 1, value = value, dim=0)
        elif isinstance(value, torch.Tensor) and lenV ==  2 and dimV  == 0 and keyDataName[0] == '<':
            return ValueInfo(len = 1, value = value, dim=0)

        if isinstance(value, list): 
            if not isinstance(value[0], (torch.Tensor, list)) or (isinstance(value[0], torch.Tensor) and value[0].dim() == 0):
                return ValueInfo(len = lenV, value = value, dim=1)
            elif not isinstance(value[0][0], (torch.Tensor, list)) or (isinstance(value[0][0], torch.Tensor) and value[0][0].dim() == 0):
                return ValueInfo(len = lenV, value = value, dim=2)
            elif not isinstance(value[0][0][0], (torch.Tensor, list)) or (isinstance(value[0][0][0], torch.Tensor) and value[0][0][0].dim() == 0):
                return ValueInfo(len = lenV, value = value, dim=3)
            else:
                _DataNodeBuilder__Logger.warning('Dimension of nested list value for key %s is more then 3 returning dimension 4'%(keyDataName))
                return ValueInfo(len = lenV, value = value, dim=4)

        elif isinstance(value, torch.Tensor):
            return ValueInfo(len = lenV, value = value, dim=dimV)
    
    def collectTime(self, start):
        # Collect time used for __setitem__
        end = perf_counter_ns()
        currentTime =  end - start
    
        timeList = self.setdefault("DataNodeTime", [])
        timeList.append(currentTime)
        startTimeList = self.setdefault("DataNodeTime_start", [])
        startTimeList.append(start)
        endTimeList = self.setdefault("DataNodeTime_end", [])
        endTimeList.append(end)
        
    # Overloaded __setitem Dictionary method - tracking sensor data and building corresponding data graph
    def __setitem__(self, _key, value):
        from ..sensor import Sensor

        start = perf_counter_ns()
        self.__addSetitemCounter()
        
        if isinstance(_key, (Sensor, Property, Concept)):
            key = _key.fullname
            if  isinstance(_key, Sensor) and not _key.build:
                if isinstance(value, torch.Tensor):
                    _DataNodeBuilder__Logger.debug('No processing (because build is set to False) - key - %s, key type - %s, value - %s, shape %s'%(key,type(_key),type(value),value.shape))
                elif isinstance(value, list):
                    _DataNodeBuilder__Logger.debug('No processing (because build is set to False) - key - %s, key type - %s, value - %s, length %s'%(key,type(_key),type(value),len(value)))
                else:
                    _DataNodeBuilder__Logger.debug('No processing (because build is set to False) - key - %s, key type - %s, value - %s'%(key,type(_key),type(value)))

                self.collectTime(start)
                return dict.__setitem__(self, _key, value)
            
            if  isinstance(_key, Property):
                if isinstance(value, torch.Tensor):
                    _DataNodeBuilder__Logger.debug('No processing Property as key - key - %s, key type - %s, value - %s, shape %s'%(key,type(_key),type(value),value.shape))
                elif isinstance(value, list):
                    _DataNodeBuilder__Logger.debug('No processing Property as key - key - %s, key type - %s, value - %s, length %s'%(key,type(_key),type(value),len(value)))
                else:
                    _DataNodeBuilder__Logger.debug('No processing Property as key - key - %s, key type - %s, value - %s'%(key,type(_key),type(value)))

                self.collectTime(start)
                return dict.__setitem__(self, _key, value)
        elif isinstance(_key, str):
            key = _key
        else:
            _DataNodeBuilder__Logger.error('key - %s, type %s is not supported'%(_key,type(_key)))
            self.collectTime(start)
            return
        
        skey = key.split('/')
        
        # Check if the key with this value has been set recently
        # If not create a new sensor for it
        # If yes stop __setitem__ and return - the same value for the key was added last time that key was set
        if not getProductionModeStatus() and self.__addSensorCounters(skey, value):
            self.myLoggerTime.info(f"DataNode Builder skipping repeated value for sensor  - {skey}")
            self.collectTime(start)
            return # Stop __setitem__ for repeated key value combination
        
        if not getProductionModeStatus():
            if isinstance(value, torch.Tensor):
                _DataNodeBuilder__Logger.info('key - %s, key type - %s, value - %s, shape %s'%(key,type(_key),type(value),value.shape))
            elif isinstance(value, list):
                _DataNodeBuilder__Logger.info('key - %s, key type - %s, value - %s, length %s'%(key,type(_key),type(value),len(value)))
            else:
                _DataNodeBuilder__Logger.info('key - %s, key type - %s, value - %s'%(key,type(_key),type(value)))

        if value is None:
            _DataNodeBuilder__Logger.error('The value for the key %s is None - abandon the update'%(key))
            self.collectTime(start)
            return dict.__setitem__(self, _key, value)
                
        if len(skey) < 2:            
            _DataNodeBuilder__Logger.error('The key %s has only two elements, needs at least three - abandon the update'%(key))
            self.collectTime(start)
            return dict.__setitem__(self, _key, value)
        
        usedGraph = dict.__getitem__(self, "graph")

        # Find if the key include concept from graph
        
        graphPathIndex = usedGraph.cutGraphName(skey)
        keyWithoutGraphName = skey[graphPathIndex:]
        graphPath =  ''.join(map(str, skey[:graphPathIndex])) 
       
        # Check if found concept in the key
        if not keyWithoutGraphName:
            _DataNodeBuilder__Logger.warning('key - %s has not concept part - returning'%(key))
            self.collectTime(start)
            return dict.__setitem__(self, _key, value)
            
        # Find description of the concept in the graph
        if isinstance(_key, Sensor):
            try:
                conceptName = _key.concept.name 
            except TypeError as _:
                conceptName = keyWithoutGraphName[0]
        else:
            conceptName = keyWithoutGraphName[0]
        concept = self.__findConcept(conceptName, usedGraph)
                
        if not concept:
            _DataNodeBuilder__Logger.warning('conceptName - %s has not been found in the used graph %s - returning'%(conceptName,usedGraph.fullname))
            self.collectTime(start)
            return dict.__setitem__(self, _key, value)
        
        conceptInfo = self.__findConceptInfo(usedGraph, concept)
        
        if isinstance(_key, Sensor):
            self.__updateConceptInfo(usedGraph, conceptInfo, _key)

        DataNodesConcepts = dict.__getitem__(self, "DataNodesConcepts")
        # Only build the datanode if it is not a full skeleton mode or if the first initial datanode has not been created yet
        if not self.skeletonDataNodeFull or ("dataNode" not in self):
            # Only build the datanode if it is not a skeleton mode or if the datanodes for the given concept have not been created yet
            # or if the concept is a relation
            if (not self.skeletonDataNode) or (conceptName not in DataNodesConcepts) or ("relationName" in conceptInfo):
                # Create key for DataNode construction
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
                    _DataNodeBuilder__Logger.debug('%s found in the graph; it is a concept'%(conceptName))
                    index = self.__buildDataNode(vInfo, conceptInfo, keyDataName)   # Build or update Data node
                    
                    if index:
                        indexKey = graphPath  + '/' + conceptName + '/index'
                        dict.__setitem__(self, indexKey, index)
                        from collections.abc import Sequence
                        if self.skeletonDataNode:
                            allDns = self.setdefault("allDns", set())

                            # Add the index to the "allDns" set
                            try:
                                if isinstance(index[0], Sequence):
                                    index = index[0]
                                allDns.update(index)
                            except TypeError as ty:
                                pass
                        
                        DataNodesConcepts[conceptName] = index
                        #dict.__setitem__(self, "DataNodesConcepts", DataNodesConcepts)
                    
                    if conceptInfo['relation']:
                        _DataNodeBuilder__Logger.debug('%s is a relation'%(conceptName))
                        self.__buildRelationLink(vInfo, conceptInfo, keyDataName) # Build or update relation link
                    
        if self.skeletonDataNode:
            if conceptName in skey:
                # Find the index of "conceptName" in skey
                index = skey.index(conceptName)

                # Join "conceptName" with the next element in skey
                keyInRootDataNode = "/".join(skey[index:index+2])
           
                # Add "/label" to the key if the concept has a label marked
                if conceptInfo['label']:
                    keyInRootDataNode += "/label"

                # Check if the key contains "<" and add it to the variable set if it does, otherwise add it to the property set
                if "<" in keyInRootDataNode:
                    self.__addVariableNameToSet((_key, keyInRootDataNode))
                else:
                    self.__addPropertyNameToSet((_key, keyInRootDataNode))
            else:
                # throw an exception
                raise Exception("The key does not contain conceptName")
                    
        # Add key to the list of keys in order
        if self.skeletonDataNodeFull:
            KeysInOrder = dict.__getitem__(self, "KeysInOrder")
            KeysInOrder.append(_key)
        
        # Add value to the underling dictionary
        r = dict.__setitem__(self, _key, value)
        
        if not r:
            pass # Error when adding entry to dictionary ?
        
        self.collectTime(start)
        return r                
                                             
    def __delitem__(self, key):
        return dict.__delitem__(self, key)

    def __contains__(self, key):
        return dict.__contains__(self, key)
    
    # Add or increase generic counter counting number of the getitem method calls
    def __addGetDataNodeCounter(self):
        counterName = 'Counter' + 'GetDataNode'
        if not dict.__contains__(self, counterName):
            dict.__setitem__(self, counterName, 1)
        else:
            currentCounter =  dict.__getitem__(self, counterName)
            dict.__setitem__(self, counterName, currentCounter + 1)
            
    def findDataNodesInBuilder(self, select=None, indexes=None):
        existingRootDns = dict.__getitem__(self, 'dataNode') # DataNodes roots
            
        if not existingRootDns:
            foundDns = []
        else:
            foundDns = existingRootDns[0].findDatanodes(dns=existingRootDns, select=select, indexes=indexes)
            
        return foundDns
        
    def createFullDataNode(self, rootDataNode):
        if not self.skeletonDataNodeFull:
            return # Do nothing if not in full skeleton mode

        startCreateFullDataNode = perf_counter()
        self.skeletonDataNodeFull = False # Set temporary flag to False to allow creation of full dataNode
        
        keysInOrder = dict.__getitem__(self, "KeysInOrder")
        
        for key in keysInOrder:
            # Run the original values through __setitem__ to build the full dataNode
            self.__setitem__(key, dict.__getitem__(self, key))
            
        if self.skeletonDataNode:
            # Get the "allDns" set from the data node, or create a new empty set if it doesn't exist
            allDns = self.get("allDns", set())
    
            # Iterate over the data nodes in "allDns" and add the "rootDataNode" attribute to them
            for dn in allDns:
                if dn == rootDataNode:
                    continue
                dn.attributes["rootDataNode"] = rootDataNode
            
        self.skeletonDataNodeFull = True # Return flag to the original 
        
        endCreateFullDataNode = perf_counter()
        elapsedCreateFullDataNode = (endCreateFullDataNode - startCreateFullDataNode) * 1000
        self.myLoggerTime.info(f'Creating Full Datanode: {elapsedCreateFullDataNode}ms')   
        
    def createBatchRootDN(self):
        if dict.__contains__(self, 'dataNode'):
            existingDns = dict.__getitem__(self, 'dataNode')
            if len(existingDns) == 1:
                rootDn = existingDns[0]
                if not getProductionModeStatus():
                    _DataNodeBuilder__Logger.info(f'No new Batch Root DataNode created - DataNode Builder already has single Root DataNode with id {rootDn.instanceID} of type {rootDn.getOntologyNode().name}')
                return
                
            # Check if there are more than one type of DataNodes in the builder
            typesInDNs = set()
            for i, d in enumerate(existingDns):
                typesInDNs.add(d.getOntologyNode().name)
            
            # If there are more than one type of DataNodes in the builder, then it is not possible to create new Batch Root DataNode
            if len(typesInDNs) > 1:
                _DataNodeBuilder__Logger.warn('DataNode Builder has DataNodes of different types: %s, not possible to create batch Datanode' % (typesInDNs))
                return
                
            # Create the Batch Root DataNode
            supGraph = existingDns[1].getOntologyNode().sup
            if supGraph is None:
                raise ValueError('Not able to create Batch Root DataNode - existing DataNodes in the Builder have concept type %s not connected to any graph: %s'%(typesInDNs))  

            batchRootDNValue = ""
            batchRootDNID = 0
            batchRootDNOntologyNode = Concept(name='batch')
            supGraph.attach(batchRootDNOntologyNode)
            
            batchRootDN = DataNode(myBuilder = self, instanceID = batchRootDNID, instanceValue = batchRootDNValue, ontologyNode = batchRootDNOntologyNode)
        
            for i, d in enumerate(existingDns):
                batchRootDN.addChildDataNode(d)  
            
            # The new Root DataNode it the batch Root DataNode
            self.__updateRootDataNodeList([batchRootDN])

            if not getProductionModeStatus():
                _DataNodeBuilder__Logger.info('Created single Batch Root DataNode with id %s of type %s'%(batchRootDNID,batchRootDNOntologyNode))
            self.myLoggerTime.info('Created single Batch Root DataNode with id %s of type %s'%(batchRootDNID,batchRootDNOntologyNode))
        else:
            raise ValueError('DataNode Builder has no DataNode started yet')
        
    # Method returning constructed DataNode - the fist in the list
    def getDataNode(self, context="interference", device='auto'):
        self.__addGetDataNodeCounter()
        
        if context=="interference":
            if self.skeletonDataNode:
                self.myLoggerTime.info("DataNode Builder is using skeleton datanode mode")
            if 'Counter' + '_setitem' in self:
                self.myLoggerTime.info("DataNode Builder the set method called - %i times"%(self['Counter' + '_setitem']))
            if 'DataNodeTime' in self:
                # self['DataNodeTime'] is in nanoseconds, so divide by 1000000 to get milliseconds
                elapsedInMsDataNodeBuilder = sum(self['DataNodeTime'])/1000000
                self.myLoggerTime.info(f"DataNode Builder time usage - {elapsedInMsDataNodeBuilder:.5f}ms")
                
                #self.myLoggerTime.info(f"DataNode Builder elapsed time in ns - {self['DataNodeTime']}")
                #self.myLoggerTime.info(f"DataNode Builder start time in ns - {self['DataNodeTime_start']}")
                #self.myLoggerTime.info(f"DataNode Builder end time in ns - {self['DataNodeTime_end']}")

        # If DataNode it created then return it
        if dict.__contains__(self, 'dataNode'):
            existingDns = dict.__getitem__(self, 'dataNode')
            
            if len(existingDns) != 0:
                returnDn = existingDns[0]
                
                # Set the torch device
                returnDn.current_device = device
                if returnDn.current_device == 'auto': # if not set use cpu or cuda if available
                    returnDn.current_device = 'cpu'
                    if torch.cuda.is_available():
                        returnDn.current_device = 'cuda'
                        
                if len(existingDns) != 1:
                    typesInDNs = {d.getOntologyNode().name for d in existingDns[1:]}
                    _DataNodeBuilder__Logger.warning(f'Returning first dataNode with id {returnDn.instanceID} of type {returnDn.getOntologyNode().name} - there are total {len(existingDns)} dataNodes of types {typesInDNs}')
                    self.myLoggerTime.info(f'Returning first dataNode with id {returnDn.instanceID} of type {returnDn.getOntologyNode().name} - there are total {len(existingDns)} dataNodes of types {typesInDNs}')
                else:
                    if not getProductionModeStatus():
                        _DataNodeBuilder__Logger.info(f'Returning dataNode with id {returnDn.instanceID} of type {returnDn.getOntologyNode().name}')
                    self.myLoggerTime.info(f'Returning dataNode with id {returnDn.instanceID} of type {returnDn.getOntologyNode().name}')
                    
                if self.skeletonDataNode:
                    # Get the "variableSet" dictionary from the data node, or create a new empty dictionary if it doesn't exist
                    variableSet = self.get("variableSet", {})

                    # Create a dictionary of the items in "variableSet" with the keys and values swapped
                    variableSetDict = {k2: self[k1] for k1, k2 in dict(variableSet).items()}
                    
                    # Add the "variableSet" dictionary to the return data node attributes
                    returnDn.attributes["variableSet"] = variableSetDict

                    # Get the "propertySet" dictionary from the data node, or create a new empty dictionary if it doesn't exist
                    propertySet = self.get("propertySet", {})

                    # Create a dictionary of the items in "propertySet" 
                    propertySetDict = {k2: self[k1] for k1, k2 in dict(propertySet).items()}

                    # Add the "propertySet" dictionary to the return data node attributes
                    returnDn.attributes["propertySet"] = propertySetDict

                    # Get the "allDns" set from the data node, or create a new empty set if it doesn't exist
                    allDns = self.get("allDns", set())

                    # Iterate over the data nodes in "allDns" and add the "rootDataNode" attribute to them
                    for dn in allDns:
                        if dn == returnDn:
                            continue
                        dn.attributes["rootDataNode"] = returnDn
        
                return returnDn
        
        _DataNodeBuilder__Logger.error('Returning None - there are no dataNode')
        return None
    
    # Method returning all constructed DataNodes 
    def getBatchDataNodes(self):
        self.__addGetDataNodeCounter()
        
        if 'Counter' + '_setitem' in self:
            self.myLoggerTime.info("DataNode Builder the set method called - %i times"%(self['Counter' + '_setitem' ]))
        if 'DataNodeTime' in self:
            # self['DataNodeTime'] is in nanoseconds, so divide by 1000000 to get milliseconds
            elapsedInMsDataNodeBuilder = sum(self['DataNodeTime'])/1000000
            self.myLoggerTime.info(f"DataNode Builder time usage - {elapsedInMsDataNodeBuilder:.5f}ms")
        
        if dict.__contains__(self, 'dataNode'):
            existingDns = dict.__getitem__(self, 'dataNode')
            
            if len(existingDns) > 0:  
                
                if not getProductionModeStatus():
                    _DataNodeBuilder__Logger.info('Returning %i dataNodes - %s'%(len(existingDns),existingDns))

                return existingDns
        
        _DataNodeBuilder__Logger.error('Returning None - there are no dataNodes')
        return None