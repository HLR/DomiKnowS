import xml.etree.ElementTree as ET
from regr.graph import DataNode

from spGraph import *

spatial_tags = ('LANDMARK', 'TRAJECTOR', 'SPATIALINDICATOR')
relation_tags = ('RELATION')

def parseSprlXMLIntoDataNodes(sprlXmlfile): 
  
    # Parse the xml tree object 
    sprlXMLTree = ET.parse(sprlXmlfile) 
  
    # Get root of the xml tree 
    sprlXMLRoot = sprlXMLTree.getroot() 
  
    sentence_dataNodes = []

    # Iterate over sentences
    for sentenceItem in sprlXMLRoot.findall('./SCENE/SENTENCE'): 
        sentenceValue = None
        phrases = {}
        relations = {}
        
        # Iterate child elements of sentence to find phrases 
        for child in sentenceItem: 
            if child.tag == 'TEXT': 
                sentenceValue = child.text
            elif child.tag in spatial_tags: # Find phrase
                childText = None
                
                if "text" in child.attrib:
                    childText = child.attrib["text"]
                    
                # Create phrase Data Node 
                phrases[child.attrib["id"]] = DataNode(instanceID = child.attrib["id"], instanceValue = childText, ontologyNode = phrase, childInstanceNodes = [], attributes = {'general_type' : child.tag})

        # Iterate child elements of sentence to find relations 
        for child in sentenceItem: 
            if child.tag in relation_tags:
                if "id" in child.attrib:
                    relationAttributes = {} # collect relation attributes
                    for attr in  child.attrib:
                        if not attr == "id":
                            relationAttributes[attr] = child.attrib[attr]
                            
                    # Create Relation Data Node - link with data nodes for phrases related by this relation
                    relations[child.attrib["id"]] = DataNode(instanceID = child.attrib["id"], ontologyNode = spatial_triplet, 
                                                             childInstanceNodes = [phrases[child.attrib["trajector_id"]], 
                                                                                   phrases[child.attrib["landmark_id"]], 
                                                                                   phrases[child.attrib["spatial_indicator_id"]]],
                                                             attributes = relationAttributes)

        # Collect all the child nodes
        sentenceChildInstanceNodes = [i for i in phrases.items()]
        for relationItem in relations.items():
            sentenceChildInstanceNodes.append(relationItem)
            
        # Create sentence Data Node
        sentenceDataNode = DataNode(instanceID = sentenceItem.attrib["id"], instanceValue = sentenceValue, ontologyNode = sentence, childInstanceNodes = sentenceChildInstanceNodes)
        
        sentence_dataNodes.append(sentenceDataNode)

    # Return list of data nods for sentences
    return sentence_dataNodes