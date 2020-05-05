from xml.dom.minidom import Document
import xml.etree.ElementTree as ET
import json
from xml.dom import minidom




def readjsonl(path):
    docs = {}
    with open(path) as fin:
        for line in fin:
            data = json.loads(line)
            if data['docno'] not in docs:
                docs[data['docno']] = {
                    'docno': data['docno'],
                    'image': data['image'],
                    'sentences': []
                }
            docs[data['docno']]['sentences'].append(data)
    return docs

#doc = Document()
# doc = ET.Element('SpRL')

sprl = ET.Element('SpRL')
#sprl = doc.createElement("SpRL")
#doc.appendChild(sprl)

def writeXML(doc_num, image_num, sent_list):
    
    scene = ET.SubElement(sprl, "SCENE")
    doc_num_xml = ET.SubElement(scene, "DOCNO")
    doc_num_xml.text = doc_num
    image_num_xml = ET.SubElement(scene, "IMAGE")
    image_num_xml.text = image_num
    
    for json_sent in sent_list:
        # xml_sent = doc.createElement("SENTENCE")
        # scene.appendChild(xml_sent)
        xml_sent = ET.SubElement(scene, "SENTENCE")
        xml_sent.set('id', json_sent['id'])
        xml_sent.set('start', json_sent['start'])
        xml_sent.set('end', json_sent['end'])
        sentence_text = ET.SubElement(xml_sent, "TEXT")
        sentence_text.text = json_sent['text']
        entity_names = {'LANDMARK': "LANDMARK", 'TRAJECTOR':"TRAJECTOR", "SPATIAL_INDICATOR":'SPATIALINDICATOR'}
        relation_names = ['region', 'direction', 'distance']
        for json_entity in entity_names.keys():
            if json_sent.get(json_entity):
                for key, value in json_sent.get(json_entity).items():
                    xml_entity = ET.SubElement(xml_sent, entity_names[json_entity])
                    xml_entity.set('id', key)
                    xml_entity.set('start', str(value[0]))
                    xml_entity.set('end', str(value[1]))
                    xml_entity.set('text', sentence_text.text[value[0]:value[1]])
            
        for each_rel in relation_names:
            if json_sent.get(each_rel):
                for key, value in json_sent.get(each_rel).items():
                    relation = ET.SubElement(xml_sent,"RELATION")
                    relation.set('id', key)
                    relation.set('landmark_id', str(value[0]))
                    relation.set('trajector_id', str(value[1]))
                    relation.set('spatial_indicator_id', str(value[2]))
                    relation.set('general_type', each_rel)
                    relation.set('specific_type',"na")
                    relation.set("RCC8_value", "na")
                    relation.set('FoR', "na")

def get_doc_sent(sprlxmlfile):

    sprlXMLTree = ET.parse(sprlxmlfile)
    sprlXMLRoot = sprlXMLTree.getroot()
    doc_list = []              
    for sentenceItem in sprlXMLRoot.findall('./SCENE'):
        sentence_list = []
        for child in sentenceItem:
            if child.tag == "DOCNO":
                docid = child.text
            if child.tag == "SENTENCE":
                sentence_list.append(child.attrib['id'])
        doc_list.append((docid, sentence_list))
    return doc_list
    
             
if __name__ == "__main__":
    #doc should contain several sentence
    docs = readjsonl('data/json_generation.jsonl')
    for key, value in docs.items():
        doc_num = key
        image_num = value['image']
        sentence_list = value['sentences']
        writeXML(doc_num, image_num, sentence_list)
    filename = "xml_generation.xml"
    xmlstr = minidom.parseString(ET.tostring(sprl)).toprettyxml(indent="   ")
    with open(filename, "w") as f:
        f.write(xmlstr)
    # tree = ET.ElementTree(doc)
    # tree.write(filename)
    
   #
   #  f = open(filename, "w")
    #f.write(doc.toprettyxml(indent="  "))
    #doc.write(filename)
   # f.close()
