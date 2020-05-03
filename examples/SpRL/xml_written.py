from xml.dom.minidom import Document
import xml.etree.ElementTree as ET
import json

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

#s(id,text)
#landmark [（1,(start1, end1）), df2]
#trajector [df3, df4]
#spatial_indicator [df5, df6]

# region[(1), 2 df5)]
# distance[(df2, df4, df6)]
doc = Document()
sprl = doc.createElement("SpRL")
doc.appendChild(sprl)

def writeXML(doc_num, sent_list):
    scene = doc.createElement("SCENE")
    sprl.appendChild(scene)
    
    for each_sent in sent_list:
        each_sentence = doc.createElement("SENTENCE")
        scene.appendChild(each_sentence)
        text = doc.createElement('TEXT')
        each_sentence.appendChild(text)
        for key in each_sent.keys():
            if key != "RELATION":
                for each_entity in each_sent.get(key):
                    entity = doc.createElement(key)
                    each_sentence.appendChild(entity)
                    entity.setAttribute('id', each_entity[0])
                    entity.setAttribute('start', each_entity[1][0])
                    entity.setAttribute('end', each_entity[1][1])
            else:
                 for each_rel in each_sent.get(key):
                    relation = doc.createElement(key)
                    each_sentence.appendChild(relation)
                    relation.setAttribute('id', each_rel[0])
                    relation.setAttribute('trajctor_id', each_rel[2][0])
                    relation.setAttribute('landmark_id', each_rel[2][1])
                    relation.setAttribute('spatial_indicator_id', each_rel[2][1])
                    relation.setAttribute('general_type', each_rel[1])
                    relation.setAttribute('specific_type',"")
                    relation.setAttribute("RCC8_value", "")
                    relation.setAttribute('FoR', "")






# for each_sentence in sentence:
#     if landmark:


#     if trajector:

#     if spatial_indicator:
    
#     if region 

#     if distance

#     if direction


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
    doc_sent_list = get_doc_sent("data/new_gold.xml")
    #doc should contain several sentence
    doc_num = 'annotations/01/1071.eng'
    sentence_list = []
    #sentence1
    sentence = {}
    landmark = [("l1",('1', '2')),("l2",('3', '4'))]
    trajector = [("t1",('5', '6')),("t2",('7', '8'))]
    spatial_indicator = [("s1", ('9', '10')),("s2",('11', '12'))]
    relation_list = [("r1", "region",("l1","t1","s1")),("r2","direction",("l2","t2","s2"))]
    sentence["LANDMARK"] = landmark
    sentence['TRAJCTOR'] = trajector
    sentence['SPATIALINDICATOR'] = spatial_indicator
    sentence['RELATION'] = relation_list
    sentence_list.append(sentence)
    
    writeXML(doc_num, sentence_list)
    filename = "people.xml"
    f = open(filename, "w")
    f.write(doc.toprettyxml(indent="  "))
    
