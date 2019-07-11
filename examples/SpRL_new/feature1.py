import spacy
nlpmodel=spacy.load("en_core_web_sm")



def getChunk(sentence):
    pre_chunk = nlpmodel(sentence)
    new_chunk = []
    for chunk in pre_chunk.noun_chunks:
        new_chunk.append(chunk.text)
    return new_chunk

def getHeadwords(phrase):
    docs=nlpmodel(phrase)

    for doc in docs.noun_chunks:
           return str(doc.root.text)

# a=getHeadwords('fantastic cars')
# print(a)