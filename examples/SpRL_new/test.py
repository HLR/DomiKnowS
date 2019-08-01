import spacy

nlpmodel=spacy.load("en_core_web_sm")

class Data():

    def __init__(self,anystr):
        self.sentence=anystr
        self.phrase=anystr
        self.docs=nlpmodel(anystr)

    def getChunks(self):
        pre_chunk=nlpmodel(self.sentence)
        new_chunk=[]
        for chunk in pre_chunk.noun_chunks:
            new_chunk.append(chunk.text)
        return new_chunk

    def getSentence(self):
        pass

    #get tokens
    def getTokens(self):
        docs = nlpmodel(self.phrase)
        tokens=[]
        for token in docs:
           tokens.append(token)
        return tokens


    def getLower(self):
        return self.phrase.lower()

    def getUpper(self):
        return self.phrase.upper()


    # headword
    def getHeadword(self):
       for doc in self.docs.noun_chunks:
           return str(doc.root.text).lower()

    #pos feature
    def getPos(self):
        pos=''
        for doc in self.docs:
            pos+=doc.pos_+"|"
        return pos

    #tag feature
    def getTag(self):
        tag = ''
        for doc in self.docs:
            tag += doc.tag_ + "|"
        return tag

    # lemma feature
    def getLemma(self):
        lemma = ''
        for doc in self.docs:
            lemma += doc.lemma_ + "|"
        return lemma

    #dependenceyrelation
    def getDenpendencyRelation(self):
        dependenceyRelation= ''
        for doc in self.docs:
            dependenceyRelation += doc.dep_ + "|"
        return dependenceyRelation

    #phrasetag
    def getPhrasetag(self):
        phrasetag=''
        with self.docs.retokenize() as retokenizer:
            retokenizer.merge(self.docs[0:len(phrase)])
        for doc in self.docs:
            phrasetag=doc.tag_
        return phrasetag

    #wordform
    def getWordform(self):
        return self.getLower()

    #semantic role
    def getSemanticRle(self):
        pass


phrase="fantastic car"
data=Data(phrase)
postag=data.getDenpendencyRelation()
print(postag)