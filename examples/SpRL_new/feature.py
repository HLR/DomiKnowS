import spacy
from allennlp.data.tokenizers import Token
nlpmodel=spacy.load("en_core_web_sm")



class DataFeature():

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
        num=0
        for token in docs:
            num+=1
        span=docs[0:num]
        return span.merge()


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
        pos=[]
        for doc in self.docs:
            pos.append(doc.pos_)
        return '|'.join(pos)

    #tag feature
    def getTag(self):
        tag = []
        for doc in self.docs:
            tag.append(doc.tag_)
        return "|".join(tag)

    # lemma feature
    def getLemma(self):
        lemma = []
        for doc in self.docs:
            lemma.append(doc.lemma_)
        return "|".join(lemma)

    #dependenceyrelation
    def getDenpendency(self):
        dependenceyRelation=[]
        for doc in self.docs:
            dependenceyRelation.append( doc.dep_ )
        return "|".join(dependenceyRelation)

    #phrasetag
    def getPhrasetag(self):
        phrasetag=''
        with self.docs.retokenize() as retokenizer:
            retokenizer.merge(self.docs[0:len(self.phrase)])
        for doc in self.docs:
            phrasetag=doc.tag_
        return phrasetag

    #wordform
    def getWordform(self):
        return self.getLower()

    #semantic role
    def getSemanticRle(self):
        pass


#
# sentence=['fantastic car','new cars','about 20 years old']
# newtokens=[]
# for word in sentence:
#     tokens=DataFeature(word).getTokens()
#     pos__=DataFeature(word).getPos()
#     tokens.set_extension("pos_", default=False, force=True)
#     tokens._.set('pos_',pos__)
#     newtokens.append(tokens)
# for i in newtokens:
#     print(i)
#     print(i._.pos_)


#
# data=DataFeature(phrase)
# postag=data.getTokens()
# print(postag)

