import spacy
from allennlp.data.tokenizers import Token
nlpmodel=spacy.load("en_core_web_sm")



class DataFeature():

    def __init__(self,sentence,phrase):
        self.sentence = sentence
        self.phrase = phrase

        self.parse_sentence = nlpmodel(self.sentence)
        self.parse_phrase = nlpmodel(self.phrase)

    def getChunks(self):
        pre_chunk=nlpmodel(self.sentence)
        new_chunk=[]
        for chunk in pre_chunk.noun_chunks:
            new_chunk.append(chunk)
        return new_chunk

    def getSentence(self):
        pass


    #get tokens
    def getPhraseTokens(self):
        # docs = nlpmodel(self.phrase)
        num=0
        for token in self.parse_phrase:
            num+=1
        span=self.parse_phrase[0:num]
        return span.merge()


    def getLower(self):
        return self.phrase.lower()

    def getUpper(self):
        return self.phrase.upper()


    # headword
    def getHeadword(self):
       if len(list(self.parse_phrase.noun_chunks))==0:
           return self.phrase
       else:
           for doc in self.parse_phrase.noun_chunks:
                    return str(doc.root.head.text).lower()

    #pos feature
    def getPos(self):
        newpos=[]
        pos = []
        for token in self.parse_sentence:
            newpos.append((token.text,token.pos_))

        for phrase_token in self.parse_phrase:
            for new_p in newpos:
                if phrase_token.text == new_p[0]:
                    pos.append(new_p[1])
        return '|'.join(pos)

    #tag feature
    def getTag(self):
        newtag = []
        tag = []
        for token in self.parse_sentence:
            newtag.append((token.text, token.tag_))

        for phrase_token in self.parse_phrase:
            for new_t in newtag:
                if phrase_token.text == new_t[0]:
                    tag.append(new_t[1])
        return '|'.join(tag)

    # lemma feature
    def getLemma(self):
        lemma = []
        for phrase in self.parse_phrase:
            lemma.append(phrase.lemma_)
        return "|".join(lemma)

    #dependenceyrelation
    def getDenpendency(self):
        newdependency = []
        dependency = []
        for token in self.parse_sentence:
            newdependency.append((token.text, token.dep_))

        for phrase_token in self.parse_phrase:
            for new_t in newdependency:
                if phrase_token.text == new_t[0]:
                    dependency.append(new_t[1])
        return '|'.join(dependency)

    #phrasetag
    def getPhrasepos(self):
        phrasepos=''
        with self.parse_phrase.retokenize() as retokenizer:
            retokenizer.merge(self.parse_phrase[0:len(self.phrase)])
        for doc in self.parse_phrase:
            phrasepos=doc.pos_
        return phrasepos

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


# sentence="About 20 kids in traditional clothing and hats waiting on stairs ."
# phrase="About 20 kids ."
# data=DataFeature(sentence,phrase)
# print(data.getHeadword())

