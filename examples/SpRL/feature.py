import spacy
import networkx as nx
from spacy.matcher import PhraseMatcher

nlpmodel=spacy.load("en_core_web_sm")
spatial_dict = []
with open("examples/SpRL/data/spatial_dic.txt") as f_sp:
    for line in f_sp:
        spatial_dict.append(line.strip())
    

class DataFeature_for_sentence():

    def __init__(self,sentence):
        self.sentence = sentence
        #self.phrase = phrase

        self.parse_sentence = nlpmodel(self.sentence)
        #self.parse_phrase = nlpmodel(self.phrase)

    def getChunks(self):
        
        pre_chunk=nlpmodel(self.sentence)
        new_chunk=[]
        # read dict
        matcher = PhraseMatcher(nlpmodel.vocab)
        patterns = [nlpmodel.make_doc(text) for text in spatial_dict]
        matcher.add("TerminologyList", None, *patterns)
        matches = matcher(pre_chunk)
        for match_id, start, end in matches:
            span = pre_chunk[start:end]
            new_chunk.append(span)            
        preposition_span = [pre_chunk[token.i:token.i+1] for token in pre_chunk if token.pos_ == "ADP" ]
        for each_span in preposition_span:
            if each_span not in new_chunk:
                new_chunk.append(each_span)
        for chunk in pre_chunk.noun_chunks:
           new_chunk.append(chunk)
        return new_chunk

    def getSentence(self):
        return self.sentence

class DataFeature_for_span():
    def __init__(self, span):
        self.span = span

    #get tokens
    def getPhraseTokens(self):
        # docs = nlpmodel(self.phrase)
        num=0
        for token in self.span:
            num+=1
        span=self.span[0:num]
        return span.merge()


    def getLower(self):
        return self.span.text.lower()

    def getUpper(self):
        return self.span.text.upper()


    # headword
    def getHeadword(self):
        if len(list(self.span.noun_chunks))==0:
            return self.span.text
        else:
            for doc in self.span.noun_chunks:
                return str(doc.root.text).lower()


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

        # with self.span.text.retokenize() as retokenizer:
        #     retokenizer.merge(self.span[0:len(self.span.text)])
        # for doc in iter(self.span):
        #     phrasepos=doc.pos_
        return self.span.root.pos_

       
        #return phrasepos

    def getShortestDependencyPath(self, entity1, entity2):
        edges = []
        for token in self.parse_sentence:
            for child in token.children:
                edges.append(('{0}'.format(token.lower_),
                              '{0}'.format(child.lower_)))
        graph = nx.Graph(edges)

        return nx.shortest_path(graph, source=entity1, target=entity2)

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


#sentence="About 20 kids in traditional clothing and hats waiting on stairs along the left side of ."
#phrase = ''
#phrase="in the front of
# entity1 = 'Convulsions'.lower()
# entity2 = 'fever'
#data=DataFeature(sentence,phrase)
#for i in data.getChunks():
   #print(i.text)
   #print(i.start)
#print(data.getHeadword())
# print(data.getShortestDependencyPath(entity1, entity2))

