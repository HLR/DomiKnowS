import spacy
import networkx as nx
from spacy.matcher import PhraseMatcher

nlpmodel=spacy.load("en_core_web_sm")
spatial_dict = []
with open("data/spatial_dic.txt") as f_sp:
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
        return list(map(DataFeature_for_span, new_chunk))
    
    def getShortestDependencyPath(self, entity1, entity2):
        try:
            edges = []
            for token in self.parse_sentence:
                for child in token.children:
                    edges.append(('{0}'.format(token.lower_),
                                '{0}'.format(child.lower_)))
            graph = nx.Graph(edges)

            return nx.shortest_path(graph, source=entity1, target=entity2)
        except:
            return [entity2]

    def getSentence(self):
        return self.sentence

class DataFeature_for_span():
    def __init__(self, span):
        self.start = span.start
        self.end = span.end
        self.doc = span.doc
        self.lemma_ = self.getLemma(span)
        self.pos_ = self.getPos(span)
        self.tag_ = self.getTag(span)
        self.dep_ = self.getDenpendency(span)
        self.headword_ = self.getHeadword(span)
        self.phrasepos_ = self.getPhrasepos(span)
        self.lower_ = self.getLower(span)
        self.upper_ = self.getUpper(span)
        # self.span.set_extension("lemma_", default=False, force=True)
        # self.span.set_extension('pos_', default=False, force=True)
        # self.span.set_extension('tag_', default=False, force=True)
        # self.span.set_extension('dep_', default=False, force=True)
        # self.span.set_extension('headword_', default=False, force=True)
        #self.span.set_extension('phrasepos_', default=False, force=True)
        # self.span.set_extension('sentence_', default=False, force=True)
        # self.span._.set('lemma_', self.getLemma())
        # self.span._.set('pos_', self.getPos())
        # self.span._.set('tag_', self.getTag())
        # self.span._.set('dep_', self.getDenpendency())
        # self.span._.set('headword_', self.getHeadword())
        #self.span._.set('phrasepos_', self.getPhrasepos())
        #self.span._.set('sentence_', self.getSentence())

    #get tokens
    # def getPhraseTokens(self):
    #     # docs = nlpmodel(self.phrase)
    #     num=0
    #     for token in self.span:
    #         num+=1
    #     span=self.span[0:num]
    #     return span.merge()

    @property
    def span(self):
        return self.doc[self.start:self.end]

    def getLower(self, span):
        return span.text.lower()

    def getUpper(self, span):
        return span.text.upper()


    # headword
    def getHeadword(self, span):
        if len(list(span.noun_chunks))==0:
            return span.text
        else:
            for doc in span.noun_chunks:
                return str(doc.root.text).lower()


    #pos feature
    def getPos(self, span):
        # newpos=[]
        pos = []
        # for token in self.parse_sentence:
        #     newpos.append((token.text,token.pos_))

        # for phrase_token in self.parse_phrase:
        #     for new_p in newpos:
        #         if phrase_token.text == new_p[0]:
        #             pos.append(new_p[1])
        for each_token in span:
            pos.append(each_token.pos_)
        return '|'.join(pos)

    #tag feature
    def getTag(self, span):
        #newtag = []
        tag = []
        # for token in self.parse_sentence:
        #     newtag.append((token.text, token.tag_))

        # for phrase_token in self.parse_phrase:
        #     for new_t in newtag:
        #         if phrase_token.text == new_t[0]:
        #             tag.append(new_t[1])
        for each_token in span:
            tag.append(each_token.tag_)
        return '|'.join(tag)

    # lemma feature
    def getLemma(self, span):
        lemma = []
        for each_token in span:
            lemma.append(each_token.lemma_)
        return "|".join(lemma)

    #dependenceyrelation
    def getDenpendency(self, span):
        #newdependency = []
        dependency = []
        # for token in self.parse_sentence:
        #     newdependency.append((token.text, token.dep_))

        # for phrase_token in self.parse_phrase:
        #     for new_t in newdependency:
        #         if phrase_token.text == new_t[0]:
        #             dependency.append(new_t[1])
        for each_token in span:
            dependency.append(each_token.dep_)
        return '|'.join(dependency)

    #phrasetag
    def getPhrasepos(self, span):

        # with self.span.text.retokenize() as retokenizer:
        #     retokenizer.merge(self.span[0:len(self.span.text)])
        # for doc in iter(self.span):
        #     phrasepos=doc.pos_
        return span.root.pos_

       
        #return phrasepos



    #wordform
    def getWordform(self, span):
        return span.getLower()

    #semantic role
    def getSemanticRle(self, span):
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


#sentence="a tall high-rise building with a facade made of glass ."
#phrase = ''
#phrase="in the front of
#entity1 = 'high-rise'.lower()
#entity2 = 'of'
#data=DataFeature_for_sentence(sentence)
#for i in data.getChunks():
   #print(i.text)
   #print(i.start)
#print(data.getHeadword())
#print(data.getShortestDependencyPath(entity1, entity2))

