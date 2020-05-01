import spacy
import networkx as nx
from spacy.matcher import PhraseMatcher

nlpmodel=spacy.load("en_core_web_sm")
spatial_dict = []
with open("data/spatial_dic.txt") as f_sp:
    for line in f_sp:
        spatial_dict.append(line.strip())
    

class DataFeature_for_sentence():
    def __init__(self, sentence):
        self.sentence = sentence
        #self.phrase = phrase

        self.parse_sentence = nlpmodel(self.sentence)
        #self.parse_phrase = nlpmodel(self.phrase)

        self.dummy = DataFeature_for_span.dummy(self)

    def __repr__(self):
        return self.sentence

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, type(self)):
            return self.sentence == other.sentence
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def getSpan(self, start, end):
        if start < 0 and end < 0:
            return self.dummy
        return DataFeature_for_span(self, start, end)

    def getChunks(self):
        def token2df(token):
            assert token.doc == self.parse_sentence
            start = token.idx
            end = token.idx + len(token)
            return DataFeature_for_span(self, start, end)
        def span2df(span):
            assert span.doc == self.parse_sentence
            start = span.start_char
            end = span.end_char
            return DataFeature_for_span(self, start, end)

        pre_chunk=self.parse_sentence
        new_chunk=[self.dummy]
        # read dict
        matcher = PhraseMatcher(nlpmodel.vocab)
        patterns = [nlpmodel.make_doc(text) for text in spatial_dict]
        matcher.add("TerminologyList", None, *patterns)
        matches = matcher(pre_chunk)
        for match_id, start, end in matches:
            span = pre_chunk[start:end]
            new_chunk.append(span2df(span))
        preposition_span = [token2df(token) for token in pre_chunk if token.pos_ == "ADP" ]
        for each_span in preposition_span:
            if each_span not in new_chunk:
                new_chunk.append(each_span)
        for chunk in pre_chunk.noun_chunks:
            new_chunk.append(span2df(chunk))
        #new_chunk.sorted(key=lambda chunk: chunk.start)

        return new_chunk
    
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
    @staticmethod
    def dummy(doc):
        return DataFeature_for_span(doc, -1, -1)

    def __init__(self, doc, start, end):
        self.doc = doc
        self.start = start
        self.end = end

        if self.start < 0 and self.end < 0:
            self.init_dummy()
        else:
            self.init_feature()

    def init_dummy(self):
        self.token_start = 0
        self.token_end = len(self.doc.parse_sentence)
        self.text = '__DUMMY__'
        self.lemma_ = '__DUMMY__'
        self.pos_ = '__DUMMY__'
        self.tag_ = '__DUMMY__'
        self.dep_ = '__DUMMY__'
        self.headword_ = '__DUMMY__'
        self.phrasepos_ = '__DUMMY__'
        self.lower_ = '__DUMMY__'
        self.upper_ = '__DUMMY__'
        self.is_dummy_ = True

    def init_feature(self):
        span, self.token_start, self.token_end = self.findSpan()

        self.text = self.getText(span)
        self.lemma_ = self.getLemma(span)
        self.pos_ = self.getPos(span)
        self.tag_ = self.getTag(span)
        self.dep_ = self.getDenpendency(span)
        self.headword_ = self.getHeadword(span)
        self.phrasepos_ = self.getPhrasepos(span)
        self.lower_ = self.getLower(span)
        self.upper_ = self.getUpper(span)
        self.is_dummy_ = False
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

    def findSpan(self):
        token_i = []
        for token in self.doc.parse_sentence:
            if self.start <= token.idx and token.idx + len(token) <= self.end:
                token_i.append(token.i)
        token_start = min(token_i)
        token_end = max(token_i) + 1
        span = self.doc.parse_sentence[token_start:token_end]
        return span, token_start, token_end

    def __repr__(self):
        return self.text

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, type(self)):
            return self.start == other.start and self.end == other.end and self.doc == other.doc
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    #get tokens
    # def getPhraseTokens(self):
    #     # docs = nlpmodel(self.phrase)
    #     num=0
    #     for token in self.span:
    #         num+=1
    #     span=self.span[0:num]
    #     return span.merge()

    @property
    def _span(self):
        return self.doc[self.token_start:self.token_end]

    def getText(self, span):
        return span.text

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

if __name__ == '__main__':
    df = DataFeature_for_sentence('people are walking through the park , others are crossing the road in the foreground .')
    chunks = df.getChunks()
    print(chunks)
