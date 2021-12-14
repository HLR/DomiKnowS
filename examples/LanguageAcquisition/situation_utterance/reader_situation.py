import os

import sys
#sys.path.append('../..')
##print("sys.path - %s"%(sys.path))


from regr.data.reader import RegrReader

class InteractionReader(RegrReader):
    
    def getsituationval(self, item):
        return item['situation']

    def getutteranceval(self, item):
        return item['utterance']
    
    def parse_file(self):
        
        object_list = []
        
        size = 1

        with open(self.file) as f:
            content = f.readlines()
        for l in content[:size]:
            item = {}
            substring = l.strip().split("\t")
            item["situation"] = [substring[0].split()]
            item["utterance"] = [substring[1].split()]
            
            object_list.append(item)
        return object_list
    
    
class PredicateReader(RegrReader):
    
    def __init__(self,filename,extension):
        
        super().__init__(filename,extension)
        
        self.color_lst = ["re1", "bl1", "ye1", "pu1", "or1", "gr1"]
        self.shape_lst = ["sq1","tr1","el1","he1","st1","ci1"]
        self.size_lst = ['bi1',"sm1","me1"]
        self.pos_lst = ['ab2','le2']
        
        self.categories = ['color', 'shape', 'size', 'position']
        self.words = [x.strip() for x in open("../data/vocabulary.txt")]
    
    def getpredicateval(self, item):
        return item['predicate']

    def getcategoryval(self, item):
        
        return [self.categories.index(item['category'][0])]
        
    def getwordval(self, item):
        
        return [self.words.index(item['word'][0])]
        
    def parse_file(self):
        
        object_list = []
        
        limit = 50
        count = 0

        with open(self.file) as f:
            content = f.readlines()
        for l in content:
            
            if count >= limit:
                break
            
            item = {}
            substring = l.split("\t")
            
            
            color_lst = ["re1", "bl1", "ye1", "pu1", "or1", "gr1"]
            shape_lst = ["sq1","tr1","el1","he1","st1","ci1"]
            size_lst = ['bi1',"sm1","me1"]
            pos_lst = ['ab2','le2']
            
            # Remove stop words from the utterance
            utterance = substring[1]
            for w in ["the","to", "of"]:
                utterance = utterance.replace(w,"")
            
            # Need to find the first instance of color 
            # item = {}
            for c,w in zip(substring[0].split(), utterance.split()):
                
                # Reset the item for each element in the situation
                item = {}
                
                if c[:3] in color_lst:
                    item["predicate"] = [c[:3]]
                    item["word"] = [w]
                    item['category'] = ['color']
                    
                elif c[:3] in shape_lst:
                    item["predicate"] = [c[:3]]
                    item["word"] = [w]
                    item['category'] = ['shape']
                    
                elif c[:3] in size_lst:
                    item["predicate"] = [c[:3]]
                    item["word"] = [w]
                    item['category'] = ['size']
                    
                elif c[:3] in pos_lst:
                    item["predicate"] = [c[:3]]
                    item["word"] = [w]
                    item['category'] = ['position']
                    
                    
                
                if item != {}:    
                    count += 1
                    
                    # print(item)
                    object_list.append(item)
            
            
        return object_list
    
    
# filename = "./data/training_set.txt"
# train_reader = PredicateReader(filename,"txt")

# print(list(iter(train_reader)))