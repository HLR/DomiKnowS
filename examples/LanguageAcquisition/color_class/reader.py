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

        with open(self.file) as f:
            content = f.readlines()
        for l in content:
            item = {}
            substring = l.split("\t")
            item["situation"] = substring[0].split()
            item["utterance"] = substring[1].split()
            
            object_list.append(item)
        return object_list
    
    
class PredicateReader(RegrReader):
    
    def getPredval(self, item):
        return item['pred']

    def getRedval(self, item):
        
        if item['label'][0] == 'red':
            return [1]
        else:
            return [0]
        
    def getBlueval(self, item):
        
        if item['label'][0] == 'blue':
            return [1]
        else:
            return [0]
        
    def getYellowval(self, item):
        
        if item['label'][0] == 'yellow':
            return [1]
        else:
            return [0]
        
    def getPurpleval(self, item):
        
        if item['label'][0] == 'purple':
            return [1]
        else:
            return [0]
        
    def getOrangeval(self, item):
        
        if item['label'][0] == 'orange':
            return [1]
        else:
            return [0]
        
    def getGreenval(self, item):
        
        if item['label'][0] == 'green':
            return [1]
        else:
            return [0]
        
    def parse_file(self):
        
        object_list = []
        
        limit = 500
        count = 0

        with open(self.file) as f:
            content = f.readlines()
        for l in content:
            
            if count >= limit:
                break
            
            item = {}
            substring = l.split("\t")
            
            
            color_lst = ["re1", "bl1", "ye1", "pu1", "or1", "gr1"]
            #color_lst = ["re1", "bl1"]
            
            # Remove stop words from the utterance
            utterance = substring[1]
            for w in ["the","to", "of"]:
                utterance = utterance.replace(w,"")
            
            # Need to find the first instance of color 
            
            for c,w in zip(substring[0].split(), utterance.split()):
                item = {}
                if c[:3] in color_lst:
                    item["pred"] = [c[:3]]
                    item["label"] = [w]    
                    
            
                if item != {}:    
                    count += 1
                    
                    # print(item)
                    object_list.append(item)
            
            
        return object_list
    
    
# filename = "./data/training_set.txt"
# train_reader = PredicateReader(filename,"txt")

# print(list(iter(train_reader)))