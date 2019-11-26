from itertools import product, permutations
import logging

from regr.solver.gurobiILPBooleanMethods import gurobiILPBooleanProcessor
from regr.solver.ilpConfig import ilpConfig 

class LogicalConstrain:
    def __init__(self, *e):
        self.e = e
        
        if e:
            contexts = self._getContext(e[0])
            if contexts:
                context = contexts[-1]
            
                lcName = "LC%i"%(len(context.logicalConstrains))
                context.logicalConstrains[lcName] = self
        
        self.myLogger = logging.getLogger(ilpConfig['log_name'])
          
    def __call__(self, model, myIlpBooleanProcessor, v): 
        pass 
            
    def _getContext(self, e):
        if isinstance(e, LogicalConstrain):
            return self._getContext(e.e[0])
        else:
            return e._context
    
class notL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        notV = list()
        
        self.myLogger.debug("Not Logical Constrain created with : %s"%(v))
        
        for currrentS in v:
            for currrentV in currrentS:
                notV.append(myIlpBooleanProcessor.notVar(model, currrentV, onlyConstrains = headConstrain))
                
        if not headConstrain:
            self.myLogger.debug("Not Logical Constrain result : %s"%(notV))
        return notV
          
class andL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        andV = list()
        
        for currentV in product(*v):     
            andV.append(myIlpBooleanProcessor.andVar(model, *currentV, onlyConstrains = headConstrain))
            
        return andV

class orL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        orV = list()
        
        for currentV in product(*v):     
            orV.append(myIlpBooleanProcessor.andVar(model, *currentV, onlyConstrains = headConstrain))
            
        return orV
        
class ifL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
    
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        
        self.myLogger.debug("If Logical Constrain created with : %s"%(v))

        if len(v) != 2:
            self.myLogger.error("If Logical Constrain created with %i set of variable - should be with exactly 2"%(len(v)))
            return {}
        
        ifV = list()
        
        for currentV in product(*v):  
            ifV.append(myIlpBooleanProcessor.ifVar(model, *currentV, onlyConstrains = headConstrain))
        
        if not headConstrain:
            self.myLogger.debug("If Logical Constrain result : %s"%(existsV))
        return ifV
        
class existsL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        existsV = list()
        
        self.myLogger.debug("Exists Logical Constrain created with : %s"%(v))

        for currentS in v:
            existsV.append(myIlpBooleanProcessor.orVar(model, *currentS, onlyConstrains = headConstrain))
    
        if not headConstrain:
            self.myLogger.debug("Exists Logical Constrain result : %s"%(existsV))
        return existsV
        