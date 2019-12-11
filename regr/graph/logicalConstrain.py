import logging

from regr.solver.gurobiILPBooleanMethods import gurobiILPBooleanProcessor
from regr.solver.ilpConfig import ilpConfig 

ifLog = False

class LogicalConstrain:
    def __init__(self, *e):
        self.e = e
        
        if e:
            contexts = self.__getContext(e[0])
            if contexts:
                context = contexts[-1]
            
                lcName = "LC%i"%(len(context.logicalConstrains))
                context.logicalConstrains[lcName] = self
        
        self.myLogger = logging.getLogger(ilpConfig['log_name'])
          
    def __call__(self, model, myIlpBooleanProcessor, v): 
        pass 
            
    def __getContext(self, e):
        if isinstance(e, LogicalConstrain):
            return self.__getContext(e.e[0])
        else:
            return e._context
        
    def createILPConstrains(self, lcName, lcFun, model, myIlpBooleanProcessor, v, headConstrain = False):
        ilpV = {}

        if ifLog: self.myLogger.debug("%s Logical Constrain invoked with variables: %s"%(lcName, [[x.VarName for x in v1] for v1 in v]))

        if len(v) != 2:
            self.myLogger.error("%s Logical Constrain created with %i sets of variables but should be with exactly 2 sets"%(lcName, len(v)))
            return ilpV
        
        commonVarName = None
        
        vKey = [key for key in v] 
        
        for varName in v[vKey[0]]:
            if varName in v[vKey[1]]:
                commonVarName = varName
                break
            
        if not commonVarName:
            return None
        
        for token in v[vKey[0]][commonVarName]:
            _ilpV = []
            
            for var1 in v[vKey[0]][commonVarName][token]:
                if not var1:
                    continue
                
                for var2 in v[vKey[1]][commonVarName][token]:
                    if not var2:
                        continue
                    
                    _ilpV.append(lcFun(model, var1, var2, onlyConstrains = headConstrain))
                    
            ilpV['concept', token] = _ilpV
        
        if  headConstrain:
            if ifLog: self.myLogger.debug("% Logical Constrain is the head constrain - only ILP constrain created"%(lcName))
        else:
            if ifLog: self.myLogger.debug("%s Logical Constrain result - ILP variables created : %s"%(lcName,[x.VarName for x in ilpV]))

        return ilpV
    
class notL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        notV = list()
        
        if ifLog: self.myLogger.debug("Not Logical Constrain invoked with variables: %s"%([[x.VarName for x in v1] for v1 in v]))
        
        for currrentS in v:
            for currrentV in currrentS:
                notV.append(myIlpBooleanProcessor.notVar(model, currrentV, onlyConstrains = headConstrain))
                
        if  headConstrain:
            if ifLog: self.myLogger.debug("Not Logical Constrain is the head constrain - only ILP constrain created")
        else:
            if ifLog: self.myLogger.debug("Not Logical Constrain result - ILP variables created: %s"%([x.VarName for x in notV]))
            
        return notV
          
class existsL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        existsV = {}
        
        if ifLog: self.myLogger.debug("Exists Logical Constrain invoked with variables: %s"%([[x.VarName for x in v1] for v1 in v]))

        existsVar = []
        for currentConcept in v:
            for currentVar in v[currentConcept]:
                for currentToken in v[currentConcept, currentVar]:
                    existsVar.extend(v[currentConcept, currentVar, currentToken])
            
            existsV[currentConcept, token] = myIlpBooleanProcessor.orVar(model, *existsVar, onlyConstrains = headConstrain)
    
        if  headConstrain:
            if ifLog: self.myLogger.debug("Exists Logical Constrain is the head constrain - only ILP constrain created")
        else:
            if ifLog: self.myLogger.debug("Exists Logical Constrain result - ILP variables created: %s"%([x.VarName for x in existsV]))
                 
        return existsV
    
class andL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createILPConstrains('And', myIlpBooleanProcessor.andVar, model, myIlpBooleanProcessor, v, headConstrain)

class orL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False):
        return self.createILPConstrains('Or', myIlpBooleanProcessor.orVar, model, myIlpBooleanProcessor, v, headConstrain)
    
class nandL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createILPConstrains('Nand', myIlpBooleanProcessor.nandVar, model, myIlpBooleanProcessor, v, headConstrain)
        
class ifL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
    
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createILPConstrains('If', myIlpBooleanProcessor.ifVar, model, myIlpBooleanProcessor, v, headConstrain)