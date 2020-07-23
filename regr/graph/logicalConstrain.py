import logging
from itertools import product
 
#from regr.solver.gurobiILPBooleanMethods import gurobiILPBooleanProcessor
from regr.solver.ilpConfig import ilpConfig 

class LogicalConstrain:
    def __init__(self, *e):
        self.active = True
        
        for e_item in e:
            if isinstance(e_item, LogicalConstrain):
                e_item.active = False
                
        self.e = e
        
        if e:
            contexts = self.__getContext(e[0])
            if contexts:
                context = contexts[-1]
            
                self.lcName = "LC%i"%(len(context.logicalConstrains))
                context.logicalConstrains[self.lcName] = self
        
        self.myLogger = logging.getLogger(ilpConfig['log_name'])
        self.ifLog =  ilpConfig['ifLog']
        
    def __str__(self):
        return self.__class__.__name__
          
    def __call__(self, model, myIlpBooleanProcessor, v): 
        pass 
            
    def __getContext(self, e):
        if isinstance(e, LogicalConstrain):
            return self.__getContext(e.e[0])
        else:
            return e._context
        
    def createILPConstrains(self, lcName, lcFun, model, myIlpBooleanProcessor, v, resultVariableName='Final', headConstrain = False):
        ilpV = {}

        #if self.ifLog: self.myLogger.debug("%s Logical Constrain invoked with variables: %s"%([[[x.VarName for _, x in t.items()] for _, t in v1.items()] for v1 in v]))

        if len(v) < 2:
            self.myLogger.error("%s Logical Constrain created with %i sets of variables which is less then two"%(lcName, len(v)))
            return ilpV
        
        zVars = {}
        vKeys = [next(iter(v[i])) for i in range(len(v))]
        
        if  vKeys[0] ==  vKeys[1]:
            for v1 in v[0][vKeys[0]]:
                tokenVars = []
                
                for i in range(len(v)):
                    currentVar = v[i][vKeys[i]][v1]
                    tokenVars.append(currentVar)
                    
                zVars[(*v1, )] = lcFun(model, *tokenVars, onlyConstrains = headConstrain)
                
            ilpKey = (*vKeys[0], )
        else: # Support only 2 elements now !
            for v1 in v[0][vKeys[0]]:
                for v2 in v[1][vKeys[1]]:        
                    
                    if len(v1) > len(v2): 
                        if (vKeys[0][0] == vKeys[1][0]) and (v1[0] != v2[0]):
                                continue
                        elif (vKeys[0][1] == vKeys[1][0]) and (v1[1] != v2[0]):
                            continue
                        
                    v1Var = v[0][vKeys[0]][v1]
                    v2Var = v[1][vKeys[1]][v2]
                    
                    zVars[(*v1, *v2)] = lcFun(model, v1Var, v2Var, onlyConstrains = headConstrain)
                        
            ilpKey = (*vKeys[0], *vKeys[1])
        
        ilpV[ilpKey] = zVars
        
        if  headConstrain:
            if self.ifLog: self.myLogger.debug("%s Logical Constrain is the head constrain - only ILP constrain created"%(lcName))
        else:
            #if self.ifLog: self.myLogger.debug("%s Logical Constrain result - ILP variables created : %s"%(lcName,[x.VarName for x in ilpV]))
            pass

        model.update()
        
        return ilpV
    
class notL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableName='Final', headConstrain = False): 
        notV = []
        _notV = {}
        notV.append(_notV)
            
        if self.ifLog: self.myLogger.debug("Not Logical Constrain invoked with variables: %s"%([[[x.VarName for _, x in t.items()] for _, t in v1.items()] for v1 in v]))
        
        if len(v) > 1:
            self.myLogger.error("Not Logical Constrain created with %i sets of variables which is more then one"%(len(v)))
            return notV
        
        _v = v[0]
        for currentVar in _v:
            _notV[resultVariableName] = {}

            for currentToken in _v[currentVar]:
                currentILPVar = _v[currentVar][currentToken]
                
                notVar= myIlpBooleanProcessor.notVar(model, currentILPVar, onlyConstrains = headConstrain)
        
                _notV[resultVariableName][currentToken] = notVar
                   
        if headConstrain:
            if self.ifLog: self.myLogger.debug("Not Logical Constrain is the head constrain - only ILP constrain created")
        else:
            if self.ifLog: self.myLogger.debug("Not Logical Constrain result - ILP variables created: %s"%([x.VarName for x in notV]))
            
        model.update()
        
        return notV
          
class existsL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableName='Final', headConstrain = False): 
        existsV = []
        _existsV = {}
        existsV.append(_existsV)
        
        if self.ifLog: self.myLogger.debug("Exists Logical Constrain invoked with variables: %s"%([[[x.VarName for _, x in t.items()] for _, t in v1.items()] for v1 in v]))

        _v = v[0]
        existsVars = []
        for currentVar in _v:
            _existsV[resultVariableName] = {}

            for currentToken in _v[currentVar]:
                existsVars.append(_v[currentVar][currentToken])
        
            existsVarResult = myIlpBooleanProcessor.orVar(model, *existsVars, onlyConstrains = headConstrain)
                
            _existsV[resultVariableName][(0,)] = [existsVarResult]

        if  headConstrain:
            if self.ifLog: self.myLogger.debug("Exists Logical Constrain is the head constrain - only ILP constrain created")
        else:
            if self.ifLog: self.myLogger.debug("Exists Logical Constrain result - ILP variables created: %s"%([x.VarName for x in existsV]))
                 
        model.update()
        
        return existsV
    
class andL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableName='Final', headConstrain = False): 
        return self.createILPConstrains('And', myIlpBooleanProcessor.andVar, model, myIlpBooleanProcessor, v, resultVariableName, headConstrain)

class orL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableName='Final', headConstrain = False):
        return self.createILPConstrains('Or', myIlpBooleanProcessor.orVar, model, myIlpBooleanProcessor, v, resultVariableName, headConstrain)
    
class nandL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableName='Final', headConstrain = False): 
        return self.createILPConstrains('Nand', myIlpBooleanProcessor.nandVar, model, myIlpBooleanProcessor, v, resultVariableName, headConstrain)
        
class ifL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
    
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableName='Final', headConstrain = False): 
        return self.createILPConstrains('If', myIlpBooleanProcessor.ifVar, model, myIlpBooleanProcessor, v, resultVariableName, headConstrain)
    
class equalA(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)

class inSetA(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
