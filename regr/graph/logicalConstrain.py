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
        #if self.ifLog: self.myLogger.debug("%s Logical Constrain invoked with variables: %s"%([[[x.VarName for _, x in t.items()] for _, t in v1.items()] for v1 in v]))

        if len(v) < 2:
            self.myLogger.error("%s Logical Constrain created with %i sets of variables which is less then two"%(lcName, len(v)))
            return None
        
        # input variable names
        vKeys = [next(iter(v[i])) for i in range(len(v))]
        
        # output variable names and variables
        ilpV = {} # output variable names 
        zVars = {} # output variables
        
        if len(v) == 2:
            # Depending on size of variable names tuples from input variables
            if len(vKeys[0]) == 1: # First variables names has one element
                if len(vKeys[1]) == 1: # Second variables names has one elements
                    if vKeys[0][0] == vKeys[1][0]: # The same variable names
                        ilpKey = vKeys[0]
                        
                        for v1 in v[0][vKeys[0]]:
                            v1Var = v[0][vKeys[0]][v1]
                            v2Var = v[1][vKeys[1]][v1]
                                
                            zVars[v1] = lcFun(model, v1Var, v2Var, onlyConstrains = headConstrain)
                    else: # Different variables names
                        ilpKey = (vKeys[0][0], vKeys[1][0])
                        for v1 in v[0][vKeys[0]]:
                            for v2 in v[1][vKeys[1]]:
                                v1Var = v[0][vKeys[0]][v1]
                                v2Var = v[1][vKeys[1]][v2]
                                    
                                zVars[v1] = lcFun(model, v1Var, v2Var, onlyConstrains = headConstrain)
                elif len(vKeys[1]) == 2: # Second variables names has two elements
                    if vKeys[0][0] in vKeys[1]:
                        ilpKey = vKeys[1]
                            
                        for v1 in v[1][vKeys[1]]:
                           
                            v1Var = v[0][vKeys[0]][(v1[0],)] 
                            v2Var = v[1][vKeys[1]][v1]
                                
                            zVars[v1] = lcFun(model, v1Var, v2Var, onlyConstrains = headConstrain)
                    else:
                        pass
                else:
                    pass # Support only 2 elements now !
            elif len(vKeys[0]) == 2:
                if len(vKeys[1]) == 1: 
                    if vKeys[0][0] == vKeys[1][0]: # First name match
                        ilpKey = vKeys[0]
                        
                        for v1 in v[0][vKeys[0]]:
                           
                            v1Var = v[0][vKeys[0]][v1]
                            v2Var = v[1][vKeys[1]][(v1[0],)]
                                
                            zVars[v1] = lcFun(model, v1Var, v2Var, onlyConstrains = headConstrain)
                    elif vKeys[0][1] == vKeys[1][0]: # Second name match
                        ilpKey = vKeys[0]
                        
                        for v1 in v[0][vKeys[0]]:
                           
                            v1Var = v[0][vKeys[0]][v1]
                            v2Var = v[1][vKeys[1]][(v1[1],)]
                                
                            zVars[v1] = lcFun(model, v1Var, v2Var, onlyConstrains = headConstrain)
                    else:  # Different variables names
                        pass
                elif len(vKeys[1]) == 2:
                    if (vKeys[0] == vKeys[1]):
                        ilpKey = vKeys[0]
                    else:
                        pass
                else:
                    pass # Support only 2 elements now !
            else:
                pass # Support only 2 elements in key now !
        
        if len(v) > 2: # Support only up to 3 keys !
            equals = True
            for k in vKeys:
                if  len(k) > 1 or vKeys[0][0] != k[0]:
                    equals = False
                    break
                
                if equals:
                    ilpKey = vKeys[0]
                    
                    for v1 in v[0][vKeys[0]]:
                        _vars = []
                        for i, _ in enumerate(vKeys):
                            _vars.append(v[i][vKeys[0]][v1])
                         
                        zVars[v1] = lcFun(model, *_vars, onlyConstrains = headConstrain)

        # Output
        ilpV[ilpKey] = zVars
        model.update()

        return ilpV
   
        # --------------    ???
        if  headConstrain:
            #if self.ifLog: self.myLogger.debug("% Logical Constrain is the head constrain - only ILP constrain created"%(lcName))
            pass
        else:
            #if self.ifLog: self.myLogger.debug("%s Logical Constrain result - ILP variables created : %s"%(lcName,[x.VarName for x in ilpV]))
            pass

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
        #if self.ifLog: self.myLogger.debug("Exists Logical Constrain invoked with variables: %s"%([[[x.VarName for _, x in t.items()] for _, t in v1.items()] for v1 in v]))
        existsV = []
        _existsV = {}
        existsV.append(_existsV)
        
        _v = v[0]
        _vKey = next(iter(_v))
        _v=next(iter(_v.values()))
        
        if isinstance(resultVariableName, str):
            existsVars = []
            for currentToken in _v:
                existsVars.append(_v[currentToken])
        
            existsVarResult = myIlpBooleanProcessor.orVar(model, *existsVars, onlyConstrains = headConstrain)
                
            _existsV[resultVariableName] = {}
            _existsV[resultVariableName][(0,)] = [existsVarResult]
        elif isinstance(resultVariableName, tuple):
            if len(resultVariableName) == 1:
                if resultVariableName[0] in _vKey:
                    n1Index = _vKey.index(resultVariableName[0]) 
                    
                    existsVars = {}
                    for currentToken in _v:
                        if currentToken[0] in existsVars:
                            existsVars[currentToken[0]].append(_v[currentToken])
                        else:
                            existsVars[currentToken[0]] = [_v[currentToken]]
                        
                    _existsV[resultVariableName] = {}     
                    for k in existsVars:
                        existsVarResult = myIlpBooleanProcessor.orVar(model, *existsVars[k], onlyConstrains = headConstrain)
        
                        _existsV[resultVariableName][(k,)] = [existsVarResult]
                else:
                    pass
            elif len(resultVariableName) == 2:
                pass
            elif len(resultVariableName) == 3:
                pass
            else:
                pass

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
