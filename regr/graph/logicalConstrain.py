import logging
from itertools import product
 
#from regr.solver.gurobiILPBooleanMethods import gurobiILPBooleanProcessor
from regr.solver.ilpConfig import ilpConfig 

ifLog = False

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

        if ifLog: self.myLogger.debug("%s Logical Constrain invoked with variables: %s"%(lcName, [[x.VarName for x in v1] for v1 in v]))

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
        else:
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
            if ifLog: self.myLogger.debug("% Logical Constrain is the head constrain - only ILP constrain created"%(lcName))
        else:
            if ifLog: self.myLogger.debug("%s Logical Constrain result - ILP variables created : %s"%(lcName,[x.VarName for x in ilpV]))

        return ilpV
    
class notL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableName='Final', headConstrain = False): 
        notV = {}
        
        if ifLog: self.myLogger.debug("Not Logical Constrain invoked with variables: %s"%([[x.VarName for x in v1] for v1 in v]))
        
        for currentConcept in v:
            notV[currentConcept] = {}

            for currentVar in v[currentConcept]:
                for currentToken in v[currentConcept][currentVar]:
                    notVar = []

                    notV[currentConcept][resultVariableName] = {}
                    for currentILPVar in v[currentConcept][currentVar][currentToken]:
                        notVar.append(myIlpBooleanProcessor.notVar(model, currentILPVar, onlyConstrains = headConstrain))
            
                    notV[currentConcept][resultVariableName][currentToken] = notVar
                   
        if  headConstrain:
            if ifLog: self.myLogger.debug("Not Logical Constrain is the head constrain - only ILP constrain created")
        else:
            if ifLog: self.myLogger.debug("Not Logical Constrain result - ILP variables created: %s"%([x.VarName for x in notV]))
            
        return notV
          
class existsL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableName='Final', headConstrain = False): 
        existsV = {}
        
        if ifLog: self.myLogger.debug("Exists Logical Constrain invoked with variables: %s"%([[x.VarName for x in v1] for v1 in v]))

        for currentConcept in v:
            existsVar = []
            tokensInConcept = []
            existsV[currentConcept] = {}

            for currentVar in v[currentConcept]:
                for currentToken in v[currentConcept][currentVar]:
                    existsVar.extend(v[currentConcept][currentVar][currentToken])
                    tokensInConcept.append(currentToken)
            
            existsVarResult = myIlpBooleanProcessor.orVar(model, *existsVar, onlyConstrains = headConstrain)
            
            existsV[currentConcept][resultVariableName] = {}
            for currentToken in tokensInConcept:
                existsV[currentConcept][resultVariableName][currentToken] = [existsVarResult]

        if  headConstrain:
            if ifLog: self.myLogger.debug("Exists Logical Constrain is the head constrain - only ILP constrain created")
        else:
            if ifLog: self.myLogger.debug("Exists Logical Constrain result - ILP variables created: %s"%([x.VarName for x in existsV]))
                 
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
