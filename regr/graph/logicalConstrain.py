import logging
from itertools import product
 
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
            
                self.lcName = "LC%i"%(len(context.logicalConstrains))
                context.logicalConstrains[self.lcName] = self
        
        self.myLogger = logging.getLogger(ilpConfig['log_name'])
          
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

        if len(v) != 2:
            self.myLogger.error("%s Logical Constrain created with %i sets of variables but should be with exactly 2 sets"%(lcName, len(v)))
            return ilpV
        
        concepts = [concept for concept in v] 
               
        commonVarName = None
        for conceptIndex, _ in enumerate(concepts):
            if conceptIndex + 2 > len(concepts):
                break
            
            for varName in v[concepts[conceptIndex]]:
                if varName in v[concepts[conceptIndex+1]]:
                    commonVarName = varName
                    break
            
        if not commonVarName:
            return None
        
        for concept in concepts:
            ilpV[concept] = {}
            ilpV[concept][resultVariableName] = {}
            
            tokens = [token for token in v[concept][commonVarName]]
    
            for token in tokens:
                _ilpV = []
            
                tokenVarSet = [v[concept][commonVarName][token] for concept in concepts if token in v[concept][commonVarName]]
                
                for varS in product(*tokenVarSet):
                    if None in varS:
                        continue
                    
                    _ilpV.append(lcFun(model, *varS, onlyConstrains = headConstrain))
                        
                ilpV[concept][resultVariableName][token] = _ilpV
        
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