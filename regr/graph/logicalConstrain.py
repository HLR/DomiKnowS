import logging

import numpy
import torch
from  collections import namedtuple
from regr.solver.ilpConfig import ilpConfig 
   
myLogger = logging.getLogger(ilpConfig['log_name'])
ifLog =  ilpConfig['ifLog']
        
V = namedtuple("V", ['name', 'match', 'v'], defaults= [None, None, None])

class LogicalConstrain:
    def __init__(self, *e, p=100):
        self.headLC = True # Indicate that it is head constrain and should be process individually
        
        if not e:
            myLogger.error("Logical Constrain initialized is empty")
            raise LogicalConstrain.LogicalConstrainError("Logical Constrain initialized is empty")
        
        self.e = e

        # Find the graph for this logical constrain - based on context defined in the concept used in constrain definition
        self.graph = None
        from regr.graph import Concept
        conceptOrLc = None
        
        if isinstance(e[0], (Concept, LogicalConstrain)):
            conceptOrLc = e[0]
        elif len(e) > 1 and isinstance(e[1], (Concept, LogicalConstrain)): 
            conceptOrLc = e[1]
        elif len(e) > 2 and isinstance(e[2], (Concept, LogicalConstrain)):
            conceptOrLc = e[2]
        else:
            myLogger.error("Logical Constrain is incorrect")
            raise LogicalConstrain.LogicalConstrainError("Logical Constrain is incorrect")
            
        if conceptOrLc != None:
            if isinstance(conceptOrLc, Concept):
                if self.__getContext(conceptOrLc):
                    self.graph = self.__getContext(conceptOrLc)[-1]
            else:
                self.graph = conceptOrLc.graph
                
        if self.graph == None:
            myLogger.error("Logical Constrain initialized is not associated with graph")
            raise LogicalConstrain.LogicalConstrainError("Logical Constrain initialized is not associated with graph")
                     
        # Create logical constrain id based on number of existing logical constrain in the graph
        self.lcName = "LC%i"%(len(self.graph.logicalConstrains))
        
        # Add the constrain to the graph
        self.graph.logicalConstrains[self.lcName] = self
                
        # Go though constrain, find nested constrains and change their property headLC to indicate that their are nested and should not be process individually
        for e_item in e:
            if isinstance(e_item, LogicalConstrain):
                e_item.headLC = False
                
        # Check soft constrain is activated though p - if certainty in the validity of the constrain or the user preference is provided by p
        if p < 0:
            self.p = 0
            myLogger.warning("%s Logical Constrain created with p equal %i sets it to 0"%(self.lcName,p))
        elif p > 100:
            self.p = 100
            myLogger.warning("%s Logical Constrain created with p equal %i sets it to 100"%(self.lcName,p))
        else:
            self.p = p
     
    class LogicalConstrainError(Exception):
        pass
    
    def __conceptVariableCount(self, e):
        variableTypes = []
        
        if len(e.has_a()) > 0: 
            for has_a in e.has_a():
                variableTypes.append(has_a.dst)
                
        if len(variableTypes) == 2:
            return 2, variableTypes
        elif len(variableTypes) == 3:
            return 3, variableTypes
        elif len(variableTypes) > 3:
            self.myLogger.error('Not supporting relation with more then 3 attributes - the relation %s has %i attributes'%(e,len(variableTypes)))
            raise ValueError('Not supporting relation with more then 3 attributes - the relation %s has %i attributes'%(e,len(variableTypes)))
                    
        for is_a in e.is_a():
            parentConcept = is_a.dst
            
            parentType, parentRelationConcepts = self.__conceptVariableCount(parentConcept)
            
            if len(parentRelationConcepts) > 0:
                return parentType, parentRelationConcepts
            
        return 1, variableTypes
    
    def __str__(self):
        return self.__class__.__name__
    
    def __repr__(self):
        return  self.lcName + '(' + self.__class__.__name__ + ')'
          
    def __call__(self, model, myIlpBooleanProcessor, v): 
        pass 
            
    def __getContext(self, e):
        if isinstance(e, LogicalConstrain):
            return self.__getContext(e.e[0])
        else:
            return e._context
       
    def createILPConstrains(self, lcName, lcFun, model, v, resultVariableNames=None, headConstrain = False):
        if len(v) < 2:
            myLogger.error("%s Logical Constrain created with %i sets of variables which is less then two"%(lcName, len(v)))
            return None
        
        # input variable names
        try:
            vKeys = [e for e in iter(v)]
        except StopIteration:
            pass
        
        # output variable names and variables
        zVars = {} # output variables
        
        if len(v) == 2:
            for i in range(len(v[vKeys[0]])):
                    v1Var = v[vKeys[0]][i]
                    v2Var = v[vKeys[1]][i]
                        
                    if v1Var is None or v2Var is None:
                        zVars[i] = None
                    else:
                        zVars[i] = lcFun(model, v1Var, v2Var, onlyConstrains = headConstrain)
                        
            return zVars

# ------------------
            if resultVariableNames and len(resultVariableNames) and resultVariableNames[0] == resultVariableNames[1]:             
                for i in range(len(v[vKeys[0]])):
                    v1Var = v[vKeys[0]][i]
                    v2Var = v[vKeys[1]][i]
                        
                    if v1Var is None or v2Var is None:
                        zVars[i] = None
                    else:
                        zVars[i] = lcFun(model, v1Var, v2Var, onlyConstrains = headConstrain)
            else:
                len1 = len(v[vKeys[0]])
                
                if len(v[vKeys[0]]) != len(v[vKeys[1]]):
                    return None
                    
                for i in range(len1):
                    for j in range(len1):
                        v1Var = v[vKeys[0]][i]
                        v2Var = v[vKeys[1]][j]
                            
                        if v1Var is None or v2Var is None:
                            zVars[i*len1 + j] = None
                        else:
                            zVars[i*len1 + j] = lcFun(model, v1Var, v2Var, onlyConstrains = headConstrain)
                            
        return zVars
    
    def oldCreateILPConstrains(self, lcName, lcFun, model, v, resultVariableNames=None, headConstrain = False):
        if ifLog: myLogger.debug("%s Logical Constrain invoked with variables: %s"%(lcName, [[[x if not isinstance(x, torch.Tensor) else x.VarName for _, x in t.items()] for _, t in v1.items()] for v1 in v]))
        
        if len(v) < 2:
            myLogger.error("%s Logical Constrain created with %i sets of variables which is less then two"%(lcName, len(v)))
            return None
        
        # input variable names
        try:
            vKeys = [next(iter(v[i])) for i in range(len(v))]
        except StopIteration:
            pass
        
        # output variable names and variables
        ilpV = {} # output variable names 
        zVars = {} # output variables
        ilpKey = None
        
        if len(v) == 2:
            # Depending on size of variable names tuples from input variables
            if len(vKeys[0]) == 1: # First variables names has one element
                if len(vKeys[1]) == 1: # Second variables names has one elements
                    if vKeys[0][0] == vKeys[1][0]: # The same variable names
                        ilpKey = vKeys[0]
                        
                        for v1 in v[0][vKeys[0]]:
                            v1Var = v[0][vKeys[0]][v1]
                            v2Var = v[1][vKeys[1]][v1]
                                
                            if v1Var is None or v2Var is None:
                                zVars[v1] = None
                            else:
                                zVars[v1] = lcFun(model, v1Var, v2Var, onlyConstrains = headConstrain)
                    else: # Different variables names
                        ilpKey = (vKeys[0][0], vKeys[1][0])
                        
                        for v1 in v[0][vKeys[0]]:
                            for v2 in v[1][vKeys[1]]:
                                v1Var = v[0][vKeys[0]][v1]
                                v2Var = v[1][vKeys[1]][v2]
                                    
                                index = (v1[0], v2[0])
                                
                                if v1Var is None or v2Var is None:
                                    zVars[index] = None
                                else:
                                    zVars[index] = lcFun(model, v1Var, v2Var, onlyConstrains = headConstrain)
                elif len(vKeys[1]) == 2: # Second variables names has two elements
                    if vKeys[0][0] in vKeys[1]:
                        ilpKey = vKeys[1]
                            
                        for v1 in v[1][vKeys[1]]:
                           
                            v1Var = v[0][vKeys[0]][(v1[0],)] 
                            v2Var = v[1][vKeys[1]][v1]
                                
                            if v1Var is None or v2Var is None:
                                zVars[v1] = None
                            else:
                                zVars[v1] = lcFun(model, v1Var, v2Var, onlyConstrains = headConstrain)
                    else:
                        pass
                else:
                    pass # Support only 2 elements now !
            elif len(vKeys[0]) == 2:  # First variables names has two elements
                if len(vKeys[1]) == 1:  # Second variables names has one elements
                    if vKeys[0][0] == vKeys[1][0]: # First name match
                        ilpKey = vKeys[0]
                        
                        for v1 in v[0][vKeys[0]]:
                           
                            v1Var = v[0][vKeys[0]][v1]
                            v2Var = v[1][vKeys[1]][(v1[0],)]
                             
                            if v1Var is None or v2Var is None:
                                zVars[v1] = None
                            else:  
                                zVars[v1] = lcFun(model, v1Var, v2Var, onlyConstrains = headConstrain)
                    elif vKeys[0][1] == vKeys[1][0]: # Second name match
                        ilpKey = vKeys[0]
                        
                        for v1 in v[0][vKeys[0]]:
                           
                            v1Var = v[0][vKeys[0]][v1]
                            v2Var = v[1][vKeys[1]][(v1[1],)]
                            
                            if v1Var is None or v2Var is None:
                                zVars[v1] = None
                            else:  
                                zVars[v1] = lcFun(model, v1Var, v2Var, onlyConstrains = headConstrain)
                    else:  # Different variables names
                        pass
                elif len(vKeys[1]) == 2:  # Second variables names has two elements
                    if (vKeys[0] == vKeys[1]): # The same sets of variables
                        ilpKey = vKeys[0]
                        
                        for v1 in v[0][vKeys[0]]:
                            
                            v1Var = v[0][vKeys[0]][v1]
                            v2Var = v[1][vKeys[1]][v1]
                                
                            if v1Var is None or v2Var is None:
                                zVars[v1] = None
                            else:
                                zVars[v1] = lcFun(model, v1Var, v2Var, onlyConstrains = headConstrain)
                    else:
                        pass
                else:
                    pass # Support only 2 elements now !
            else:
                pass # Support only 2 elements in key now !
        
        if len(v) > 2: # Support more then 2 variables  if all are the same
            equals = True
            for k in vKeys:
                if  len(k) > 2 or vKeys[0][0] != k[0]:
                    equals = False
                    break
                
                if equals:
                    ilpKey = vKeys[0]
                    
                    for v1 in v[0][vKeys[0]]:
                        _vars = []
                        for i, _ in enumerate(vKeys):
                            _vars.append(v[i][vKeys[0]][v1])
                         
                        if None in _vars:
                            zVars[v1] = None
                        else:
                            zVars[v1] = lcFun(model, *_vars, onlyConstrains = headConstrain)

        # Output
        if ilpKey:
            ilpV[ilpKey] = zVars
        else:
            if ifLog: myLogger.warning("%s Logical Constrain is not supported"%(self.lcName))
            ilpV[()] = zVars

        if model: model.update()

        return ilpV
    
    def createILPCount(self, model, myIlpBooleanProcessor, lcMethodName, v, resultVariableNames=None, headConstrain = False, cVariable = None, cOperation = None, cLimit = 1): 
        if ifLog: myLogger.debug("%s Logical Constrain invoked with variables: %s"%(lcMethodName, [[[x if x is None or isinstance(x, (int, numpy.float64, numpy.int32)) else x.VarName for _, x in t.items()] for _, t in v1.items()] for v1 in v]))
        
        if isinstance(self.e[0], tuple):
            cVariable = self.e[0]
                  
        if not resultVariableNames:
            resultVariableNames = ('x',)
            
        existsV = {}
        
        _v = v[0]
        _vKey = next(iter(_v))
        _v=next(iter(_v.values()))
        
        if cVariable is None:
            existsVars = []
            for currentToken in _v:
                existsVars.append(_v[currentToken])
        
            existsVarResult = myIlpBooleanProcessor.countVar(model, *existsVars, onlyConstrains = headConstrain, limitOp = cOperation, limit=cLimit)
                
            existsV[resultVariableNames] = {}
            existsV[resultVariableNames][(0,)] = existsVarResult
        elif isinstance(cVariable, tuple):
            if len(cVariable) == 1:
                if cVariable[0] in _vKey:
                    n1Index = _vKey.index(cVariable[0]) 
                    
                    existsVars = {}
                    for currentToken in _v:
                        key = (currentToken[:n1Index] + currentToken[n1Index+1:])
                        
                        if key in existsVars:
                            existsVars[key].append(_v[currentToken])
                        else:
                            existsVars[key] = [_v[currentToken]]
                        
                    existsV[resultVariableNames] = {}     
                    for k in existsVars:
                        existsVarResult = myIlpBooleanProcessor.countVar(model, *existsVars[k], onlyConstrains = headConstrain, limitOp = cOperation, limit=cLimit)
        
                        existsV[resultVariableNames][k] = existsVarResult
                else:
                    pass
            elif len(cVariable) == 2:
                pass
            elif len(cVariable) == 3:
                pass
            else:
                pass

        if  headConstrain:
            if ifLog: myLogger.debug("%s Logical Constrain is the head constrain - only ILP constrain created"%(lcMethodName))
        else:
            #if ifLog: myLogger.debug("Exists Logical Constrain result - ILP variables created: %s"%([x.VarName for x in existsV]))
            pass
                 
        model.update()
        
        return existsV

class andL(LogicalConstrain):
    def __init__(self, *e, p=100):
        LogicalConstrain.__init__(self, *e, p=p)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableNames=None, headConstrain = False): 
        return self.createILPConstrains('And', myIlpBooleanProcessor.andVar, model, v, resultVariableNames, headConstrain)        

class orL(LogicalConstrain):
    def __init__(self, *e, p=100):
        LogicalConstrain.__init__(self, *e, p=p)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableNames=None, headConstrain = False):
        return self.createILPConstrains('Or', myIlpBooleanProcessor.orVar, model, v, resultVariableNames, headConstrain)
    
class nandL(LogicalConstrain):
    def __init__(self, *e, p=100):
        LogicalConstrain.__init__(self, *e, p=p)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableNames=None, headConstrain = False): 
        return self.createILPConstrains('Nand', myIlpBooleanProcessor.nandVar, model, v, resultVariableNames, headConstrain)
        
class ifL(LogicalConstrain):
    def __init__(self, *e, p=100):
        LogicalConstrain.__init__(self, *e, p=p)
    
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableNames=None, headConstrain = False): 
        return self.createILPConstrains('If', myIlpBooleanProcessor.ifVar, model, v, resultVariableNames, headConstrain)
    
class norL(LogicalConstrain):
    def __init__(self, *e, p=100):
        LogicalConstrain.__init__(self, *e, p=p)
    
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableNames=None, headConstrain = False): 
        return self.createILPConstrains('Nor', myIlpBooleanProcessor.ifVar, model, v, resultVariableNames, headConstrain)

class xorL(LogicalConstrain):
    def __init__(self, *e, p=100):
        LogicalConstrain.__init__(self, *e, p=p)
    
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableNames=None, headConstrain = False): 
        return self.createILPConstrains('Xor', myIlpBooleanProcessor.ifVar, model, v, resultVariableNames, headConstrain)
    
class epqL(LogicalConstrain):
    def __init__(self, *e, p=100):
        LogicalConstrain.__init__(self, *e, p=p)
    
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableNames=None, headConstrain = False): 
        return self.createILPConstrains('Epq', myIlpBooleanProcessor.ifVar, model, v, resultVariableNames, headConstrain)
        
class eqL(LogicalConstrain):
    def __init__(self, *e):
        LogicalConstrain.__init__(self, *e, p=100)
      
class notL(LogicalConstrain):
    def __init__(self, *e, p=100):
        LogicalConstrain.__init__(self, *e, p=p)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableNames= None, headConstrain = False): 
        lcName = 'notL'
        if ifLog: myLogger.debug("%s Logical Constrain invoked with variables: %s"%(lcName, [[[x if not isinstance(x, torch.Tensor) else x.VarName for _, x in t.items()] for _, t in v1.items()] for v1 in v]))
              
        if not resultVariableNames:
            resultVariableNames = ('x',)
            
        notV = {}
                    
        if len(v) > 1:
            myLogger.error("Not Logical Constrain created with %i sets of variables which is more then one"%(len(v)))
            return notV
        
        _v = v[0]
        _vKey = next(iter(_v))
        _v=next(iter(_v.values()))
        
        notV[resultVariableNames] = {}

        for currentToken in _v:
            currentILPVar = _v[currentToken]
            
            notVar = myIlpBooleanProcessor.notVar(model, currentILPVar, onlyConstrains = headConstrain)
            notV[resultVariableNames][currentToken] = notVar
        
        if headConstrain:
            if ifLog: myLogger.debug("Not Logical Constrain is the head constrain - only ILP constrain created")
        else:
            if ifLog: myLogger.debug("Not Logical Constrain result - ILP variable created: %s"%(notVar.VarName))
            
        model.update()
        
        return notV

class exactL(LogicalConstrain):
    def __init__(self, *e, p=100):
        LogicalConstrain.__init__(self, *e, p=p)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableNames=None, headConstrain = False): 
        if isinstance(self.e[0], int):
            cLimit = self.e[0]
        else:
            cLimit = 1
            
        cVariable = self.e[1]
        lcMethodName = 'exactL'
        cOperation = '='
        
        return self.createILPCount(model, myIlpBooleanProcessor, lcMethodName, v, resultVariableNames, headConstrain, cVariable, cOperation, cLimit)

class existsL(LogicalConstrain):
    def __init__(self, *e, p=100):
        LogicalConstrain.__init__(self, *e, p=p)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableNames=None, headConstrain = False): 
        cLimit = 1

        cVariable = self.e[1]
        lcMethodName = 'existsL'
        cOperation = '>'
        
        return self.createILPCount(model, myIlpBooleanProcessor, lcMethodName, v, resultVariableNames, headConstrain, cVariable, cOperation, cLimit)

class atLeastL(LogicalConstrain):
    def __init__(self, *e, p=100):
        LogicalConstrain.__init__(self, *e, p=p)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableNames=None, headConstrain = False): 
        if isinstance(self.e[0], int):
            cLimit = self.e[0]
            
        cVariable = self.e[1]
        lcMethodName = 'atLeastL'
        cOperation = '>'
        
        return self.createILPCount(model, myIlpBooleanProcessor, lcMethodName, v, resultVariableNames, headConstrain, cVariable, cOperation, cLimit)
    
class atMostL(LogicalConstrain):
    def __init__(self, *e, p=100):
        LogicalConstrain.__init__(self, *e, p=p)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableNames=None, headConstrain = False): 
        if isinstance(self.e[0], int):
            cLimit = self.e[0]
            
        cVariable = self.e[1]
        lcMethodName = 'atMostL'
        cOperation = '<'
        
        return self.createILPCount(model, myIlpBooleanProcessor, lcMethodName, v, resultVariableNames, headConstrain, cVariable, cOperation, cLimit)