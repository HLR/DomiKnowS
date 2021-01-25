import logging

import numpy
import torch
from  collections import namedtuple
from regr.solver.ilpConfig import ilpConfig 
   
myLogger = logging.getLogger(ilpConfig['log_name'])
ifLog =  ilpConfig['ifLog']
        
V = namedtuple("V", ['name', 'v'], defaults= [None, None])

class LogicalConstrain:
    def __init__(self, *e, p=100):
        self.headLC = True # Indicate that it is head constrain and should be process individually
        
        if not e:
            myLogger.error("Logical Constrain initialized is empty")
            raise LogicalConstrain.LogicalConstrainError("Logical Constrain initialized is empty")
        
        self.e = e

        # -- Find the graph for this logical constrain - based on context defined in the concept used in constrain definition
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
        
        zVars = [] # output variables
        
        if len(v) == 2:
            for i in range(len(v[vKeys[0]])):
                
                if not (0 <= i < len(v[vKeys[0]])) or  not (0 <= i < len(v[vKeys[1]])):
                    myLogger.error("%s Logical Constrain created with %i has not equal number of elements in provided sets: %i - %i"%(lcName, len(v[vKeys[0]]), len(v[vKeys[1]])))
                    break
                    
                v1Var = v[vKeys[0]][i][0]
                v2Var = v[vKeys[1]][i][0]

                if v1Var is None or v2Var is None:
                    zVars.append([None])
                else:
                    zVars.append([lcFun(model, v1Var, v2Var, onlyConstrains = headConstrain)])
                        
            return zVars
    
    def createILPCount(self, model, myIlpBooleanProcessor, lcMethodName, v, resultVariableNames=None, headConstrain = False, cVariable = None, cOperation = None, cLimit = 1, logicMethodName = "COUNT"):         
        if cVariable not in v:
            return 
        
        existsV = []
        
        cVariableV = v[cVariable]
        for _v in cVariableV:
            existsVarResult = myIlpBooleanProcessor.countVar(model, *_v, onlyConstrains = headConstrain, limitOp = cOperation, limit=cLimit, logicMethodName = logicMethodName)
            
            existsV.append([existsVarResult])

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
        self.headLC = False

class notL(LogicalConstrain):
    def __init__(self, *e, p=100):
        LogicalConstrain.__init__(self, *e, p=p)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableNames= None, headConstrain = False): 
        lcName = 'notL'
              
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
        if isinstance(self.e[3], int):
            cLimit = self.e[3]
        else:
            cLimit = 1
            
        cVariable = self.e[2]
        lcMethodName = 'exactL'
        cOperation = '='
        
        return self.createILPCount(model, myIlpBooleanProcessor, lcMethodName, v, resultVariableNames, headConstrain, cVariable, cOperation, cLimit, logicMethodName = str(self))

class existsL(LogicalConstrain):
    def __init__(self, *e, p=100):
        LogicalConstrain.__init__(self, *e, p=p)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableNames=None, headConstrain = False): 
        cLimit = 1

        cVariable = self.e[2]
        lcMethodName = 'existsL'
        cOperation = '>'
        
        return self.createILPCount(model, myIlpBooleanProcessor, lcMethodName, v, resultVariableNames, headConstrain, cVariable, cOperation, cLimit, logicMethodName = str(self))

class atLeastL(LogicalConstrain):
    def __init__(self, *e, p=100):
        LogicalConstrain.__init__(self, *e, p=p)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableNames=None, headConstrain = False): 
        if isinstance(self.e[3], int):
            cLimit = self.e[3]
            
        cVariable = self.e[2]
        lcMethodName = 'atLeastL'
        cOperation = '>'
        
        return self.createILPCount(model, myIlpBooleanProcessor, lcMethodName, v, resultVariableNames, headConstrain, cVariable, cOperation, cLimit, logicMethodName = str(self))
    
class atMostL(LogicalConstrain):
    def __init__(self, *e, p=100):
        LogicalConstrain.__init__(self, *e, p=p)
        
    def __call__(self, model, myIlpBooleanProcessor, v, resultVariableNames=None, headConstrain = False): 
        if isinstance(self.e[3], int):
            cLimit = self.e[3]
            
        cVariable = self.e[2]
        lcMethodName = 'atMostL'
        cOperation = '<'
        
        return self.createILPCount(model, myIlpBooleanProcessor, lcMethodName, v, resultVariableNames, headConstrain, cVariable, cOperation, cLimit, logicMethodName = str(self))