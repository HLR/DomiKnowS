import logging

import numpy
import torch
from  collections import namedtuple
from regr.solver.ilpConfig import ilpConfig 
from future.builtins.misc import isinstance
   
myLogger = logging.getLogger(ilpConfig['log_name'])
ifLog =  ilpConfig['ifLog']
        
V = namedtuple("V", ['name', 'v'], defaults= [None, None])

class LogicalConstrain:
    def __init__(self, *e, p=100):
        self.headLC = True # Indicate that it is head constrain and should be process individually
        
        if not e:
            myLogger.error("Logical Constrain initialized is empty")
            raise LogicalConstrain.LogicalConstrainError("Logical Constrain initialized is empty")
          
        from regr.graph import Concept
        
        updatedE = []
        for i, eItem in enumerate(e):
            if isinstance(eItem, Concept):
                updatedE.append((eItem, 0))
            else:
                updatedE.append(eItem)
                
        self.e = updatedE

        # -- Find the graph for this logical constrain - based on context defined in the concept used in constrain definition
        self.graph = None
       
        conceptOrLc = None
        
        for i, eItem in enumerate(self.e):
            if isinstance(eItem, LogicalConstrain):
                conceptOrLc = eItem
                break
            elif isinstance(eItem, tuple):
                conceptOrLc = eItem[0]
                break
    
        if conceptOrLc is None:
            myLogger.error("Logical Constrain is incorrect")
            raise LogicalConstrain.LogicalConstrainError("Logical Constrain is incorrect")
            
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
        for e_item in self.e:
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
        
        for i in range(len(v[vKeys[0]])):
            var = []
            error = False
            varIsNone = False
            
            for k in vKeys:
                if not (0 <= i < len(v[k])):
                    myLogger.error("%s Logical Constrain created with %i has not equal number of elements in provided sets: %i - %i"%(lcName, len(v[vKeys[0]]), len(v[vKeys[1]])))
                    error = True
                    break
                
                cVar = v[k][i][0]
                
                if cVar is None:
                    zVars.append([None])
                    varIsNone = True
                    break
            
                var.append(v[k][i][0])
                
            if error: break
            
            if varIsNone: continue
                
            zVars.append([lcFun(model, *var, onlyConstrains = headConstrain)])
        
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
              
        notV = []
                    
        if len(v) > 1:
            myLogger.error("Not Logical Constrain created with %i sets of variables which is more then one"%(len(v)))
            return notV
    
        for currentILPVars in v.values():
            if not currentILPVars:
                notV.append([None])
                
            if len(currentILPVars) > 1:
                myLogger.error("Not Logical Constrain created with %i sets of variables which is more then one"%(len(currentILPVars)))
                return notV
                
            currentILPVar = currentILPVars[0]
            
            notVar = myIlpBooleanProcessor.notVar(model, *currentILPVar, onlyConstrains = headConstrain)
            notV.append([notVar])
        
        if headConstrain:
            if ifLog: myLogger.debug("Not Logical Constrain is the head constrain - only ILP constrain created")
        
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