from collections import namedtuple
from regr.solver.ilpConfig import ilpConfig 
from regr.graph import Concept

import logging
myLogger = logging.getLogger(ilpConfig['log_name'])
ifLog =  ilpConfig['ifLog']
        
V = namedtuple("V", ['name', 'v'], defaults= [None, None])

class LogicalConstrain:
    def __init__(self, *e, p=100, active = True, name = None):
        self.headLC = True # Indicate that it is head constrain and should be process individually
        self.active = active
        
        if not e:
            myLogger.error("Logical Constrain initialized is empty")
            raise LogicalConstrain.LogicalConstrainError("Logical Constrain initialized is empty")
        
        updatedE = []
        for _, eItem in enumerate(e):
            if isinstance(eItem, (LogicalConstrain, Concept)):
                updatedE.append(eItem)
            elif callable(eItem):
                newEItem = eItem.__call__()
                updatedE.extend(newEItem)
            elif isinstance(eItem, list):
                updatedE.extend(eItem)
            else:
                updatedE.append(eItem)
                
        self.e = updatedE
        
        updatedE = []
        for _, eItem in enumerate(self.e):
            if isinstance(eItem, V):
                updatedE.append(eItem)
            elif isinstance(eItem, Concept):
                updatedE.append((eItem, 1, 0))
            elif isinstance(eItem, tuple) and (len(eItem) <= 2):
                updatedE.append((eItem[0], eItem[1], eItem[1]))
            else:
                updatedE.append(eItem)
                
        self.e = updatedE

        # -- Find the graph for this logical constrain - based on context defined in the concept used in constrain definition
        self.graph = None
       
        conceptOrLc = None
        
        for _, eItem in enumerate(self.e):
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
        elif isinstance(conceptOrLc, str):
            self.graph = None
        else:
            self.graph = conceptOrLc.graph
                
        if self.graph == None:
            myLogger.error("Logical Constrain initialized is not associated with graph")
            raise LogicalConstrain.LogicalConstrainError("Logical Constrain initialized is not associated with graph")
                     
        # Create logical constrain id based on number of existing logical constrain in the graph
        self.lcName = "LC%i"%(len(self.graph.logicalConstrains))
        
        if name is not None:
            self.name = name
        else:
            self.name = self.lcName
            
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
       
    def strEs(self):
        strsE = []
        for _, eItem in enumerate(self.e):
            if isinstance(eItem, V):
                new_V = []
                if eItem[0] is not None:
                    new_V.append(eItem[0])
                    if eItem[1] is not None: new_V.append(',')
                if eItem[1]:
                    if isinstance(eItem[1], eqL):
                        strsE.append("eql")
                    else:
                        new_v = [v if isinstance(v, (str, tuple, LogicalConstrain)) else v.name for v in eItem[1]]
                        new_V.append("path = {}".format(tuple(new_v)))
                strsE.append("{}".format(tuple(new_V)))
            elif isinstance(eItem, Concept):
                strsE.append(eItem.name)
            elif isinstance(eItem, LogicalConstrain):
                strsE.append(eItem)
            elif isinstance(eItem, tuple) and (len(eItem) == 3):
                strsE.append(eItem[0].name)
        
        return strsE
    
    #------------
    
    def createSingleVarILPConstrains(self, lcName, lcFun, model, v, headConstrain = False):  
        singleV = []
                    
        if len(v) != 1:
            myLogger.error("%s Logical Constrain created with %i sets of variables which is not  one"%(lcName,len(v)))
            return singleV
    
        v1 = list(v.values())[0]
        for currentILPVars in v1:
            if not currentILPVars:
                singleV.append([None])
                
            cSingleVar = []
            for cv in currentILPVars:
                singleVar = lcFun(model, cv, onlyConstrains = headConstrain)
                cSingleVar.append(singleVar)
                
            singleV.append(cSingleVar)
        
        if headConstrain:
            if ifLog: myLogger.debug("%s Logical Constrain is the head constrain - only ILP constrain created"%(lcName))
        
        if model is not None:
            model.update()
        
        return singleV
    
    # Collects setups of ILP variables for logical methods calls for the created Logical constrain - recursive method
    # sVars - returned list of ILP variables setups
    def _collectILPVariableSetups(self, lcVariableName, lcVariableNames, index, v, lcVars, sVars):
        cLcVariableSet = v[lcVariableName][index] # Current logical variable (lcVariableName) sets of ILP variables
        
        if len(cLcVariableSet) == 0: # No ILP variables for a given logical variable
            lcVars.append(None)
            
            if lcVariableNames:
                self._collectILPVariableSetups(lcVariableNames[0], lcVariableNames[1:], index, v, lcVars, sVars)
            
            return
                                       
        for ilvV in cLcVariableSet:
            newLcVars = lcVars[:] # clone
            
            if ilvV is None:
                newLcVars.append(None) # alternative code: sVars.append([None]), continue
            else:
                newLcVars.append(ilvV)

            if lcVariableNames: # There still remaining lc variables
                self._collectILPVariableSetups(lcVariableNames[0], lcVariableNames[1:], index, v, newLcVars, sVars)
            else:
                sVars.append(newLcVars) # This is the last logical variable  - setup of ILP variables finish - collect it

    # Method building ILP constraints
    def createILPConstrains(self, lcName, lcFun, model, v, headConstrain = False):
        if len(v) < 2:
            myLogger.error("%s Logical Constrain created with %i sets of variables which is less then two"%(lcName, len(v)))
            return None
        
        # input variable names
        try:
            lcVariableNames = [e for e in iter(v)]
        except StopIteration:
            pass
                
        lcVariableName0 = lcVariableNames[0] # First LC variable
        lcVariableSet0 = v[lcVariableName0]

        rVars = [] # output variables

        # Check consistency of provided sets of ILP variables
        for cLcVariableName in lcVariableNames:
            cLcVariableSet = v[cLcVariableName]

            if len(cLcVariableSet) != len(lcVariableSet0):
                myLogger.error("%s Logical Constrain has no equal number of elements in provided sets: %s has %i elements and %s as %i elements"%(lcName, lcVariableName0, len(v[lcVariableName0]), cLcVariableName, len(cLcVariableSet)))
                
                return rVars
            
        # Loop through input ILP variables sets in the list of the first input LC variable
        zVars = []
        for i in range(len(lcVariableSet0)):
            sVars = []
            self._collectILPVariableSetups(lcVariableName0, lcVariableNames[1:], i, v, [], sVars)
            zVars.append(sVars)
        
        for z in zVars:
            tVars = []
            for t in z:
                tVars.append(lcFun(model, *t, onlyConstrains = headConstrain))
                
            rVars.append(tVars)
        
        return rVars
    
    def createILPCount(self, model, myIlpBooleanProcessor, lcMethodName, v, headConstrain = False, cOperation = None, cLimit = 1, logicMethodName = "COUNT"):         
        try:
            lcVariableNames = [e for e in iter(v)]
        except StopIteration:
            pass
        
        lcVariableName0 = lcVariableNames[0] # First variable
        lcVariableSet0 =  v[lcVariableName0]

        zVars = [] # Output ILP variables
        
        varsSetup = []
        for i, _ in enumerate(lcVariableSet0):
            
            var = []
            for currentV in iter(v):
                var.extend(v[currentV][i])
                
            if len(var) == 0:
                varsSetup.append([None])
                continue
            
            varsSetup.append(var)
             
        # -- use ILP variable setup to create constrains            
        for current_var in varsSetup:
            zVars.append([myIlpBooleanProcessor.countVar(model, *current_var, onlyConstrains = headConstrain, limitOp = cOperation, limit=cLimit, logicMethodName = logicMethodName)])
       
        if model is not None:
            model.update()
            
        return zVars
        
    def createILPCountI(self, model, myIlpBooleanProcessor, lcMethodName, v, headConstrain = False, cOperation = None, cLimit = 1, logicMethodName = "COUNT"):         
        try:
            lcVariableNames = [e for e in iter(v)]
        except StopIteration:
            pass
        
        lcVariableName0 = lcVariableNames[0] # First variable
        lcVariableSet0 =  v[lcVariableName0]

        zVars = [] # Output ILP variables
        
        varsSetup = []
        for i, _ in enumerate(lcVariableSet0):
            
            var = []
            for currentV in iter(v):
                var.extend(v[currentV][i])
                
            if len(var) == 0:
                continue
                         
            varsSetup.extend(var)
        
        # -- use ILP variable setup to create constrains            
        zVars.append([myIlpBooleanProcessor.countVar(model, *varsSetup, onlyConstrains = headConstrain, limitOp = cOperation, limit=cLimit, logicMethodName = logicMethodName)])
       
        if model is not None:
            model.update()
            
        return zVars
    
class andL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createILPConstrains('And', myIlpBooleanProcessor.andVar, model, v, headConstrain)        

class orL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False):
        return self.createILPConstrains('Or', myIlpBooleanProcessor.orVar, model, v, headConstrain)
    
class nandL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createILPConstrains('Nand', myIlpBooleanProcessor.nandVar, model, v, headConstrain)
        
class ifL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
    
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createILPConstrains('If', myIlpBooleanProcessor.ifVar, model, v, headConstrain)
    
class norL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
    
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createILPConstrains('Nor', myIlpBooleanProcessor.ifVar, model, v, headConstrain)

class xorL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
    
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createILPConstrains('Xor', myIlpBooleanProcessor.ifVar, model, v, headConstrain)
    
class epqL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
    
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createILPConstrains('Epq', myIlpBooleanProcessor.ifVar, model, v, headConstrain)
       
class eqL(LogicalConstrain):
    def __init__(self, *e, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=100)
        self.headLC = False

class notL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createSingleVarILPConstrains("Not", myIlpBooleanProcessor.notVar, model, v, headConstrain)
    
class FixedL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createSingleVarILPConstrains("Fixed", myIlpBooleanProcessor.FixedVar, model, v, headConstrain)
    
# ----------------- Class Count

class exactL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1
            
        lcMethodName = 'exactL'
        cOperation = '=='
        
        return self.createILPCount(model, myIlpBooleanProcessor, lcMethodName, v, headConstrain, cOperation, cLimit, logicMethodName = str(self))

class existsL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        cLimit = 1

        lcMethodName = 'existsL'
        cOperation = '>='
        
        return self.createILPCount(model, myIlpBooleanProcessor, lcMethodName, v, headConstrain, cOperation, cLimit, logicMethodName = str(self))

class atLeastL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1
            
        lcMethodName = 'atLeastL'
        cOperation = '>='
        
        return self.createILPCount(model, myIlpBooleanProcessor, lcMethodName, v, headConstrain, cOperation, cLimit, logicMethodName = str(self))
    
class atMostL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1
            
        lcMethodName = 'atMostL'
        cOperation = '<='
        
        return self.createILPCount(model, myIlpBooleanProcessor, lcMethodName, v, headConstrain, cOperation, cLimit, logicMethodName = str(self))

# ----------------- Instance count

class exactI(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1
            
        lcMethodName = 'exactL'
        cOperation = '=='
        
        return self.createILPCountI(model, myIlpBooleanProcessor, lcMethodName, v, headConstrain, cOperation, cLimit, logicMethodName = str(self))

class existsI(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        cLimit = 1

        lcMethodName = 'existsL'
        cOperation = '>='
        
        return self.createILPCountI(model, myIlpBooleanProcessor, lcMethodName, v, headConstrain, cOperation, cLimit, logicMethodName = str(self))

class atLeastI(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1
            
        lcMethodName = 'atLeastL'
        cOperation = '>='
        
        return self.createILPCountI(model, myIlpBooleanProcessor, lcMethodName, v, headConstrain, cOperation, cLimit, logicMethodName = str(self))
    
class atMostI(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1
            
        lcMethodName = 'atMostL'
        cOperation = '<='
        
        return self.createILPCountI(model, myIlpBooleanProcessor, lcMethodName, v, headConstrain, cOperation, cLimit, logicMethodName = str(self))