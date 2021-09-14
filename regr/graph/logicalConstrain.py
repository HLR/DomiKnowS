from  collections import namedtuple
from regr.solver.ilpConfig import ilpConfig 
   
import logging
myLogger = logging.getLogger(ilpConfig['log_name'])
ifLog =  ilpConfig['ifLog']
        
V = namedtuple("V", ['name', 'v'], defaults= [None, None])

class LogicalConstrain:
    def __init__(self, *e, p=100, active = True):
        self.headLC = True # Indicate that it is head constrain and should be process individually
        self.active = active
        
        if not e:
            myLogger.error("Logical Constrain initialized is empty")
            raise LogicalConstrain.LogicalConstrainError("Logical Constrain initialized is empty")
          
        updatedE = []
        for _, eItem in enumerate(e):
            if isinstance(eItem, list):
                updatedE.extend(eItem)
            else:
                updatedE.append(eItem)
                
        self.e = updatedE
        
        from regr.graph import Concept
        
        updatedE = []
        for _, eItem in enumerate(self.e):
            if isinstance(eItem, V):
                updatedE.append(eItem)
            elif isinstance(eItem, Concept):
                updatedE.append((eItem, 1, 0))
            elif isinstance(eItem, tuple):
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
       
    # Collects setups of ILP variables for logical methods calls for the created Logical constrain - recursive method
    # sVars - returned list of ILP variables setups
    def _collectILPVariableSetups(self, lcVariableName, lcVariableNames, index, v, lcVars, sVars):
        cLcVariableSet = v[lcVariableName][index] # Current logical variable (lcVariableName) sets of ILP variables
        
        if len(cLcVariableSet) == 0: # No ILP variables for a given logical variable
            lcVars.append(None)
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
            for t in z:
                tVars = []
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

        zVars = [] # Output variables
        
        # Loop through input ILP variables sets in the list of the first input LC variable
        for i, _ in enumerate(lcVariableSet0):
            
            var = []
            for currentV in iter(v):
                var.extend(v[currentV][i])
                
            if len(var) == 0:
                zVars.append([None])
                continue
                         
            zVars.append([myIlpBooleanProcessor.countVar(model, *var, onlyConstrains = headConstrain, limitOp = cOperation, limit=cLimit, logicMethodName = logicMethodName)])
       
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

        zVars = [] # Output variables
        
        # Loop through input ILP variables sets in the list of the first input LC variable
        _var = []
        
        for i, _ in enumerate(lcVariableSet0):
            
            var = []
            for currentV in iter(v):
                var.extend(v[currentV][i])
                
            if len(var) == 0:
                continue
                         
            _var.extend(var)
            
        zVars.append([myIlpBooleanProcessor.countVar(model, *_var, onlyConstrains = headConstrain, limitOp = cOperation, limit=cLimit, logicMethodName = logicMethodName)])
       
        if model is not None:
            model.update()
            
        return zVars
    
class andL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True):
        LogicalConstrain.__init__(self, *e, p=p, active=active)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createILPConstrains('And', myIlpBooleanProcessor.andVar, model, v, headConstrain)        

class orL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True):
        LogicalConstrain.__init__(self, *e, p=p, active=active)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False):
        return self.createILPConstrains('Or', myIlpBooleanProcessor.orVar, model, v, headConstrain)
    
class nandL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True):
        LogicalConstrain.__init__(self, *e, p=p, active=active)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createILPConstrains('Nand', myIlpBooleanProcessor.nandVar, model, v, headConstrain)
        
class ifL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True):
        LogicalConstrain.__init__(self, *e, p=p, active=active)
    
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createILPConstrains('If', myIlpBooleanProcessor.ifVar, model, v, headConstrain)
    
class norL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True):
        LogicalConstrain.__init__(self, *e, p=p, active=active)
    
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createILPConstrains('Nor', myIlpBooleanProcessor.ifVar, model, v, headConstrain)

class xorL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True):
        LogicalConstrain.__init__(self, *e, p=p, active=active)
    
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createILPConstrains('Xor', myIlpBooleanProcessor.ifVar, model, v, headConstrain)
    
class epqL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True):
        LogicalConstrain.__init__(self, *e, p=p, active=active)
    
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        return self.createILPConstrains('Epq', myIlpBooleanProcessor.ifVar, model, v, headConstrain)
       
class eqL(LogicalConstrain):
    def __init__(self, *e, active = True):
        LogicalConstrain.__init__(self, *e, p=100)
        self.headLC = False

class notL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True):
        LogicalConstrain.__init__(self, *e, p=p, active=active)
        
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

# ----------------- Class Count

class exactL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True):
        LogicalConstrain.__init__(self, *e, p=p, active=active)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1
            
        lcMethodName = 'exactL'
        cOperation = '=='
        
        return self.createILPCount(model, myIlpBooleanProcessor, lcMethodName, v, headConstrain, cOperation, cLimit, logicMethodName = str(self))

class existsL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True):
        LogicalConstrain.__init__(self, *e, p=p, active=active)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        cLimit = 1

        lcMethodName = 'existsL'
        cOperation = '>='
        
        return self.createILPCount(model, myIlpBooleanProcessor, lcMethodName, v, headConstrain, cOperation, cLimit, logicMethodName = str(self))

class atLeastL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True):
        LogicalConstrain.__init__(self, *e, p=p, active=active)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1
            
        lcMethodName = 'atLeastL'
        cOperation = '>='
        
        return self.createILPCount(model, myIlpBooleanProcessor, lcMethodName, v, headConstrain, cOperation, cLimit, logicMethodName = str(self))
    
class atMostL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True):
        LogicalConstrain.__init__(self, *e, p=p, active=active)
        
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
    def __init__(self, *e, p=100, active = True):
        LogicalConstrain.__init__(self, *e, p=p, active=active)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1
            
        lcMethodName = 'exactL'
        cOperation = '=='
        
        return self.createILPCountI(model, myIlpBooleanProcessor, lcMethodName, v, headConstrain, cOperation, cLimit, logicMethodName = str(self))

class existsI(LogicalConstrain):
    def __init__(self, *e, p=100, active = True):
        LogicalConstrain.__init__(self, *e, p=p, active=active)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        cLimit = 1

        lcMethodName = 'existsL'
        cOperation = '>='
        
        return self.createILPCountI(model, myIlpBooleanProcessor, lcMethodName, v, headConstrain, cOperation, cLimit, logicMethodName = str(self))

class atLeastI(LogicalConstrain):
    def __init__(self, *e, p=100, active = True):
        LogicalConstrain.__init__(self, *e, p=p, active=active)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1
            
        lcMethodName = 'atLeastL'
        cOperation = '>='
        
        return self.createILPCountI(model, myIlpBooleanProcessor, lcMethodName, v, headConstrain, cOperation, cLimit, logicMethodName = str(self))
    
class atMostI(LogicalConstrain):
    def __init__(self, *e, p=100, active = True):
        LogicalConstrain.__init__(self, *e, p=p, active=active)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1
            
        lcMethodName = 'atMostL'
        cOperation = '<='
        
        return self.createILPCountI(model, myIlpBooleanProcessor, lcMethodName, v, headConstrain, cOperation, cLimit, logicMethodName = str(self))