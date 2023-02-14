from collections import namedtuple
from domiknows.solver.ilpConfig import ilpConfig 
from domiknows.graph import Concept

import logging
import torch
myLogger = logging.getLogger(ilpConfig['log_name'])
ifLog =  ilpConfig['ifLog']
        
V = namedtuple("V", ['name', 'v'], defaults= [None, None])

class LogicalConstrain:
    def __init__(self, *e, p=100, active = True, sampleEntries  = False, name = None):
        self.headLC = True # Indicate that it is head constrain and should be process individually
        self.active = active
        self.sampleEntries = sampleEntries
        
        if not e:
            myLogger.error("Logical Constraint initialized is empty")
            raise LogicalConstrain.LogicalConstrainError("Logical Constraint initialized is empty")
        
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
            if isinstance(eItem, Concept):
                updatedE.append((eItem, eItem.name, None, 1))
            else:
                updatedE.append(eItem)
                
        self.e = updatedE

        # -- Find the graph for this Logical Constraint - based on context defined in the concept used in constrain definition
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
            myLogger.error("Logical Constraint is incorrect")
            raise LogicalConstrain.LogicalConstrainError("Logical Constraint is incorrect")
            
        if isinstance(conceptOrLc, Concept):
            if self.__getContext(conceptOrLc):
                self.graph = self.__getContext(conceptOrLc)[-1]
        elif isinstance(conceptOrLc, str):
            self.graph = None
        else:
            self.graph = conceptOrLc.graph
                
        if self.graph == None:
            myLogger.error("Logical Constraint initialized is not associated with graph")
            raise LogicalConstrain.LogicalConstrainError("Logical Constraint initialized is not associated with graph")
                     
        # Create Logical Constraint id based on number of existing Logical Constraint in the graph
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
            myLogger.warning("%s Logical Constraint created with p equal %i sets it to 0"%(self.lcName,p))
        elif p > 100:
            self.p = 100
            myLogger.warning("%s Logical Constraint created with p equal %i sets it to 100"%(self.lcName,p))
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
       
    def getLcConcepts(self):
        lcConcepts = set()
        
        for _, eItem in enumerate(self.e):
            if isinstance(eItem, V):
                if eItem[1] and eItem[1][1]:
                    if isinstance(eItem[1][1], eqL):
                        if eItem[1][1].e[0]:
                            lcConcepts.add(eItem[1][1].e[0][0].name)
            elif isinstance(eItem, Concept):
                lcConcepts.add(eItem.name)
            elif isinstance(eItem, LogicalConstrain):
                lcConcepts.update(eItem.getLcConcepts())
            elif isinstance(eItem, tuple) and (len(eItem) == 4):
                lcConcepts.add(eItem[0].name)
                
        return lcConcepts
        
    # Get string representation of  Logical Constraintt
    def strEs(self):
        strsE = []
        for _, eItem in enumerate(self.e):
            if isinstance(eItem, V):
                new_V = []
                if eItem[0] is not None:
                    new_V.append(eItem[0])
                    #if eItem[1] is not None: new_V.append(',')
                if eItem[1]:
                    if isinstance(eItem[1], eqL):
                        strsE.append("eql")
                    else:
                        from domiknows.graph import Relation
                        new_v = [v if isinstance(v, (str, tuple, LogicalConstrain, Relation)) else v.name for v in eItem[1]]
                        new_V.append("path={}".format(tuple(new_v)))
                strsE.append("{}".format(tuple(new_V)))
            elif isinstance(eItem, Concept):
                strsE.append(eItem.name)
            elif isinstance(eItem, LogicalConstrain):
                strsE.append(eItem.__repr__())
            elif isinstance(eItem, tuple) and (len(eItem) == 4):
                if eItem[2]:
                    strsE.append(eItem[0].name + '\\' + eItem[1])
                else:
                    strsE.append(eItem[0].name)

        newStrsE = "["
        for s in strsE:
            newStrsE+= s
            newStrsE+= ", "
            
        newStrsE = newStrsE[:-2]
        newStrsE=newStrsE.replace(",)", ")")
        newStrsE=newStrsE.replace("'", "")
        newStrsE=newStrsE.replace('"', '')
        newStrsE=newStrsE.replace("'", "")
        newStrsE=newStrsE.replace(", (", "(")
        newStrsE +="]"
            
        return newStrsE
    
    #------------
    
    def createSingleVarILPConstrains(self, lcName, lcFun, model, v, headConstrain):  
        singleV = []
                    
        if len(v) != 1:
            myLogger.error("%s Logical Constraint created with %i sets of variables which is not  one"%(lcName,len(v)))
            return singleV
    
        v1 = list(v.values())[0]
        for currentILPVars in v1:
            if not currentILPVars:
                singleV.append([None])
                continue
                
            cSingleVar = []
            for cv in currentILPVars:
                singleVar = lcFun(model, cv, onlyConstrains = headConstrain)
                cSingleVar.append(singleVar)
                
            singleV.append(cSingleVar)
        
        if headConstrain:
            if ifLog: myLogger.debug("%s Logical Constraint is the head constrain - only ILP constrain created"%(lcName))
        
        if model is not None:
            model.update()
        
        return singleV
    
    # Collects setups of ILP variables for logical methods calls for the created Logical Constraint - recursive method
    def _collectILPVariableSetups(self, lcVariableName, lcVariableNames, v, lcVars = []): 
        
        # Get set of ILP variables lists for the current variable name
        cLcVariables = v[lcVariableName]
        
        # List of lists containing sets of ILP variables for particular position 
        newLcVars = []
        
        # --- Update the lcVars setup with ILP variables from this iteration
        
        if not lcVars: # If ILP variables setup is not initialized yet - this is the first iteration of the _collectILPVariableSetups method
            if cLcVariables is None:
                newV = [[None]]
                newLcVars.append(newV)
            else:
                for cV in cLcVariables:
                    newV = []
                    for cvElement in cV:
                        if cvElement is None:
                            pass
                        newElement = [cvElement]
                        newV.append(newElement)
                    newLcVars.append(newV)
        elif len(cLcVariables) == 1: # Single variable
            for indexLcV, lcV in enumerate(lcVars):
                newV = []
                for lcVelement in lcV:
                    newElemenet = lcVelement.copy()
                    if cLcVariables is None:
                        newElemenet.append(None)  
                    else:
                        if cLcVariables[0]:
                            newElemenet.append(cLcVariables[0][0])
                        else:
                            newElemenet.append(None)
                    newV.append(newElemenet)
                                    
                newLcVars.append(newV)                
        else: # Many ILP variables in the current set
            for indexLcV, lcV in enumerate(lcVars):
                newV = []
                for indexElement, lcVelement in enumerate(lcV):
                    if cLcVariables is None:
                        newLcVelement = lcVelement.copy()
                        newLcVelement.append(None)
                        
                        newV.append(newLcVelement)
                    elif len(lcV) == len(cLcVariables[indexLcV]):
                        cV = cLcVariables[indexLcV][indexElement]
                        newLcVelement = lcVelement.copy()
                        newLcVelement.append(cV)
                            
                        newV.append(newLcVelement)
                    else:
                        for cV in cLcVariables[indexLcV]:
                            newLcVelement = lcVelement.copy()
                            newLcVelement.append(cV)
                            
                            newV.append(newLcVelement)
                                
                newLcVars.append(newV)                
                            
        if lcVariableNames:
            # Recursive call - lcVars contains currently collected ILP variables setups
            return self._collectILPVariableSetups(lcVariableNames[0], lcVariableNames[1:], v, lcVars=newLcVars)
        else:
            # Return collected setups
            return newLcVars

    # Method building ILP constraints
    def createILPConstrains(self, lcName, lcFun, model, v, headConstrain):
        if len(v) < 2:
            myLogger.error("%s Logical Constraint created with %i sets of variables which is less then two"%(lcName, len(v)))
            return None
        
        # Input variable names
        try:
            lcVariableNames = [e for e in iter(v)]
        except StopIteration:
            pass
                
        lcVariableName0 = lcVariableNames[0] # First LC variable
        lcVariableSet0 = v[lcVariableName0]

        rVars = [] # Output variables

        # Check consistency of provided sets of ILP variables
        for cLcVariableName in lcVariableNames:
            cLcVariableSet = v[cLcVariableName]

            if len(cLcVariableSet) != len(lcVariableSet0):
                myLogger.error("%s Logical Constraint has no equal number of elements in provided sets: %s has %i elements and %s as %i elements"
                               %(lcName, lcVariableName0, len(v[lcVariableName0]), cLcVariableName, len(cLcVariableSet)))
                
                return rVars
            
        # Collect variables setups for ILP constraints
        sVar = self._collectILPVariableSetups(lcVariableName0, lcVariableNames[1:], v)
        
        # Apply collected setups and create ILP constraint
        for z in sVar:
            tVars = [] # Collect ILP constraints results
            for t in z:
                tVars.append(lcFun(model, *t, onlyConstrains = headConstrain))
                
            rVars.append(tVars)
        
        # Return results from created ILP constraints - 
        # None if headConstrain is True or no ILP constraint created, ILP variable representing the value of ILP constraint, loss calculated
        return rVars
    
    def createILPCount(self, model, myIlpBooleanProcessor, v, headConstrain, cOperation, cLimit, integrate, logicMethodName = "COUNT"):         
        try:
            lcVariableNames = [e for e in iter(v)]
        except StopIteration:
            pass
        
        if cLimit== None:
            cLimit
        lcVariableName0 = lcVariableNames[0] # First variable
        lcVariableSet0 =  v[lcVariableName0]

        zVars = [] # Output ILP variables
        
        for i, _ in enumerate(lcVariableSet0):
            varsSetup = []

            var = []
            for currentV in iter(v):
                var.extend(v[currentV][i])
                
            if len(var) == 0:
                if not (headConstrain or integrate):
                    varsSetup.append([None])
                    
                continue
            
            if headConstrain or integrate:
                varsSetup.extend(var)
            else:
                varsSetup.append(var)
             
            # -- Use ILP variable setup to create constrains   
            if headConstrain or integrate:
                zVars.append([myIlpBooleanProcessor.countVar(model, *varsSetup, onlyConstrains = headConstrain, limitOp = cOperation, limit=cLimit, 
                                                             logicMethodName = logicMethodName)])
            else:
                for current_var in varsSetup:
                    zVars.append([myIlpBooleanProcessor.countVar(model, *current_var, onlyConstrains = headConstrain, limitOp = cOperation, limit=cLimit, 
                                                             logicMethodName = logicMethodName)])
           
        if model is not None:
            model.update()
            
        return zVars
    
    def createILPAccumulatedCount(self, model, myIlpBooleanProcessor, v, headConstrain, cOperation, cLimit, integrate, logicMethodName = "COUNT"):         
        try:
            lcVariableNames = [e for e in iter(v)]
        except StopIteration:
            pass
        
        if cLimit== None:
            cLimit
        lcVariableName0 = lcVariableNames[0] # First variable
        lcVariableSet0 =  v[lcVariableName0]

        zVars = [] # Output ILP variables
        
        # Accumulate all variables
        varsSetup = []
        for i, _ in enumerate(lcVariableSet0):
            
            var = []
            for currentV in iter(v):
                var.extend(v[currentV][i])
                
            if len(var) == 0:
                if not (headConstrain or integrate):
                    varsSetup.append([None])
                    
                continue
            
            if headConstrain or integrate:
                varsSetup.extend(var)
            else:
                varsSetup.append(var)
             
        # -- Use ILP variable setup to create constrains   
        if headConstrain or integrate:
            zVars.append([myIlpBooleanProcessor.countVar(model, *varsSetup, onlyConstrains = headConstrain, limitOp = cOperation, limit=cLimit, 
                                                         logicMethodName = logicMethodName)])
        else:
            for current_var in varsSetup:
                zVars.append([myIlpBooleanProcessor.countVar(model, *current_var, onlyConstrains = headConstrain, limitOp = cOperation, limit=cLimit, 
                                                         logicMethodName = logicMethodName)])
       
        if model is not None:
            model.update()
            
        return zVars
        

def use_grad(grad):
    if not grad:
        torch.no_grad()
        
# ----------------- Logical Single Variable

class notL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False): 
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad):
            return self.createSingleVarILPConstrains("Not", myIlpBooleanProcessor.notVar, model, v, headConstrain)

# ----------------- Logical

class andL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False): 
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad):
            return self.createILPConstrains('And', myIlpBooleanProcessor.andVar, model, v, headConstrain)        

class orL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False):
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad):
            return self.createILPConstrains('Or', myIlpBooleanProcessor.orVar, model, v, headConstrain)
    
class nandL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False): 
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad):
            return self.createILPConstrains('Nand', myIlpBooleanProcessor.nandVar, model, v, headConstrain)
        
class ifL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
    
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False): 
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad): #use_grad(myIlpBooleanProcessor.grad):
            return self.createILPConstrains('If', myIlpBooleanProcessor.ifVar, model, v, headConstrain)
    
class norL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
    
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False): 
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad):
            return self.createILPConstrains('Nor', myIlpBooleanProcessor.ifVar, model, v, headConstrain)

class xorL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
    
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False): 
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad):
            return self.createILPConstrains('Xor', myIlpBooleanProcessor.ifVar, model, v, headConstrain)
    
class epqL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
    
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False):
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad): 
            return self.createILPConstrains('Epq', myIlpBooleanProcessor.ifVar, model, v, headConstrain)
     
# ----------------- Auxiliary
     
class eqL(LogicalConstrain):
    def __init__(self, *e, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=100)
        self.headLC = False
    
class fixedL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False):
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad): 
            return self.createSingleVarILPConstrains("Fixed", myIlpBooleanProcessor.fixedVar, model, v, headConstrain)
    
# ----------------- Counting

class exactL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1

        cOperation = '=='
        
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad):
            return self.createILPCount(model, myIlpBooleanProcessor, v, headConstrain, cOperation, cLimit, integrate, logicMethodName = str(self))

class existsL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False): 
        cLimit = 1

        cOperation = '>='
        
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad):
            return self.createILPCount(model, myIlpBooleanProcessor, v, headConstrain, cOperation, cLimit, integrate, logicMethodName = str(self))

class atLeastL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1
            
        cOperation = '>='
        
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad):
            return self.createILPCount(model, myIlpBooleanProcessor, v, headConstrain, cOperation, cLimit, integrate, logicMethodName = str(self))
    
class atMostL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1
            
        cOperation = '<='
        
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad):
            return self.createILPCount(model, myIlpBooleanProcessor, v, headConstrain, cOperation, cLimit, integrate, logicMethodName = str(self))
        
        
# ----------------- Accumulated Counting

class exactAL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1

        cOperation = '=='
        
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad):
            return self.createILPAccumulatedCount(model, myIlpBooleanProcessor, v, headConstrain, cOperation, cLimit, integrate, logicMethodName = str(self))

class existsAL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False): 
        cLimit = 1

        cOperation = '>='
        
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad):
            return self.createILPAccumulatedCount(model, myIlpBooleanProcessor, v, headConstrain, cOperation, cLimit, integrate, logicMethodName = str(self))

class atLeastAL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1
            
        cOperation = '>='
        
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad):
            return self.createILPAccumulatedCount(model, myIlpBooleanProcessor, v, headConstrain, cOperation, cLimit, integrate, logicMethodName = str(self))
    
class atMostAL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myIlpBooleanProcessor, v, headConstrain = False, integrate = False): 
        if isinstance(self.e[-1], int):
            cLimit = self.e[-1]
        else:
            cLimit = 1
            
        cOperation = '<='
        
        with torch.set_grad_enabled(myIlpBooleanProcessor.grad):
            return self.createILPAccumulatedCount(model, myIlpBooleanProcessor, v, headConstrain, cOperation, cLimit, integrate, logicMethodName = str(self))