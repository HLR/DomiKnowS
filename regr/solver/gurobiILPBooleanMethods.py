import logging
from itertools import permutations

from regr.solver.ilpBooleanMethods import ilpBooleanProcessor 
from regr.solver.ilpConfig import ilpConfig 

from gurobipy import Var, GRB, LinExpr

class gurobiILPBooleanProcessor(ilpBooleanProcessor):
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
                
        self.myLogger = logging.getLogger(ilpConfig['log_name'])
        self.ifLog =  ilpConfig['ifLog']

        self.constrainCaches = {}

    def resetCaches(self):
        self.constrainCaches = {}

    def __addToConstrainCaches(self, lmName, onlyConstrains, var, cachedValue):
        if isinstance(var, list):
            var = tuple(var)
            
            try:
                h = hash(var)
            except Exception as e:
                return 
            
        if lmName in self.constrainCaches:
            if onlyConstrains in self.constrainCaches[lmName]:
                try:    
                    self.constrainCaches[lmName][onlyConstrains][var] = cachedValue
                except Exception as ex:
                    pass
            else:
                self.constrainCaches[lmName][onlyConstrains] = {}
                self.constrainCaches[lmName][onlyConstrains][var] = cachedValue
        else:
            self.constrainCaches[lmName] = {}
            self.constrainCaches[lmName][onlyConstrains] = {}

            self.constrainCaches[lmName][onlyConstrains][var] = cachedValue
            
    def __isInConstrainCaches(self, lmName, onlyConstrains, var):
        if isinstance(var, list):
            var = tuple(var)
            
            try:
                h = hash(var)
            except Exception as e:
                return (False, None)
        try:    
            if lmName in self.constrainCaches:
                if onlyConstrains in self.constrainCaches[lmName]:
                    for currentVarPermutation in permutations(var):
                        if currentVarPermutation in self.constrainCaches[lmName][onlyConstrains]:
                            #if self.ifLog: self.myLogger.debug("%s already created constraint for this variables %s - does nothing"%(lmName, [x.VarName for x in var]))
                            return (True, self.constrainCaches[lmName][onlyConstrains][currentVarPermutation])
        except Exception as ex:
            pass
                        
        return (False, None)
                
    def __varIsNumber(self, var):
        return not isinstance(var, Var)
    
    def __fixVar(self, var):
        if var is None:
            return 0
        else:
            return  var
        
    def __fixVars(self, var):
        varFixed = []  
        for v in var:
            if v is None:
                varFixed.append(0)
            else:
                varFixed.append(v)
        
        return varFixed
    
    def notVar(self, m, var, onlyConstrains = False):
        methodName = "notVar"
        logicMethodName = "NOT"
        
        var = self.__fixVar(var)
        
        varName = var
        if not self.__varIsNumber(var):
            varName = var.VarName
            
        if self.ifLog: self.myLogger.debug("%s called with : %s"%(logicMethodName,varName))

        if not self.__varIsNumber(var):
            cacheResult = self.__isInConstrainCaches(methodName, onlyConstrains, (var,))
            if cacheResult[0]:
                if self.ifLog: self.myLogger.debug("%s constraint already created - doing nothing"%(logicMethodName))
                return cacheResult[1]
            
        # If only constructing constrains forcing NOT to be true 
        if onlyConstrains:
            if self.__varIsNumber(var):
                self.myLogger.warning("%s has set value: %s - do nothing"%(logicMethodName,varName))
                return 
            
            m.addConstr(1 - var >= 1, name='Not:')
            if self.ifLog: self.myLogger.debug("%s created constraint only: not %s > %i"%(logicMethodName,varName,1))

            #self.__addToConstrainCaches(methodName, onlyConstrains, (var, ), None)
            return
        
        # ------- If creating variables representing value of NOT build of provided variable

        if self.__varIsNumber(var):
            if var == 0:
                if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,1))
                return 1
            else:
                if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,0))
                return 0
            
        varNOTName = "not_%s"%(varName)
        varNOTName = varNOTName[:254]
        varNOT = m.addVar(vtype=GRB.BINARY, name=varNOTName)
        if m: m.update()

        m.addConstr(1 - var == varNOT, name='Not:')
        if self.ifLog: self.myLogger.debug("%s created constraint: %i - %s == %s "%(logicMethodName,1,varName,varNOT.VarName))

        # Update cache
        self.__addToConstrainCaches(methodName, onlyConstrains, (var,), varNOT)          

        if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varNOT.VarName))    
        return varNOT
    
    def and2Var(self, m, var1, var2, onlyConstrains = False):
        methodName = "and2Var"
        logicMethodName = "AND"
        
        var1 = self.__fixVar(var1)
        var2 = self.__fixVar(var2)

        # Get names of ILP variables
        var1Name = var1
        var2Name = var2
        if not self.__varIsNumber(var1):
            var1Name = var1.VarName
        if  not self.__varIsNumber(var2):
            var2Name = var2.VarName
            
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1Name,var2Name))
        
        # Check caches
        if not self.__varIsNumber(var1) and not self.__varIsNumber(var2):
            cacheResult = self.__isInConstrainCaches(methodName, onlyConstrains, (var1, var2))
            if cacheResult[0]:
                if self.ifLog: self.myLogger.debug("%s constraint already created - doing nothing"%(logicMethodName))
                if cacheResult[1]:
                    if self.ifLog: self.myLogger.debug("%s returns existing variable: %s"%(logicMethodName,cacheResult[1].VarName))     
                
                return cacheResult[1]
        
        # If only constructing constrains forcing AND to be true 
        if onlyConstrains:
            if self.__varIsNumber(var1) and self.__varIsNumber(var2):
                if var1 == 1 and var2 == 1:
                    if self.ifLog: self.myLogger.debug("%s always True returning %i"%(logicMethodName,1))
                    return 1
                else:
                    if self.ifLog: self.myLogger.debug("%s always False returning %i"%(logicMethodName,0))
                    return 0
            elif self.__varIsNumber(var1):
                if var1 == 1:
                    m.addConstr(var2 >= 1, name='And:')
                    if self.ifLog: self.myLogger.debug("%s created constraint only: %s >= 1"%(logicMethodName,var2Name))
                    return
                else:
                    self.myLogger.error("%s always False: %s is 0"%(logicMethodName,var1Name))
                    return 0
            elif  self.__varIsNumber(var2):
                if var2 == 1:
                    m.addConstr(var1 >= 1, name='And:')
                    if self.ifLog: self.myLogger.debug("%s created constraint only: %s >= 1"%(logicMethodName,var1Name))
                    return
                else:
                    if self.ifLog: self.myLogger.error("%s always False: %s is 0"%(logicMethodName,var2Name))
                    return 0
            else: # Both variables are ILP variables
                m.addConstr(var1 + var2 >= 2, name='And:') 
                if self.ifLog: self.myLogger.debug("% created constraint only: and %s %s >= 2"%(logicMethodName,var1Name,var2.Name))
    
                #self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), None)
                return
        
        # ------- If creating variables representing value of AND build of provided variables
        
        if self.__varIsNumber(var1) and self.__varIsNumber(var2):
            if var1 == 1 and var2 == 1:
                if self.ifLog: self.myLogger.debug("%s always True returning %i"%(logicMethodName,1))
                return 1
            else:
                if self.ifLog: self.myLogger.debug("%s always True returning %i"%(logicMethodName,0))
                return 0
        elif self.__varIsNumber(var1):
            if var1 == 0:
                if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,0))
                return 0
            else:
                if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,var2Name))
                return var2
        elif   self.__varIsNumber(var2):
            if var2 == 0:
                if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,0))
                return 0
            else:
                if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,var1Name))
                return var1
        else:
            varANDName = "and_%s_%s"%(var1Name, var2Name)
            varANDName = varANDName[:254]
            varAND = m.addVar(vtype=GRB.BINARY, name=varANDName)
            if m: m.update()

            m.addConstr(varAND - var1 <= 0, name='And:') # varAND <= var1
            if self.ifLog: self.myLogger.debug("%s created constraint: %s - %s <= %i"%(logicMethodName,varAND.VarName,var1Name,0))

            m.addConstr(varAND - var2 <= 0, name='And:') # varAND <= var2
            if self.ifLog: self.myLogger.debug("%s created constraint: %s - %s <= %i"%(logicMethodName,varAND.VarName,var2Name,0))

            m.addConstr(var1 + var2 - varAND <= 1, name='And:') # var1 + var2 <= varAND + 2 - 1
            if self.ifLog: self.myLogger.debug("%s created constraint: %s + %s - %s <= %i"%(logicMethodName,var1Name,varAND.VarName,var2Name,1))

            # Update cache
            self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), varAND) 
            
            if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varAND.VarName))
            return varAND
    
    def andVar(self, m, *var, onlyConstrains = False):
        methodName = "andVar"
        logicMethodName = "AND"
        
        var = self.__fixVars(var)
        
        # Get names of variables - some of them can be numbers
        noOfVars = 0 # count the numbers in variables
        varsNames = []
        for currentVar in var:
            if self.__varIsNumber(currentVar):
                varsNames.append(currentVar)
            else:
                try:
                    varsNames.append(currentVar.VarName)
                except AttributeError:
                    pass
                
                noOfVars = noOfVars + 1
            
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, varsNames))
        
        # Check number of variables
        N = len(var)
        
        if N <= 1:
            if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,'None'))
            return None
        
        if N == 2: return self.and2Var(m, var[0], var[1], onlyConstrains)
        
        # If all are variables then check cache
        if N - noOfVars == 0:
            cacheResult = self.__isInConstrainCaches(methodName, onlyConstrains, var)
            if cacheResult[0]:
                if self.ifLog: self.myLogger.debug("%s constraint already created - doing nothing"%(logicMethodName))
                if cacheResult[1]:
                    if self.ifLog: self.myLogger.debug("%s returns existing variable: %s"%(logicMethodName,cacheResult[1].VarName))
                
                return cacheResult[1]
        
        # If only constructing constrains forcing AND to be true 
        if onlyConstrains:
            varSumLinExpr = LinExpr()
            for currentVar in var:
                if self.__varIsNumber(currentVar):
                    continue
                
                varSumLinExpr.addTerms(1.0, currentVar)
        
            if self.ifLog: self.myLogger.debug("%s created constraint only: and %s > 1"%(logicMethodName,varSumLinExpr))

            #self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), None)
            m.addConstr(N - noOfVars - varSumLinExpr <= 0, name='And:') # varSumLinExpr >= N
            return
            
        # ------- If creating variables representing value of AND build of provided variables

        # Build new variables name and add it to model
        noOfOnes = 0
        andVarName = ""
        for currentVar in var:
            andVarName = andVarName + "and"
            if self.__varIsNumber(currentVar):
                #orVarName += "_%s_" % (currentVar)
                if currentVar == 1:
                    noOfOnes = noOfOnes + 1
                elif currentVar == 0:
                    if self.ifLog: self.myLogger.debug("%s created no new variable method value is 0 - returning 1"%(logicMethodName))
                    return 0
                else:
                    if self.ifLog: self.myLogger.warning("%s ignoring %f - incorrect"%(logicMethodName,currentVar)) 
            else:
                andVarName += "_%s_" % (currentVar.VarName)

        andVarName = '{:.200}'.format(andVarName)
        
        # If only single variable; rest is zeros 
        if (N - noOfOnes == 1) and noOfOnes == 1:
            for currentVar in var:
                if not self.__varIsNumber(currentVar):
                    if self.ifLog: self.myLogger.debug("%s has ones and only single variable: %s, it is returned"%(logicMethodName,currentVar))
                    return currentVar
                
        # Create new variable
        andVarName = andVarName[:254]
        varAND = m.addVar(vtype=GRB.BINARY, name=andVarName)
        if m: m.update()

        # Build constraints 
        for currentVar in var:
            if self.__varIsNumber(currentVar):
                continue
            
            m.addConstr(varAND - currentVar <= 0, name='And:') # varAND <= currentVar

        varSumLinExpr = LinExpr()
        for currentVar in var:
            if self.__varIsNumber(currentVar):
                continue
                
            varSumLinExpr.addTerms(1.0, currentVar)
        
        m.addConstr(varSumLinExpr - varAND <= N - noOfVars - 1, name='And:') #  varSumLinExpr <= varAND + N - 1
            
        # Update cache
        self.__addToConstrainCaches(methodName, onlyConstrains, var, varAND) 
        
        if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varAND.VarName))
        return varAND
    
    def or2Var(self, m, var1, var2, onlyConstrains = False):
        methodName = "or2Var"
        logicMethodName = "OR"
        
        var1 = self.__fixVar(var1)
        var2 = self.__fixVar(var2)
        
        # Get names of ILP variables
        var1Name = var1
        var2Name = var2
        if not self.__varIsNumber(var1):
            var1Name = var1.VarName
        if not self.__varIsNumber(var2):
            var2Name = var2.VarName
            
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1Name,var2Name))

        # Check caches
        if not self.__varIsNumber(var1) and not self.__varIsNumber(var2):
            cacheResult = self.__isInConstrainCaches(methodName, onlyConstrains, (var1, var2))
            if cacheResult[0]:
                if self.ifLog: self.myLogger.debug("%s constraint already created"%(logicMethodName))
                if cacheResult[1]:
                    if self.ifLog: self.myLogger.debug("%s returns existing variable: %s"%(logicMethodName,cacheResult[1].VarName))
                
                return cacheResult[1]
        
        # If only constructing constrains forcing OR to be true 
        if onlyConstrains:
            if self.__varIsNumber(var1):
                if var1 == 0:
                    m.addConstr(var2 >= 1, name='Or:')
                    if self.ifLog: self.myLogger.debug("%s created constraint only: %s >= %i"%(logicMethodName,var2Name,1))
                    return
                else:
                    if self.ifLog: self.myLogger.debug("%s created no constraint first variable is already %i"%(logicMethodName,1))
                    return
            if self.__varIsNumber(var2):
                if var2 == 0:
                    m.addConstr(var1 >= 1, name='Or:')
                    if self.ifLog: self.myLogger.debug("%s created constraint only: %s >= %i"%(logicMethodName,var1Name,1))
                    return
                else:
                    if self.ifLog: self.myLogger.debug("%s created no constraint second variable is already 1"%(logicMethodName))
                    return
            else:
                m.addConstr(var1 + var2 >= 1, name='Or:') 
                if self.ifLog: self.myLogger.debug("%s created constraint only: %s + %s >= %i"%(logicMethodName,var1Name,var2Name,1))

                #self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), None)
                return
            
        # ------- If creating variables representing value of OR build of provided variables
        
        if self.__varIsNumber(var1): 
            if var1 == 1:
                if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,1))
                return 1
            else:
                if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,var2Name))
                return var2
        if self.__varIsNumber(var1):
            if var2 == 1:
                if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,1))
                return 1
            else:
                if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,var1Name))
                return var1
        else: # Both variables are ILP variables
            varORName = "or_%s_%s"%(var1Name, var2Name)
            varORName = varORName[:254]
            varOR=m.addVar(vtype=GRB.BINARY, name=varORName)
            if m: m.update()

            m.addConstr(var1 - varOR <= 0, name='Or:') # var1 <= varOR
            if self.ifLog: self.myLogger.debug("%s created constraint: %s - %s <= %i"%(logicMethodName,var1Name,var2Name,0))

            m.addConstr(var2 - varOR <= 0, name='Or:') # var2 <= varOR
            if self.ifLog: self.myLogger.debug("%s created constraint: %s - %s <= %i"%(logicMethodName,var2Name,var2Name,0))

            m.addConstr(var1 + var2 - varOR >= 1-1, name='Or:') # var1 + var2 >= varOR
            if self.ifLog: self.myLogger.debug("%s created constraint: %s + %s - %s >= %i"%(logicMethodName,var1Name,var2Name,varOR.VarName,1-1))

            # Update cache
            self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), varOR) 
            
            if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varOR.VarName))
            return varOR
    
    def orVar(self, m, *var, onlyConstrains = False):
        methodName = "orVar"
        logicMethodName = "OR"
        
        var = self.__fixVars(var)
        
        # Get names of variables - some of them can be numbers
        noOfVars = 0 # count the numbers in variables
        varsNames = []
        for currentVar in var:
            if self.__varIsNumber(currentVar):
                varsNames.append(currentVar)
            else:
                varsNames.append(currentVar.VarName)
                noOfVars = noOfVars + 1
            
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, varsNames))
        
        # Check number of variables
        N = len(var)
        
        if N <= 1:
            if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,'None'))
            return None
        
        if N == 2: return self.or2Var(m, var[0], var[1], onlyConstrains)
        
        # If all are variables then check cache
        if N - noOfVars == 0:
            cacheResult = self.__isInConstrainCaches(methodName, onlyConstrains, var)
            if cacheResult[0]:
                if self.ifLog: self.myLogger.debug("%s constraint already created - doing nothing"%(logicMethodName))
                if cacheResult[1]:
                    if self.ifLog: self.myLogger.debug("%s returns existing variable: %s"%(logicMethodName,cacheResult[1].VarName))
                
                return cacheResult[1]
        
        # If only constructing constrains forcing OR to be true 
        if onlyConstrains:
            varSumLinExpr = LinExpr()
            for currentVar in var:
                if not self.__varIsNumber(currentVar):
                    varSumLinExpr.addTerms(1.0, currentVar)
                elif currentVar == 1: # currentVar is Number 
                    if self.ifLog: self.myLogger.debug("%s created no constraint variable is already %f"%(logicMethodName, currentVar))
                    return
                elif currentVar == 0: # currentVar is Number 
                    if self.ifLog: self.myLogger.debug("%s ignoring %f has not effect on value"%(logicMethodName,currentVar)) 
                else:
                    if self.ifLog: self.myLogger.warning("%s ignoring %f - incorrect"%(logicMethodName,currentVar)) 

            if varSumLinExpr.size() == 0:
                if self.ifLog: self.myLogger.debug("%s created no constraint - the value of the method is 0"%(logicMethodName))
                return
            
            m.addConstr(varSumLinExpr >= 1, name='Or:')
            
            varSumLinExprStr = str(varSumLinExpr)
            if self.ifLog: self.myLogger.debug("%s created constraint only: %s >= %i"%(logicMethodName,varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('>')],1))
            
            if N - noOfVars == 0: self.__addToConstrainCaches(methodName, onlyConstrains, var, None)           
            return
        
        # ------- If creating variables representing value of OR build of provided variables
        
        # Build new variables name and add it to model
        noOfZeros = 0
        orVarName = ""
        for currentVar in var:
            orVarName = orVarName + "or"
            if self.__varIsNumber(currentVar):
                #orVarName += "_%s_" % (currentVar)
                if currentVar == 1:
                    if self.ifLog: self.myLogger.debug("%s created no new variable method value is 1 - returning 1"%(logicMethodName))
                    return 1
                elif currentVar == 0:
                    noOfZeros = noOfZeros + 1
                else:
                    if self.ifLog: self.myLogger.warning("%s ignoring %f - incorrect"%(logicMethodName,currentVar)) 
            else:
                orVarName += "_%s_" % (currentVar.VarName)

        orVarName = '{:.200}'.format(orVarName)

        # If only single variable; rest is zeros 
        if (N - noOfZeros == 1) and noOfVars == 1:
            for currentVar in var:
                if not self.__varIsNumber(currentVar):
                    if self.ifLog: self.myLogger.debug("%s has zeros and only single variable: %s, it is returned"%(logicMethodName,currentVar))
                    return currentVar
                
        # Create new variable
        orVarName = orVarName[:254]
        varOR = m.addVar(vtype=GRB.BINARY, name=orVarName)
        if m: m.update()

        # Build constrains
        for currentVar in var:
            if not self.__varIsNumber(currentVar):
                m.addConstr(currentVar - varOR <= 0, name='Or:') # currentVar <= varOR
                if self.ifLog: self.myLogger.debug("%s created constraint: %s - %s <= %i"%(logicMethodName,currentVar.VarName,orVarName,0))
            else:
                pass # Only 0 possible now - has no effect on Or value

        varSumLinExpr = LinExpr()
        for currentVar in var:
            if not self.__varIsNumber(currentVar):
                varSumLinExpr.addTerms(1.0, currentVar)
            else:
                pass
            
        m.addConstr(varSumLinExpr - varOR >= 1-1, name='Or:') # varSumLinExpr >= varOR
        varSumLinExprStr = str(varSumLinExpr)
        if self.ifLog: self.myLogger.debug("%s created constraint: %s - %s >= %i"
                                           %(logicMethodName,varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('>')],orVarName,1-1))

        # Update Cache
        if N - noOfVars == 0: self.__addToConstrainCaches(methodName, onlyConstrains, var, varOR)
             
        if self.ifLog: self.myLogger.debug("%s returns new variable: %s"%(logicMethodName,varOR.VarName))
        return varOR
    
    def nand2Var(self, m, var1, var2, onlyConstrains = False):
        methodName = "nand2Var"
        logicMethodName = "NAND"
        
        var1 = self.__fixVar(var1)
        var2 = self.__fixVar(var2)
        
        # Get names of ILP variables
        var1Name = var1
        var2Name = var2
        if not self.__varIsNumber(var1):
            var1Name = var1.VarName
        if not self.__varIsNumber(var2):
            var2Name = var2.VarName
            
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1Name,var2Name))
        
        # Check caches
        if not self.__varIsNumber(var1) and not self.__varIsNumber(var2):
            cacheResult = self.__isInConstrainCaches(methodName, onlyConstrains, (var1, var2))
            if cacheResult[0]:
                if self.ifLog: self.myLogger.debug("%s constraint already created - doing nothing"%(logicMethodName))
                return cacheResult[1]
        
        # If only constructing constrains forcing NAND to be true 
        if onlyConstrains:
            if self.__varIsNumber(var1) and self.__varIsNumber(var2):
                if var1 == 1 and var2 == 1:
                    if self.ifLog: self.myLogger.debug("%s always False returning %i"%(logicMethodName,0))
                    return 0
                else:
                    if self.ifLog: self.myLogger.debug("%s always True returning %i"%(logicMethodName,1))
                    return 1
            elif self.__varIsNumber(var1):
                if var1 == 1:
                    m.addConstr(var2 <= 0, name='Nand:')
                    if self.ifLog: self.myLogger.debug("%s created constraint only: %s <= %i"%(logicMethodName,var2Name,0))
                    return
                else:
                    self.myLogger.error("%s always True: %s is %i"%(logicMethodName,var1Name,var1))
                    return 1
            elif self.__varIsNumber(var2):
                if var2 == 1:
                    m.addConstr(var1 <= 0, name='Nand:')
                    if self.ifLog: self.myLogger.debug("%s created constraint only: %s <= %i"%(logicMethodName,var1Name,0))
                    return
                else:
                    self.myLogger.error("%s always True: %s is %i"%(logicMethodName,var2Name,var2))
                    return 1
            else: # Both variables are ILP variables
                m.addConstr(var1 + var2 <= 1, name='Nand:')
                if self.ifLog: self.myLogger.debug("%s created constraint only: and %s %s <= %i"%(logicMethodName,var1Name,var2Name,1))
    
                #self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), None)
                return
        
        # ------- If creating variables representing value of NAND build of provided variables
        
        if self.__varIsNumber(var1) and self.__varIsNumber(var2):
            if var1 == 1 and var2 == 1:
                if self.ifLog: self.myLogger.debug("%s always False returning %i"%(logicMethodName,0))
                return 0
            else:
                if self.ifLog: self.myLogger.debug("%s always True returning %i"%(logicMethodName,1))
                return 1
        elif self.__varIsNumber(var1):
            if var1 == 0:
                if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,1))
                return 1
            else:
                if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,var2Name))
                return var2
        elif self.__varIsNumber(var2):
            if var2 == 0:
                if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,1))
                return 1
            else:
                if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,var2Name))
                return var1
        else:
            varNANDName = "nand_%s_%s"%(var1, var2)
            varNANDName = varNANDName[:254]
            varNAND = m.addVar(vtype=GRB.BINARY, name=varNANDName)
            if m: m.update()

            m.addConstr(self.notVar(m, varNAND) <= var1, name='Nand:')
            if self.ifLog: self.myLogger.debug("%s created constraint: %s - %s <= 0 "%(logicMethodName,varNAND.VarName,var1Name))

            m.addConstr(self.notVar(m, varNAND) <= var2, name='Nand:') # varAND <= var2
            if self.ifLog: self.myLogger.debug("%s created constraint: %s - %s <= 0 "%(logicMethodName,varNAND.VarName,var2Name))

            m.addConstr(var1 + var2 <= self.notVar(m, varNAND) + 2 - 1, name='Nand:') # var1 + var2 <= varAND + 2 - 1
            if self.ifLog: self.myLogger.debug("%s created constraint: %s + %s - %s <= 1 "%(logicMethodName,var1Name,var2Name,varNAND.VarName))

            # Update cache
            self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), varNAND) 
            
            if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varNAND.VarName))
            
            return varNAND
    
    def nandVar(self, m, *var, onlyConstrains = False):
        methodName = "nandVar"
        logicMethodName = "NAND"
       
        var = self.__fixVars(var)
       
        # Get names of variables - some of them can be numbers
        noOfVars = 0 # count the numbers in variables
        varsNames = []
        for currentVar in var:
            if self.__varIsNumber(currentVar):
                varsNames.append(currentVar)
            else:
                varsNames.append(currentVar)
                noOfVars = noOfVars + 1
            
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, varsNames))
        
        # Check number of variables
        N = len(var)
        
        if N <= 1:
            if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,'None'))
            return None
        
        if N == 2: return self.nand2Var(m, var[0], var[1], onlyConstrains)
    
        # If all are variables then check cache
        if N - noOfVars == 0:
            cacheResult = self.__isInConstrainCaches(methodName, onlyConstrains, var)
            if cacheResult[0]:
                if cacheResult[1]:
                    if self.ifLog: self.myLogger.debug("%s returns existing variable: %s"%(logicMethodName,cacheResult[1].VarName))
                else:
                    if self.ifLog: self.myLogger.debug("%s returns existing variable: %s"%(logicMethodName,None))

                return cacheResult[1]
        
        # If only constructing constrains forcing NAND to be true 
        if onlyConstrains:
            varSumLinExpr = LinExpr()
            for currentVar in var:
                varSumLinExpr.addTerms(1.0, currentVar)
        
            m.addConstr(varSumLinExpr <= N - 1, name='Nand:')
                        
            varSumLinExprStr = str(varSumLinExpr)
            if self.ifLog: self.myLogger.debug("NAND created constraint only: %s <= %i"%(varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('>')], N-1))
            
            self.__addToConstrainCaches('nandVar', onlyConstrains, var, None)
      
            return
        
        # ------- If creating variables representing value of OR build of provided variables
        nandVarName = "nand"
        for currentVar in var:
            nandVarName += "_%s"%(currentVar)
            
        nandVarName = nandVarName[:254]
        varNAND = m.addVar(vtype=GRB.BINARY, name=nandVarName)
        for currentVar in var:
            m.addConstr(self.notVar(m, varNAND) <= currentVar, name='Nand:')

        varSumLinExpr = LinExpr()
        for currentVar in var:
            varSumLinExpr.addTerms(1.0, currentVar)
    
        m.addConstr(varSumLinExpr <= self.notVar(m, varNAND) + N - 1, name='Nand:')
    
        self.__addToConstrainCaches('nandVar', onlyConstrains, var, varNAND)

        return varNAND
    
    def nor2Var(self, m, var1, var2, onlyConstrains = False):
        if onlyConstrains:
            m.addConstr(var1 + var2 <= 0)
            return
        
        varNORName = "nor_%s_%s"%(var1, var2)
        varNORName = varNORName[:254]
        varNOR = m.addVar(vtype=GRB.BINARY, name=varNORName)
            
        m.addConstr(var1 <= self.notVar(m, varNOR))
        m.addConstr(var2 <= self.notVar(m, varNOR))
            
        m.addConstr(var1 + var2 >= self.notVar(m, varNOR))
        
        return varNOR
    
    def norVar(self, m, *var, onlyConstrains = False):
        
        var = self.__fixVars(var)
        
        N = len(var)
        
        if N <= 1:
            return None
        
        if onlyConstrains:
            varSumLinExpr = LinExpr()
            for currentVar in var:
                varSumLinExpr.addTerms(1.0, currentVar)
        
            m.addConstr(varSumLinExpr <= 0)
            return
        
        _norVarName = "nor"
        for currentVar in var:
            _norVarName += "_%s"%(currentVar)
           
        _norVarName = _norVarName [:254]
        varNOR = m.addVar(vtype=GRB.BINARY, name=_norVarName)
        for currentVar in var:
            m.addConstr(currentVar <= self.notVar(m, varNOR))
        
        varSumLinExpr = LinExpr()
        for currentVar in var:
            varSumLinExpr.addTerms(1.0, currentVar)
            
            m.addConstr(varSumLinExpr >= self.notVar(m, varNOR))
    
        return varNOR
    
    def xorVar(self, m, var1, var2, onlyConstrains = False):
        #if self.ifLog: self.myLogger.debug("XOR called with : %s"%(var1,var2))
        
        cacheResult = self.__isInConstrainCaches('xorVar', onlyConstrains, (var1, var2))
        if cacheResult[0]:
            return cacheResult[1]

        var1 = self.__fixVar(var1)
        var2 = self.__fixVar(var2)
        if (not var1) or (not var2):
            return
    
        # Get names of ILP variables
        var1Name = var1
        var2Name = var2
        if not self.__varIsNumber(var1):
            var1Name = var1.VarName
        if not self.__varIsNumber(var2):
            var2Name = var2.VarName
            
        if onlyConstrains:
            m.addConstr(var1 + var2 <= 1)
            m.addConstr(var1 + var2 >= 1)
            if self.ifLog: self.myLogger.debug("IF created constraint only: %s <= %s"%(var1Name, var2Name))

            self.__addToConstrainCaches('ifVar', onlyConstrains, (var1, var2), None)
            return
        
        varXORName = "xor_%s_%s"%(var1, var2)
        varXORName = varXORName[:254]
        varXOR = m.addVar(vtype=GRB.BINARY, name=varXORName)
            
        m.addConstr(var1 + var2 + varXOR <= 2)
        m.addConstr(-var1 - var2 + varXOR <= 0)
        m.addConstr(var1 - var2 + varXOR >= 0)
        m.addConstr(-var1 + var2 + varXOR >= 0)
            
        return varXOR
    
    def ifVar(self, m, var1, var2, onlyConstrains = False):
        methodName = "ifVar"
        logicMethodName = "IF"

        var1 = self.__fixVar(var1)
        var2 = self.__fixVar(var2)
        if (not var1) or (not var2):
            return
    
        # Get names of ILP variables
        var1Name = var1
        var2Name = var2
        if not self.__varIsNumber(var1):
            var1Name = var1.VarName
        if not self.__varIsNumber(var2):
            var2Name = var2.VarName
            
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1Name,var2Name))
    
        # Check caches
        if not self.__varIsNumber(var1) and not self.__varIsNumber(var2):
            cacheResult = self.__isInConstrainCaches(methodName, onlyConstrains, (var1, var2))
            if cacheResult[0]:
                if self.ifLog: self.myLogger.debug("%s constraint already created"%(logicMethodName))
                if self.ifLog: self.myLogger.debug("%s returns existing variable: %s"%(logicMethodName,cacheResult[1].VarName))
                return cacheResult[1]
        
        # If only constructing constrains forcing OR to be true 
        if onlyConstrains:
            if self.__varIsNumber(var1) and self.__varIsNumber(var2):
                if var1 == 1 and var2 == 0:
                    if self.ifLog: self.myLogger.debug("%s is False returning %i"%(logicMethodName,0))
                    return 0
                else:
                    if self.ifLog: self.myLogger.debug("%s is True returning %i"%(logicMethodName,1))
                    return 1
            elif self.__varIsNumber(var1):
                if var1 == 0:
                    if self.ifLog: self.myLogger.debug("%s is True returning %i"%(logicMethodName,1))
                    return 1
                else:
                    if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,var2Name))
                    return var2
            elif   self.__varIsNumber(var2):
                if var2 == 1:
                    if self.ifLog: self.myLogger.debug("%s is True returning %i"%(logicMethodName,1))
                    return 1
                else:
                    notVar1 = self.notVar(m, var1)
                    if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,notVar1.VarName))
                    return notVar1
            else:
                m.addConstr(var1 - var2 <= 0, name='If:') #var1 <= var1
                if self.ifLog: self.myLogger.debug("%s created constraint only: %s <= %s"%(logicMethodName,var1Name,var2Name))
                return
    
        varIFName = "if_%s_then_%s"%(var1Name, var2Name)
        varIFName = varIFName[:254]
        varIF = m.addVar(vtype=GRB.BINARY, name=varIFName)
            
        m.addConstr(1 - var1 <= varIF, name='If:')
        m.addConstr(var2 <= varIF, name='If:')
        m.addConstr(1 - var1 + var2 >= varIF, name='If:')
            
        m.update()

        self.__addToConstrainCaches('ifVar', onlyConstrains, (var1, var2), varIF)
        
        if self.ifLog: self.myLogger.debug("IF returns : %s"%(varIF.VarName))

        return varIF
           
    def epqVar(self, m, var1, var2, onlyConstrains = False):
        #if self.ifLog: self.myLogger.debug("EQ called with : %s"%(var1,var2))

        cacheResult = self.__isInConstrainCaches('eqVar', onlyConstrains, (var1, var2))
        if cacheResult[0]:
            if self.ifLog: self.myLogger.debug("EQ constraint already created - doing nothing")
            return cacheResult[1]
        
        var1 = self.__fixVar(var1)
        var2 = self.__fixVar(var2)
        if (not var1) or (not var2):
            return
    
        # Get names of ILP variables
        var1Name = var1
        var2Name = var2
        if not self.__varIsNumber(var1):
            var1Name = var1.VarName
        if not self.__varIsNumber(var2):
            var2Name = var2.VarName
            
        if onlyConstrains:
            m.addConstr(var1 >= var2)
            if self.ifLog: self.myLogger.debug("EQ created constraint only: %s => %s"%(var1Name, var2Name))
            
            m.addConstr(var1 <= var2)
            if self.ifLog: self.myLogger.debug("EQ created constraint only: %s <= %s"%(var1Name, var2Name))

            self.__addToConstrainCaches('eqVar', onlyConstrains, (var1, var2), None)
            return
        
        varEQName = "epq_%s_%s"%(var1, var2)
        varEQName = varEQName[:254]
        varEQ = m.addVar(vtype=GRB.BINARY, name=varEQName)
            
        m.addConstr(var1 + var2 - varEQ <= 1)
        m.addConstr(var1 + var2 + varEQ >= 1)
        m.addConstr(-var1 + var2 + varEQ <= 1)
        m.addConstr(var1 - var2 + varEQ <= 1)
        
        m.update()
             
        self.__addToConstrainCaches('eqVar', onlyConstrains, (var1, var2), varEQ)

        if self.ifLog: self.myLogger.debug("EQ returns : %s"%(varEQ.VarName))
        return varEQ
    
    def countVar(self, m, *var, onlyConstrains = False, limitOp = 'None', limit = 1, logicMethodName = "COUNT"):
        methodName = "countVar"
        #logicMethodName = "COUNT"
        
        var = self.__fixVars(var)
        
        if not limitOp:
            if self.ifLog: self.myLogger.error("%s called with no operation specified for comparing limit"%(logicMethodName))
            return None

        if limitOp not in ('<=', '>=', '=='):
            if self.ifLog: self.myLogger.error("%s called with incorrect operation specified for comparing limit %s"%(logicMethodName,limitOp))
            return None
            
        if self.ifLog: self.myLogger.debug("%s called with limit: %i and operation %s"%(logicMethodName,limit,limitOp))

        # Get names of variables - some of them can be numbers
        noOfVars = 0 # count the numbers in variables
        varsNames = []
        for currentVar in var:
            if self.__varIsNumber(currentVar):
                varsNames.append(currentVar)
            else:
                varsNames.append(currentVar.VarName)
                noOfVars = noOfVars + 1
            
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, varsNames))
        
        # Purge vars
        varSet = set()
        for v in var:
            varSet.add(v)
            
        var = varSet
        
        # Check number of variables
        N = len(var)
        
        # If only constructing constrains forcing OR to be true 
        if onlyConstrains:
            varSumLinExpr = LinExpr()
            countOnes = 0
            for currentVar in var:
                if not self.__varIsNumber(currentVar):
                    varSumLinExpr.addTerms(1.0, currentVar)
                elif currentVar == 1: # currentVar is Number 
                    countOnes = countOnes + 1

            if limitOp == '>=':
                if countOnes > limit:
                    if self.ifLog: self.myLogger.debug("%s created no constraint - the value of the method is True"%(logicMethodName))
                    return 1
                elif varSumLinExpr.size() - (limit - countOnes) < 0:
                    if self.ifLog: self.myLogger.debug("%s created no constraint - the value of the method is False"%(logicMethodName))
                    return 0
                else:
                    m.addConstr(varSumLinExpr >= limit - countOnes, name='Count %s:'%(logicMethodName))
            if limitOp == '<=':
                if varSumLinExpr.size() == 0:
                    if countOnes < limit:
                        if self.ifLog: self.myLogger.debug("%s created no constraint - the value of the method is True"%(logicMethodName))
                        return 1
                    else:
                        if self.ifLog: self.myLogger.debug("%s created no constraint - the value of the method is False"%(logicMethodName))
                        return 0
                else:
                    if limit < countOnes:
                        if self.ifLog: self.myLogger.debug("%s created no constraint - the value of the method is False"%(logicMethodName))
                        return 0
                    else:
                        m.addConstr(varSumLinExpr <= limit - countOnes, name='Count %s:'%(logicMethodName))
            if limitOp == '==':
                if varSumLinExpr.size() == 0:
                    if countOnes == limit:
                        if self.ifLog: self.myLogger.debug("%s created no constraint - the value of the method is True"%(logicMethodName))
                        return 1
                    else:
                        if self.ifLog: self.myLogger.debug("%s created no constraint - the value of the method is False"%(logicMethodName))
                        return 0
                else:
                    m.addConstr(varSumLinExpr == limit - countOnes, name='Count %s:'%(logicMethodName))
                 
            varSumLinExprStr = str(varSumLinExpr)
            if self.ifLog: self.myLogger.debug("%s created constraint only: %s %s= %i - %i"
                                               %(logicMethodName,varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('>')],limitOp,limit,countOnes))
            
            return
        
        # ------- If creating variables representing value of OR build of provided variables
        
        # Build new variables name and add it to model
        countVarName = logicMethodName
        for currentVar in var:
            if self.__varIsNumber(currentVar):
                countVarName += "_%s_"%(currentVar)
            else:
                countVarName += "_%s_"%(currentVar.VarName)
            
        countVarName = countVarName[:-1]
        countVarName = countVarName[:254]
        # Create new variable
        varCOUNT = m.addVar(vtype=GRB.BINARY, name=countVarName)
        if m: m.update()

        # Build constrains
        varSumLinExpr = LinExpr()
        countOnes = 0
        for currentVar in var:
            if not self.__varIsNumber(currentVar):
                varSumLinExpr.addTerms(1.0, currentVar)
            elif currentVar == 1:
                countOnes = countOnes + 1
                
        varSumLinExprStr = str(varSumLinExpr)
        
        if limitOp == '>=':
            if countOnes > limit:
                if self.ifLog: self.myLogger.debug("%s created no constraint - the value of the method is True"%(logicMethodName))
                return 1
            elif varSumLinExpr.size() - (limit - countOnes) < 0:
                if self.ifLog: self.myLogger.debug("%s created contradictory not constraint - the value of the method is False"%(logicMethodName))
                return 0
            else:
                m.addConstr(varSumLinExpr - varCOUNT >= limit - 1 - countOnes, name='Count %s:'%(logicMethodName))
                if self.ifLog: self.myLogger.debug("%s created constraint: %s - varCOUNT %s= %i - 1 - %i"
                                                   %(logicMethodName,varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('>')],limitOp,limit,countOnes))

        if limitOp == '<=':
            if varSumLinExpr.size() == 0:
                if countOnes < limit:
                    if self.ifLog: self.myLogger.debug("%s created no constraint - the value of the method is True"%(logicMethodName))
                    return 1
                else:
                    if self.ifLog: self.myLogger.debug("%s created no constraint - the value of the method is False"%(logicMethodName))
                    return 0
            else:
                if limit < countOnes:
                    if self.ifLog: self.myLogger.debug("%s created no constraint - the value of the method is False"%(logicMethodName))
                    return 0
                else:
                    m.addConstr(varSumLinExpr - varCOUNT <= limit - 1 - countOnes, name='Count %s:'%(logicMethodName))
                    if self.ifLog: self.myLogger.debug("%s created constraint: %s - varCOUNT %s= %i - 1 - %i"
                                                       %(logicMethodName,varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('<')],limitOp,limit,countOnes))

        if limitOp == '==':
            if varSumLinExpr.size() == 0:
                if countOnes == limit:
                    if self.ifLog: self.myLogger.debug("%s created no constraint - the value of the method is True"%(logicMethodName))
                    return 1
                else:
                    if self.ifLog: self.myLogger.debug("%s created no constraint - the value of the method is False"%(logicMethodName))
                    return 0
            else:
                m.addConstr(varSumLinExpr == limit - countOnes, name='Count %s:'%(logicMethodName))
                if self.ifLog: self.myLogger.debug("%s created constraint: %s - varCOUNT %s= %i - 1 - %i"
                                                   %(logicMethodName,varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('=')],limitOp,limit,countOnes))

        if self.ifLog: self.myLogger.debug("%s returns new variable: %s"%(logicMethodName,varCOUNT.VarName))
        return varCOUNT
    
    def fixedVar(self, m, var, onlyConstrains = False): 
        methodName = "fixedVar"
        logicMethodName = "FIXED"
        
        if var is None: # not create Fixed constraint for None
            return
        
        var = self.__fixVar(var)
        
        varName = var
        if not self.__varIsNumber(var):
            varName = var.VarName
            
        if self.ifLog: self.myLogger.debug("%s called with : %s"%(logicMethodName,varName))

        # If only constructing constrains 
       
        if self.__varIsNumber(var):
            self.myLogger.warning("%s has set value: %s - do nothing"%(logicMethodName,varName))
            return 
        
        fixedTag = None
        if var.VTag.startswith("True"):
            fixedTag = True
        elif var.VTag.startswith("False"):
            fixedTag = False
        
        if fixedTag == None: # Label in datanode was -100 
            return 1
        
        if fixedTag:    
            m.addConstr(var == 1, name='Fixed:')
            if self.ifLog: self.myLogger.debug("%s created constraint: Fixed %s == %i"%(logicMethodName,varName,1))

        elif not fixedTag:    
            m.addConstr(var == 0, name='Fixed:')
            if self.ifLog: self.myLogger.debug("%s created constraint: Fixed %s == %i"%(logicMethodName,varName,0))

        else:
            return # error
        
        if onlyConstrains:
            return
        
        if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,1))
        return 1