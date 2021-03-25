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
        if lmName in self.constrainCaches:
            if onlyConstrains in self.constrainCaches[lmName]:
                self.constrainCaches[lmName][onlyConstrains][var] = cachedValue
            else:
                self.constrainCaches[lmName][onlyConstrains] = {}
                self.constrainCaches[lmName][onlyConstrains][var] = cachedValue
        else:
            self.constrainCaches[lmName] = {}
            self.constrainCaches[lmName][onlyConstrains] = {}

            self.constrainCaches[lmName][onlyConstrains][var] = cachedValue
            
    def __isInConstrainCaches(self, lmName, onlyConstrains, var):
        if lmName in self.constrainCaches:
            if onlyConstrains in self.constrainCaches[lmName]:
                for currentVarPermutation in permutations(var):
                    if currentVarPermutation in self.constrainCaches[lmName][onlyConstrains]:
                        #if self.ifLog: self.myLogger.debug("%s already created constrain for this variables %s - does nothing"%(lmName, [x.VarName for x in var]))
                        return (True, self.constrainCaches[lmName][onlyConstrains][currentVarPermutation])
                    
        return (False, None)
                
    def __varIsNumber(self, var):
        return not isinstance(var, Var)
    
    def notVar(self, m, var, onlyConstrains = False):
        methodName = "notVar"
        logicMethodName = "NOT"
        
        varName = var
        if not self.__varIsNumber(var):
            varName = var.VarName
            
        if self.ifLog: self.myLogger.debug("%s called with : %s"%(logicMethodName,varName))

        if not self.__varIsNumber(var):
            cacheResult = self.__isInConstrainCaches(methodName, onlyConstrains, (var,))
            if cacheResult[0]:
                if self.ifLog: self.myLogger.debug("%s constrain already created - doing nothing"(logicMethodName))
                return cacheResult[1]
            
        # If only constructing constrains forcing NOT to be true 
        if onlyConstrains:
            if self.__varIsNumber(var):
                self.myLogger.warning("%s has set value: %s - do nothing"%(logicMethodName,varName))
                return 
            
            m.addConstr(1 - var >= 1)
            if self.ifLog: self.myLogger.debug("%s created constrain only: not %s > %i"%(logicMethodName,var.VarName,1))

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
            
        varNOT = m.addVar(vtype=GRB.BINARY, name="not_%s"%(var.VarName))
        if m: m.update()

        m.addConstr(1 - var == varNOT)
        if self.ifLog: self.myLogger.debug("%s created constrain: %i - %s == %s "%(logicMethodName,1,varName,varNOT.VarName))

        # Update cache
        self.__addToConstrainCaches(methodName, onlyConstrains, (var,), varNOT)          

        if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varNOT.VarName))    
        return varNOT
    
    def and2Var(self, m, var1, var2, onlyConstrains = False):
        methodName = "and2Var"
        logicMethodName = "AND"
        
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
                if self.ifLog: self.myLogger.debug("%s constrain already created - doing nothing"%(logicMethodName))
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
                    m.addConstr(var2 >= 1)
                    if self.ifLog: self.myLogger.debug("%s created constrain only: %s >= 1"%(logicMethodName,var2Name))
                    return
                else:
                    self.myLogger.error("%s always False: %s is 0"%(logicMethodName,var1Name))
                    return 0
            elif  self.__varIsNumber(var2):
                if var2 == 1:
                    m.addConstr(var1 >= 1)
                    if self.ifLog: self.myLogger.debug("%s created constrain only: %s >= 1"%(logicMethodName,var1Name))
                    return
                else:
                    if self.ifLog: self.myLogger.error("%s always False: %s is 0"%(logicMethodName,var2Name))
                    return 0
            else: # Both variables are ILP variables
                m.addConstr(var1 + var2 >= 2) 
                if self.ifLog: self.myLogger.debug("% created constrain only: and %s %s >= 2"%(logicMethodName,var1.VarName,var2.Name))
    
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
            varAND = m.addVar(vtype=GRB.BINARY, name="and_%s_%s"%(var1Name, var2Name))
            if m: m.update()

            m.addConstr(varAND - var1 <= 0) # varAND <= var1
            if self.ifLog: self.myLogger.debug("%s created constrain: %s - %s <= %i"%(logicMethodName,varAND.VarName,var1Name,0))

            m.addConstr(varAND - var2 <= 0) # varAND <= var2
            if self.ifLog: self.myLogger.debug("%s created constrain: %s - %s <= %i"%(logicMethodName,varAND.VarName,var2Name,0))

            m.addConstr(var1 + var2 - varAND <= 1) # var1 + var2 <= varAND + 2 - 1
            if self.ifLog: self.myLogger.debug("%s created constrain: %s + %s - %s <= %i"%(logicMethodName,var1Name,varAND.VarName,var2Name,1))

            # Update cache
            self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), varAND) 
            
            if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varAND.VarName))
            return varAND
    
    def andVar(self, m, *var, onlyConstrains = False):
        methodName = "andVar"
        logicMethodName = "AND"
        
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
        
        if N == 2: return self.and2Var(m, var[0], var[1], onlyConstrains)
        
        # If all are variables then check cache
        if N - noOfVars == 0:
            cacheResult = self.__isInConstrainCaches(methodName, onlyConstrains, var)
            if cacheResult[0]:
                if self.ifLog: self.myLogger.debug("%s constrain already created - doing nothing"%(logicMethodName))
                if cacheResult[1]:
                    if self.ifLog: self.myLogger.debug("%s returns existing variable: %s"%(logicMethodName,cacheResult[1].VarName))
                
                return cacheResult[1]
        
        # If only constructing constrains forcing AND to be true 
        if onlyConstrains:
            varSumLinExpr = LinExpr()
            for currentVar in var:
                varSumLinExpr.addTerms(1.0, currentVar)
        
            if self.ifLog: self.myLogger.debug("%s created constrain only: and %s > 1"%(logicMethodName,varSumLinExpr))

            #self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), None)
            m.addConstr(varSumLinExpr >= N)
            return
            
        # ------- If creating variables representing value of AND build of provided variables

        varAND = m.addVar(vtype=GRB.BINARY)
        if m: m.update()

        for currentVar in var:
            m.addConstr(varAND - currentVar <= 0) # varAND <= currentVar

        varSumLinExpr = LinExpr()
        for currentVar in var:
            varSumLinExpr.addTerms(1.0, currentVar)
        
        m.addConstr(varSumLinExpr - varAND <= N - 1) # varSumLinExpr <= varAND + N - 1
            
        # Update cache
        self.__addToConstrainCaches(methodName, onlyConstrains, var, varAND) 
        
        if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varAND.VarName))
        return varAND
    
    def or2Var(self, m, var1, var2, onlyConstrains = False):
        methodName = "or2Var"
        logicMethodName = "OR"
        
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
                if self.ifLog: self.myLogger.debug("%s constrain already created"%(logicMethodName))
                if cacheResult[1]:
                    if self.ifLog: self.myLogger.debug("%s returns existing variable: %s"%(logicMethodName,cacheResult[1].VarName))
                
                return cacheResult[1]
        
        # If only constructing constrains forcing OR to be true 
        if onlyConstrains:
            if self.__varIsNumber(var1):
                if var1 == 0:
                    m.addConstr(var2 >= 1)
                    if self.ifLog: self.myLogger.debug("%s created constrain only: %s >= %i"%(logicMethodName,var2Name,1))
                    return
                else:
                    if self.ifLog: self.myLogger.debug("%s created no constrain first variable is already %i"%(logicMethodName,1))
                    return
            if self.__varIsNumber(var2):
                if var2 == 0:
                    m.addConstr(var1 >= 1)
                    if self.ifLog: self.myLogger.debug("%s created constrain only: %s >= %i"%(logicMethodName,var1Name,1))
                    return
                else:
                    if self.ifLog: self.myLogger.debug("%s created no constrain second variable is already 1"%(logicMethodName))
                    return
            else:
                m.addConstr(var1 + var2 >= 1) 
                if self.ifLog: self.myLogger.debug("%s created constrain only: %s + %s >= %i"%(logicMethodName,var1Name,var2Name,1))

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
            varOR=m.addVar(vtype=GRB.BINARY, name="or_%s_%s"%(var1Name, var2Name))
            if m: m.update()

            m.addConstr(var1 - varOR <= 0) # var1 <= varOR
            if self.ifLog: self.myLogger.debug("%s created constrain: %s - %s <= %i"%(logicMethodName,var1Name,var2Name,0))

            m.addConstr(var2 - varOR <= 0) # var2 <= varOR
            if self.ifLog: self.myLogger.debug("%s created constrain: %s - %s <= %i"%(logicMethodName,var2Name,var2Name,0))

            m.addConstr(var1 + var2 - varOR >= 1-1) # var1 + var2 >= varOR
            if self.ifLog: self.myLogger.debug("%s created constrain: %s + %s - %s >= %i"%(logicMethodName,var1Name,var2Name,varOR.VarName,1-1))

            # Update cache
            self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), varOR) 
            
            if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varOR.VarName))
            return varOR
    
    def orVar(self, m, *var, onlyConstrains = False):
        methodName = "orVar"
        logicMethodName = "OR"
        
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
                if self.ifLog: self.myLogger.debug("%s constrain already created - doing nothing"%(logicMethodName))
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
                    if self.ifLog: self.myLogger.debug("%s created no constrain variable is already %f"%(logicMethodName, currentVar))
                    return
                elif currentVar == 0: # currentVar is Number 
                    if self.ifLog: self.myLogger.debug("%s ignoring %f has not effect on value"%(logicMethodName,currentVar)) 
                else:
                    if self.ifLog: self.myLogger.warning("%s ignoring %f - incorrect"%(logicMethodName,currentVar)) 

            if varSumLinExpr.size() == 0:
                if self.ifLog: self.myLogger.debug("%s created no constrain - the value of the method is 0"%(logicMethodName))
                return
            
            m.addConstr(varSumLinExpr >= 1)
            
            varSumLinExprStr = str(varSumLinExpr)
            if self.ifLog: self.myLogger.debug("%s created constrain only: %s >= %i"%(logicMethodName,varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('>')],1))
            
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

        # If only single variable; rest is zeros 
        if (N - noOfZeros == 1) and noOfVars == 1:
            for currentVar in var:
                if not self.__varIsNumber(currentVar):
                    if self.ifLog: self.myLogger.debug("%s has zeros and only single variable: %s, it is returned"%(logicMethodName,currentVar.VarName))
                    return currentVar
                
        # Create new variable
        varOR = m.addVar(vtype=GRB.BINARY, name=orVarName)
        if m: m.update()

        # Build constrains
        for currentVar in var:
            if not self.__varIsNumber(currentVar):
                m.addConstr(currentVar - varOR <= 0) # currentVar <= varOR
                if self.ifLog: self.myLogger.debug("%s created constrain: %s - %s <= %i"%(logicMethodName,currentVar.VarName,orVarName,0))
            else:
                pass # Only 0 possible now - has no effect on Or value

        varSumLinExpr = LinExpr()
        for currentVar in var:
            if not self.__varIsNumber(currentVar):
                varSumLinExpr.addTerms(1.0, currentVar)
            else:
                pass
            
        m.addConstr(varSumLinExpr - varOR >= 1-1) # varSumLinExpr >= varOR
        varSumLinExprStr = str(varSumLinExpr)
        if self.ifLog: self.myLogger.debug("%s created constrain: %s - %s >= %i"%(logicMethodName,varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('>')],orVarName,1-1))

        # Update Cache
        if N - noOfVars == 0: self.__addToConstrainCaches(methodName, onlyConstrains, var, varOR)
             
        if self.ifLog: self.myLogger.debug("%s returns new variable: %s"%(logicMethodName,varOR.VarName))
        return varOR
    
    def nand2Var(self, m, var1, var2, onlyConstrains = False):
        methodName = "nand2Var"
        logicMethodName = "NAND"
        
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
                if self.ifLog: self.myLogger.debug("%s constrain already created - doing nothing"%(logicMethodName))
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
                    m.addConstr(var2 <= 0)
                    if self.ifLog: self.myLogger.debug("%s created constrain only: %s <= %i"%(logicMethodName,var2Name,0))
                    return
                else:
                    self.myLogger.error("%s always True: %s is %i"%(logicMethodName,var1Name,var1))
                    return 1
            elif self.__varIsNumber(var2):
                if var2 == 1:
                    m.addConstr(var1 <= 0)
                    if self.ifLog: self.myLogger.debug("%s created constrain only: %s <= %i"%(logicMethodName,var1Name,0))
                    return
                else:
                    self.myLogger.error("%s always True: %s is %i"%(logicMethodName,var2Name,var2))
                    return 1
            else: # Both variables are ILP variables
                m.addConstr(var1 + var2 <= 1)
                if self.ifLog: self.myLogger.debug("%s created constrain only: and %s %s <= %i"%(logicMethodName,var1Name,var2Name,1))
    
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
            varNAND = m.addVar(vtype=GRB.BINARY, name="nand_%s_%s"%(var1, var2))
            if m: m.update()

            m.addConstr(self.notVar(m, varNAND) <= var1)
            if self.ifLog: self.myLogger.debug("%s created constrain: %s - %s <= 0 "%(logicMethodName,varNAND.VarName,var1Name))

            m.addConstr(self.notVar(m, varNAND) <= var2) # varAND <= var2
            if self.ifLog: self.myLogger.debug("%s created constrain: %s - %s <= 0 "%(logicMethodName,varNAND.VarName,var2Name))

            m.addConstr(var1 + var2 <= self.notVar(m, varNAND) + 2 - 1) # var1 + var2 <= varAND + 2 - 1
            if self.ifLog: self.myLogger.debug("%s created constrain: %s + %s - %s <= 1 "%(logicMethodName,var1Name,var2Name,varNAND.VarName))

            # Update cache
            self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), varNAND) 
            
            if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varNAND.VarName))
            
            return varNAND
    
    def nandVar(self, m, *var, onlyConstrains = False):
        methodName = "nandVar"
        logicMethodName = "NAND"
       
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
        
            m.addConstr(varSumLinExpr <= N - 1)
                        
            varSumLinExprStr = str(varSumLinExpr)
            if self.ifLog: self.myLogger.debug("NAND created constrain only: %s <= %i"%(varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('>')], N-1))
            
            self.__addToConstrainCaches('nandVar', onlyConstrains, var, None)
      
            return
        
        # ------- If creating variables representing value of OR build of provided variables
        nandVarName = "nand"
        for currentVar in var:
            nandVarName += "_%s"%(currentVar)
            
        varNAND = m.addVar(vtype=GRB.BINARY, name=nandVarName)
        for currentVar in var:
            m.addConstr(self.notVar(m, varNAND) <= currentVar)

        varSumLinExpr = LinExpr()
        for currentVar in var:
            varSumLinExpr.addTerms(1.0, currentVar)
    
        m.addConstr(varSumLinExpr <= self.notVar(m, varNAND) + N - 1)
    
        self.__addToConstrainCaches('nandVar', onlyConstrains, var, varNAND)

        return varNAND
    
    def nor2Var(self, m, var1, var2, onlyConstrains = False):
        if onlyConstrains:
            m.addConstr(var1 + var2 <= 0)
            return
        
        varNOR = m.addVar(vtype=GRB.BINARY, name="nor_%s_%s"%(var1, var2))
            
        m.addConstr(var1 <= self.notVar(m, varNOR))
        m.addConstr(var2 <= self.notVar(m, varNOR))
            
        m.addConstr(var1 + var2 >= self.notVar(m, varNOR))
        
        return varNOR
    
    def norVar(self, m, *var, onlyConstrains = False):
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

        if onlyConstrains:
            m.addConstr(var1 + var2 <= 1)
            m.addConstr(var1 + var2 >= 1)
            if self.ifLog: self.myLogger.debug("IF created constrain only: %s <= %s"%(var1.VarName, var2.VarName))

            self.__addToConstrainCaches('ifVar', onlyConstrains, (var1, var2), None)
            return
        
        varXOR = m.addVar(vtype=GRB.BINARY, name="xor_%s_%s"%(var1, var2))
            
        m.addConstr(var1 + var2 + varXOR <= 2)
        m.addConstr(-var1 - var2 + varXOR <= 0)
        m.addConstr(var1 - var2 + varXOR >= 0)
        m.addConstr(-var1 + var2 + varXOR >= 0)
            
        return varXOR
    
    def ifVar(self, m, var1, var2, onlyConstrains = False):
        methodName = "ifVar"
        logicMethodName = "IF"

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
                if self.ifLog: self.myLogger.debug("%s constrain already created"%(logicMethodName))
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
                m.addConstr(var1 <= var2)
                if self.ifLog: self.myLogger.debug("%s created constrain only: %s <= %s"%(logicMethodName,var1Name,var2Name))

                #self.__addToConstrainCaches(methodName, onlyConstrains, (var1, var2), None)
                return
    
        varIF = m.addVar(vtype=GRB.BINARY, name="if_%s_then_%s"%(var1.VarName, var2.VarName))
            
        m.addConstr(1 - var1 <= varIF)
        m.addConstr(var2 <= varIF)
        m.addConstr(1 - var1 + var2 >= varIF)
            
        m.update()

        self.__addToConstrainCaches('ifVar', onlyConstrains, (var1, var2), varIF)
        
        if self.ifLog: self.myLogger.debug("IF returns : %s"%(varIF.VarName))

        return varIF
           
    def epqVar(self, m, var1, var2, onlyConstrains = False):
        #if self.ifLog: self.myLogger.debug("EQ called with : %s"%(var1,var2))

        cacheResult = self.__isInConstrainCaches('eqVar', onlyConstrains, (var1, var2))
        if cacheResult[0]:
            if self.ifLog: self.myLogger.debug("EQ constrain already created - doing nothing")
            return cacheResult[1]
        
        if onlyConstrains:
            m.addConstr(var1 >= var2)
            if self.ifLog: self.myLogger.debug("EQ created constrain only: %s => %s"%(var1.VarName, var2.VarName))
            
            m.addConstr(var1 <= var2)
            if self.ifLog: self.myLogger.debug("EQ created constrain only: %s <= %s"%(var1.VarName, var2.VarName))

            self.__addToConstrainCaches('eqVar', onlyConstrains, (var1, var2), None)
            return
        
        varEQ = m.addVar(vtype=GRB.BINARY, name="epq_%s_%s"%(var1, var2))
            
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
        
        if not limitOp:
            if self.ifLog: self.myLogger.error("%s called with no operation specified for comparing limit"%(logicMethodName))
            return None

        if limitOp not in ('<', '>', '='):
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

            if limitOp == '>':
                if countOnes > limit:
                    if self.ifLog: self.myLogger.debug("%s created no constrain - the value of the method is True"%(logicMethodName))
                    return 1
                elif varSumLinExpr.size() - (limit - countOnes) < 0:
                    m.addConstr(1 <= 0)
                    if self.ifLog: self.myLogger.warning("%s created contradictory constrain 1 <= 0 - the value of the method is False"%(logicMethodName))
                    return 0
                else:
                    m.addConstr(varSumLinExpr >= limit - countOnes)
            if limitOp == '<':
                if varSumLinExpr.size() == 0:
                    if countOnes < limit:
                        if self.ifLog: self.myLogger.debug("%s created no constrain - the value of the method is True"%(logicMethodName))
                        return 1
                    else:
                        m.addConstr(1 <= 0)
                        if self.ifLog: self.myLogger.warning("%s created contradictory constrain 1 <= 0 - the value of the method is False"%(logicMethodName))
                        return 0
                else:
                    if limit < countOnes:
                        m.addConstr(1 <= 0)
                        if self.ifLog: self.myLogger.warning("%s created contradictory constrain 1 <= 0 - the value of the method is False"%(logicMethodName))
                        return 0
                    else:
                        m.addConstr(varSumLinExpr <= limit - countOnes)
            if limitOp == '=':
                if varSumLinExpr.size() == 0:
                    if countOnes == limit:
                        if self.ifLog: self.myLogger.debug("%s created no constrain - the value of the method is True"%(logicMethodName))
                        return 1
                    else:
                        m.addConstr(1 <= 0)
                        if self.ifLog: self.myLogger.warning("%s created contradictory constrain 1 <= 0 - the value of the method is False"%(logicMethodName))
                        return 0
                else:
                    m.addConstr(varSumLinExpr == limit - countOnes)
                 
            varSumLinExprStr = str(varSumLinExpr)
            if self.ifLog: self.myLogger.debug("%s created constrain only: %s %s= %i - %i"%(logicMethodName,varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('>')],limitOp,limit,countOnes))
            
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

        if limitOp == '>':
            if countOnes > limit:
                if self.ifLog: self.myLogger.debug("%s created no constrain - the value of the method is True"%(logicMethodName))
                return 1
            elif varSumLinExpr.size() - (limit - countOnes) < 0:
                m.addConstr(1 <= 0)
                if self.ifLog: self.myLogger.warning("%s created contradictory constrain 1 <= 0 - the value of the method is False"%(logicMethodName))
                return 0
            else:
                m.addConstr(varSumLinExpr - varCOUNT >= limit - 1 - countOnes)
        if limitOp == '<':
            if varSumLinExpr.size() == 0:
                if countOnes < limit:
                    if self.ifLog: self.myLogger.debug("%s created no constrain - the value of the method is True"%(logicMethodName))
                    return 1
                else:
                    m.addConstr(1 <= 0)
                    if self.ifLog: self.myLogger.warning("%s created contradictory constrain 1 <= 0 - the value of the method is False"%(logicMethodName))
                    return 0
            else:
                if limit < countOnes:
                    m.addConstr(1 <= 0)
                    if self.ifLog: self.myLogger.warning("%s created contradictory constrain 1 <= 0 - the value of the method is False"%(logicMethodName))
                    return 0
                else:
                    m.addConstr(varSumLinExpr - varCOUNT <= limit - 1 - countOnes)
        if limitOp == '=':
            if varSumLinExpr.size() == 0:
                if countOnes == limit:
                    if self.ifLog: self.myLogger.debug("%s created no constrain - the value of the method is True"%(logicMethodName))
                    return 1
                else:
                    m.addConstr(1 <= 0)
                    if self.ifLog: self.myLogger.warning("%s created contradictory constrain 1 <= 0 - the value of the method is False"%(logicMethodName))
                    return 0
            else:
                m.addConstr(varSumLinExpr == limit - countOnes)
                 
        m.addConstr(varSumLinExpr - varCOUNT >= limit-1)
        
        varSumLinExprStr = str(varSumLinExpr)
        if self.ifLog: self.myLogger.debug("%s created constrain: %s - varCOUNT %s= %i - 1 - %i"%(logicMethodName,varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('>')],limitOp,limit,countOnes))
             
        if self.ifLog: self.myLogger.debug("%s returns new variable: %s"%(logicMethodName,varCOUNT.VarName))
        return varCOUNT