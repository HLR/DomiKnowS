import logging
from itertools import permutations

from regr.solver.ilpBooleanMethods import ilpBooleanProcessor 
from regr.solver.ilpConfig import ilpConfig 

class lcLossBooleanMethods(ilpBooleanProcessor):
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
                
        self.myLogger = logging.getLogger(ilpConfig['log_name'])
        self.ifLog =  ilpConfig['ifLog']

    def notVar(self, m, var, onlyConstrains = False):
        methodName = "notVar"
        logicMethodName = "NOT"
                
        if self.ifLog: self.myLogger.debug("%s called with : %s"%(logicMethodName,var))

        notLoss = 1 - var   
        
        return notLoss
    
    def and2Var(self, m, var1, var2, onlyConstrains = False):
        methodName = "and2Var"
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
        
        and2Loss = 2 - var1 - var2
        
        return and2Loss
        
    def andVar(self, m, *var, onlyConstrains = False):
        methodName = "andVar"
        logicMethodName = "AND"
            
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
                
        N = len(var)
        
        varSum = 0
        for currentVar in var:
            varSum =+ currentVar
            
        andLoss = N - varSum
        
        return andLoss       
    
    def or2Var(self, m, var1, var2, onlyConstrains = False):
        methodName = "or2Var"
        logicMethodName = "OR"
       
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1,var2))

        or2Loss = 1 - var1 - var2
       
        return or2Loss
    
    def orVar(self, m, *var, onlyConstrains = False):
        methodName = "orVar"
        logicMethodName = "OR"
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
        
        varSum = 0
        for currentVar in var:
            varSum =+ currentVar
           
        orLoss = 1 - varSum
            
        return orLoss
        
    def nand2Var(self, m, var1, var2, onlyConstrains = False):
        methodName = "nand2Var"
        logicMethodName = "NAND"
        
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
        
        nand2Loss = var1 + var2 - 1
        
        return nand2Loss
    
    def nandVar(self, m, *var, onlyConstrains = False):
        methodName = "nandVar"
        logicMethodName = "NAND"
       
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, var))
        
        N = len(var)
        
        varSum = 0
        for currentVar in var:
            varSum =+ currentVar
        
        nandLoss = varSum - N + 1
                    
        return nandLoss
    
    def ifVar(self, m, var1, var2, onlyConstrains = False):
        methodName = "ifVar"
        logicMethodName = "IF"

        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1,var2))
     
        ifLoss = var1 - var2
        
        return ifLoss
               
    def norVar(self, m, *var, onlyConstrains = False):
        varSum = 0
        for currentVar in var:
            varSum =+ currentVar
    
        norLoss = varSum
        
        return norLoss
    
    # -------------------- Update

    def xorVar(self, m, var1, var2, onlyConstrains = False):
        
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
    
    def countVar(self, m, *var, onlyConstrains = False, limitOp = 'None', limit = 1):
        methodName = "countVar"
        logicMethodName = "COUNT"
        
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
        countVarName = ""
        for currentVar in var:
            countVarName = countVarName + "or"
            if self.__varIsNumber(currentVar):
                countVarName += "_%s_"%(currentVar)
            else:
                countVarName += "_%s_"%(currentVar.VarName)

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
        if self.ifLog: self.myLogger.debug("%s created constrain: %s - %s %s= %i - 1 - %i"%(logicMethodName,varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('>')],limitOp,countVarName,limit,countOnes))
             
        if self.ifLog: self.myLogger.debug("%s returns new variable: %s"%(logicMethodName,varCOUNT.VarName))
        return varCOUNT