import logging

from regr.solver.ilpBooleanMethods import ilpBooleanProcessor 
from regr.solver.ilpConfig import ilpConfig 

from gurobipy import Var, GRB, LinExpr

USE_De_Morgan = False

class gurobiILPBooleanProcessor(ilpBooleanProcessor):
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
                
        self.grad = False
        
        self.myLogger = logging.getLogger(ilpConfig['log_name'])
        self.ifLog =  ilpConfig['ifLog']
                
    def __varIsNumber(self, var):
        return not isinstance(var, Var)
    
    def preprocessLogicalMethodVar(self, var, logicMethodName, varNameConnector, onlyConstrains = False, minN = 2):
        # -- Check types of vars - gather information about them
        varsInfo = {}
        varsInfo['N'] = len(var) # Number of variables
        varsInfo['iLPVars'] = [] # ILP variables
        varsInfo['varsNames'] = [] # Names of vars
        varsInfo['varName'] = "" # Name of the new ILP variable if created
        varsInfo['numberMul'] = 1 # multiplication of numbers if present
        varsInfo['numberSum'] = 0 # summation of numbers if present
        varsInfo['varSumLinExpr'] = LinExpr()

        for currentVar in var:
            if self.__varIsNumber(currentVar):
                varsInfo['varsNames'].append(currentVar)
                varsInfo['numberMul'] *= currentVar
                varsInfo['numberSum'] += currentVar
            else:
                try:
                    varsInfo['varsNames'].append(currentVar.VarName)
                except AttributeError:
                    pass
                
                varsInfo['iLPVars'].append(currentVar)

                varsInfo['varSumLinExpr'].addTerms(1.0, currentVar)

                varsInfo['varName'] += varNameConnector
                varsInfo['varName'] += "_%s_" % (currentVar.VarName)
                
        varsInfo['varSumLinExprStr'] = str(varsInfo['varSumLinExpr'])
        varsInfo['varSumLinExprStr'] = varsInfo['varSumLinExprStr'][ varsInfo['varSumLinExprStr'].index(':') + 1 :  varsInfo['varSumLinExprStr'].index('>')]
                
        varsInfo['varName'] = varsInfo['varName'][:-1] # Remove last '_'
        varsInfo['varName'] = '{:.200}'.format(varsInfo['varName'])
        varsInfo['varName'] = varsInfo['varName'][:254] # Limit size of the new ILP variable name
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, varsInfo['varsNames']))
        
        if onlyConstrains:
            if varsInfo['N'] < minN: # Less than min variables
                if self.ifLog: self.myLogger.debug("%s has no enough variable - %i, returning without creating constraint"%(logicMethodName,varsInfo['N']))
                return
                
            if len(varsInfo['iLPVars']) == 0: # No ILP variables
                if self.ifLog: self.myLogger.debug("%s has no  ILP variable, returning without creating constraint"%(logicMethodName))
                return 
        else:      
            if varsInfo['N'] < minN: # Less than min variables
                if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,'None'))
                return None
            
        return varsInfo
    
    def notVar(self, m, var, onlyConstrains = False):
        logicMethodName = "NOT"
        
        # -- Consider None
        varFixed = []  
        if var is None:
            varFixed.append(0)
        else:
            varFixed.append(var)
        # --
        
        varsInfo = self.preprocessLogicalMethodVar(varFixed, logicMethodName, "not", onlyConstrains=onlyConstrains, minN=1)
        if varsInfo == None:
            return
        
        # If only constructing constrains forcing NOT to be true 
        if onlyConstrains:
            m.addConstr(varsInfo['iLPVars'][0] == 0, name='Not:')
            if self.ifLog: self.myLogger.debug("%s created constraint only: not %s == %i"%(logicMethodName,varsInfo['varsNames'][0],0))

            return
        
        else:
            # ------- If creating variables representing value of NOT build of provided variable
            if len(varsInfo['iLPVars']) == 0:
                if varFixed[0] == 0:
                    if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,1))
                    return 1
                else:
                    if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,0))
                    return 0
            else:
                varNOT = m.addVar(vtype=GRB.BINARY, name=varsInfo['varName'])
                if m: m.update()
        
                m.addConstr(varNOT + varsInfo['iLPVars'][0] == 1, name='Not:')
                if self.ifLog: self.myLogger.debug("%s created constraint: %s + %s == %i "%(logicMethodName,varNOT.VarName,varsInfo['varsNames'][0],1))
        
                if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varsInfo['varName']))
                return varNOT
            
    def and2Var(self, m, var1, var2, onlyConstrains = False):
        return self.andVar(m, (var1, var2), onlyConstrains=onlyConstrains)
    
    def andVar(self, m, *var, onlyConstrains = False):
        logicMethodName = "AND"
        
        # -- Consider None
        varFixed = []  
        for v in var:
            if v is None:
                varFixed.append(1)
            else:
                varFixed.append(v)
        # --
        
        varsInfo = self.preprocessLogicalMethodVar(varFixed, logicMethodName, "and", onlyConstrains=onlyConstrains)
        if varsInfo == None:
            return
        
        # -- If only constructing constrains forcing AND to be true 
        if onlyConstrains:    
            if varsInfo['numberMul'] == 0: # Vars numbers multiply to 0 - at least one zero present
                if self.ifLog: self.myLogger.debug("%s has zero, returning without creating constraint"%(logicMethodName))
                return
            
            if len(varsInfo['iLPVars']) == 1: # if only one ILP variable
                if self.ifLog: self.myLogger.debug("%s has no enough ILP variable - %i, returning without creating constraint"%(logicMethodName,len(varsInfo['iLPVars'])))
                return
            
            # Create constraint as there are at least two ILP variables and all numbers, if present, are 1
            if self.ifLog: self.myLogger.debug("%s created constraint only: and %s > 1"%(logicMethodName,varsInfo['varSumLinExprStr']))

            m.addConstr(len(varsInfo['iLPVars']) - varsInfo['varSumLinExpr'] <= 0, name='And:') # varSumLinExpr >= N
            return
        else:  
            # -- If creating variables representing value of AND build of provided variables
            
            if varsInfo['numberMul'] == 0: # Vars numbers multiply to 0 - at least one zero present
                if self.ifLog: self.myLogger.debug("%s has zero, returning 0 without creating additional constraint"%(logicMethodName))
                return 0
            
            if len(varsInfo['iLPVars']) == 0: # No ILP variables
                if self.ifLog: self.myLogger.debug("%s has no ILP variable, returning %i without creating additional constraint"%(logicMethodName, varsInfo['numberMul']))
                return varsInfo['numberMul']
            
            if len(varsInfo['iLPVars']) == 1: # Only single ILP variable; rest has to be ones here
                if self.ifLog: self.myLogger.debug("%s has ones and only single variable: %s, it is returned"%(logicMethodName,varsInfo['iLPVars'][0]))
                return varsInfo['iLPVars'][0]
                    
            # -- More than one ILP variable and rest is ones 
            
            # Create new variable
            varAND = m.addVar(vtype=GRB.BINARY, name=varsInfo['varName'])
            if m: m.update()
    
            # Build constraints 
            for currentVar in varsInfo['iLPVars']:
                m.addConstr(varAND - currentVar <= 0, name='And:') # varAND <= currentVar
            
            m.addConstr(varsInfo['varSumLinExpr'] - varAND <= len(varsInfo['iLPVars']) - 1, name='And:') #  varSumLinExpr <= varAND + N - 1

            if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varAND.VarName))
            return varAND
    
    def or2Var(self, m, var1, var2, onlyConstrains = False):
        return self.orVar(m, (var1, var2), onlyConstrains = onlyConstrains)
    
    def orVar(self, m, *var, onlyConstrains = False):
        if USE_De_Morgan:
            notVar = []
            for v in var:
                notVar.append(self.notVar(m, v))
            
            return self.notVar(m, self.andVar(m, *notVar), onlyConstrains=onlyConstrains)
        
        logicMethodName = "OR"
        
        # -- Consider None
        varFixed = []  
        for v in var:
            if v is None:
                varFixed.append(0)
            else:
                varFixed.append(v)
        # --
        
        varsInfo = self.preprocessLogicalMethodVar(varFixed, logicMethodName, "or", onlyConstrains=onlyConstrains)
        if varsInfo == None:
            return
        
        # If only constructing constrains forcing OR to be true 
        if onlyConstrains:
            if varsInfo['numberSum'] > 0: # Vars numbers sum non 0 - at least one present
                if self.ifLog: self.myLogger.debug("%s has ones, returning without creating constraint"%(logicMethodName))
                return
            
            if len(varsInfo['iLPVars']) == 1: # if only one ILP variable
                if self.ifLog: self.myLogger.debug("%s has no enough ILP variable - %i, returning without creating constraint"%(logicMethodName,len(varsInfo['iLPVars'])))
                return
            
            # Create constraint as there are at least two ILP variables and all numbers, if present, are 1            
            m.addConstr(varsInfo['varSumLinExpr'] >= 1, name='Or:')
            
            if self.ifLog: self.myLogger.debug("%s created constraint only: %s >= %i"%(logicMethodName,varsInfo['varSumLinExprStr'],1))
            
            return
        
        else:
            # ------- If creating variables representing value of OR build of provided variables
            if varsInfo['numberSum'] > 0: # Vars numbers sum non 0 - at least one present
                if self.ifLog: self.myLogger.debug("%s has ones, returning 1 without creating additional constraint"%(logicMethodName))
                return 1
            
            if len(varsInfo['iLPVars']) == 0: # No ILP variables
                if self.ifLog: self.myLogger.debug("%s has no ILP variable, returning %i without creating additional constraint"%(logicMethodName, varsInfo['numberSum']))
                return varsInfo['numberSum']
            
            if len(varsInfo['iLPVars']) == 1: # Only single ILP variable; rest has to be zeros: see first if above
                if self.ifLog: self.myLogger.debug("%s has zeros and only single variable: %s, it is returned"%(logicMethodName,varsInfo['iLPVars'][0]))
                return varsInfo['iLPVars'][0]
            
            # Create new variable
            varOR = m.addVar(vtype=GRB.BINARY, name=varsInfo['varName'])
            if m: m.update()
    
            # Build constrains
            for currentVar in varsInfo['iLPVars']:
                m.addConstr(currentVar - varOR <= 0, name='Or:') # currentVar <= varOR
                if self.ifLog: self.myLogger.debug("%s created constraint: %s - %s <= %i"%(logicMethodName,currentVar.VarName,varsInfo['varName'],0))
    
            m.addConstr(varsInfo['varSumLinExpr'] - varOR >= 0, name='Or:') # varSumLinExpr >= varOR
            if self.ifLog: self.myLogger.debug("%s created constraint: %s - %s >= %i"%(logicMethodName,varsInfo['varSumLinExprStr'],1-1))
    
            if self.ifLog: self.myLogger.debug("%s returns new variable: %s"%(logicMethodName,varsInfo['varName']))
            return varOR
        
    def nand2Var(self, m, var1, var2, onlyConstrains = False):
        logicMethodName = "NAND"
        
        # -- Consider None
        if var1 is None:
            var1 = 0
            
        if var2 is None:
            var2 = 0
        # --
        
        # Get names of ILP variables
        var1Name = var1
        var2Name = var2
        if not self.__varIsNumber(var1):
            var1Name = var1.VarName
        if not self.__varIsNumber(var2):
            var2Name = var2.VarName
            
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1Name,var2Name))
        
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
            
            if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varNAND.VarName))
            
            return varNAND
    
    def nandVar(self, m, *var, onlyConstrains = False):
        logicMethodName = "NAND"
       
        # -- Consider None
        varFixed = []  
        for v in var:
            if v is None:
                varFixed.append(0)
            else:
                varFixed.append(v)
        
        var = varFixed
        # --
       
        # Get names of variables - some of them can be numbers
        noOfILPVars = 0 # count the numbers in variables
        varsNames = []
        for currentVar in var:
            if self.__varIsNumber(currentVar):
                varsNames.append(currentVar)
            else:
                varsNames.append(currentVar)
                noOfILPVars = noOfILPVars + 1
            
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, varsNames))
        
        # Check number of variables
        N = len(var)
        
        if N <= 1:
            if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,'None'))
            return None
        
        if N == 2: return self.nand2Var(m, var[0], var[1], onlyConstrains)
        
        # If only constructing constrains forcing NAND to be true 
        if onlyConstrains:
            varSumLinExpr = LinExpr()
            for currentVar in var:
                varSumLinExpr.addTerms(1.0, currentVar)
        
            m.addConstr(varSumLinExpr <= N - 1, name='Nand:')
                        
            varSumLinExprStr = str(varSumLinExpr)
            if self.ifLog: self.myLogger.debug("NAND created constraint only: %s <= %i"
                                               %(varSumLinExprStr[varSumLinExprStr.index(':') + 1 : varSumLinExprStr.index('>')], N-1))
                  
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
    
        return varNAND
    
    def nor2Var(self, m, var1, var2, onlyConstrains = False):
        
        # -- Consider None
        if var1 is None:
            var1 = 1
            
        if var2 is None:
            var2 = 1
        # --
        
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
        
        # -- Consider None
        varFixed = []  
        for v in var:
            if v is None:
                varFixed.append(1)
            else:
                varFixed.append(v)
        
        var = varFixed
        # --
        
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
        
        # -- Consider None
        if var1 is None:
            var1 = 1
            
        if var2 is None:
            var2 = 1
        # --

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
        logicMethodName = "IF"

        # -- Consider None
        if var1 is None:
            var1 = 1
            
        if var2 is None:
            var2 = 0
        # --
    
        # Get names of ILP variables
        var1Name = var1
        var2Name = var2
        if not self.__varIsNumber(var1):
            var1Name = var1.VarName
        if not self.__varIsNumber(var2):
            var2Name = var2.VarName
            
        if self.ifLog: self.myLogger.debug("%s called with : var1 - %s, var2 - %s"%(logicMethodName,var1Name,var2Name))
    
        # When some of vars are numbers
        if self.__varIsNumber(var1) and self.__varIsNumber(var2):
            if var1 == 0:
                if self.ifLog: self.myLogger.debug("%s is True returning %i"%(logicMethodName,1))
                return 1
            elif var1 == 1 and var2 == 1:
                if self.ifLog: self.myLogger.debug("%s is True returning %i"%(logicMethodName,1))
                return 1
            else:
                self.ifLog: self.myLogger.debug("%s is False returning %i"%(logicMethodName,0))
                return 0
        elif self.__varIsNumber(var1):
            if var1 == 0:
                if self.ifLog: self.myLogger.debug("%s is True returning %i"%(logicMethodName,1))
                return 1
            else:
                if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,var2Name))
                if onlyConstrains:
                    m.addConstr(var2 >= 1, name='If:')
                    return
                else:
                    return var2
        elif  self.__varIsNumber(var2):
            if var2 == 1:
                if self.ifLog: self.myLogger.debug("%s is True returning %i"%(logicMethodName,1))
                return 1
            else:
                notVar1 = self.notVar(m, var1)
                if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,notVar1.VarName))
                if onlyConstrains:
                    m.addConstr(notVar1 >= 1, name='If:')
                    return
                else:
                    return notVar1
            
        # If only constructing constrains forcing OR to be true 
        if onlyConstrains:
            m.addConstr(var1 - var2 <= 0, name='If:') #var1 <= var1
            if self.ifLog: self.myLogger.debug("%s created constraint only: %s <= %s"%(logicMethodName,var1Name,var2Name))
            
            return
        else:
            varIFName = "if_%s_then_%s"%(var1Name, var2Name)
            varIFName = varIFName[:254]
            varIF = m.addVar(vtype=GRB.BINARY, name=varIFName)
                
            m.addConstr(1 - var1 <= varIF, name='If:')
            m.addConstr(var2 <= varIF, name='If:')
            m.addConstr(1 - var1 + var2 >= varIF, name='If:')
                
            m.update()
            
            if self.ifLog: self.myLogger.debug("IF returns : %s"%(varIF.VarName))
    
            return varIF
           
    def epqVar(self, m, var1, var2, onlyConstrains = False):
        #if self.ifLog: self.myLogger.debug("EQ called with : %s"%(var1,var2))

        # -- Consider None
        if var1 is None:
            var1 = 0
            
        if var2 is None:
            var2 = 0
        # --
    
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

            return
        
        varEQName = "epq_%s_%s"%(var1, var2)
        varEQName = varEQName[:254]
        varEQ = m.addVar(vtype=GRB.BINARY, name=varEQName)
            
        m.addConstr(var1 + var2 - varEQ <= 1)
        m.addConstr(var1 + var2 + varEQ >= 1)
        m.addConstr(-var1 + var2 + varEQ <= 1)
        m.addConstr(var1 - var2 + varEQ <= 1)
        
        m.update()
             
        if self.ifLog: self.myLogger.debug("EQ returns : %s"%(varEQ.VarName))
        return varEQ
    
    def countVar(self, m, *var, onlyConstrains = False, limitOp = 'None', limit = 1, logicMethodName = "COUNT"):
        BigM = 100
        
        if not limitOp:
            if self.ifLog: self.myLogger.error("%s called with no operation specified for comparing limit"%(logicMethodName))
            return None

        if limitOp not in ('<=', '>=', '=='):
            if self.ifLog: self.myLogger.error("%s called with incorrect operation specified for comparing limit %s"%(logicMethodName,limitOp))
            return None
            
        if self.ifLog: self.myLogger.debug("%s called with limit: %i and operation %s"%(logicMethodName,limit,limitOp))
        
        # -- Consider None
        varFixed = []  
        for v in var:
            if v is None:
                varFixed.append(0)
            else:
                varFixed.append(v)
        
        var = varFixed
        # --
        
        varsInfo = self.preprocessLogicalMethodVar(varFixed, logicMethodName, logicMethodName, onlyConstrains=onlyConstrains)
        if varsInfo == None:
            return
            
        updatedLimit = limit - varsInfo['numberSum']
        
        # If only constructing constrains forcing OR to be true 
        if onlyConstrains:
            if updatedLimit < 0: updatedLimit = 0

            if limitOp == '>=':
                m.addConstr(varsInfo['varSumLinExpr'] >= updatedLimit, name='Count %s:'%(logicMethodName))
            if limitOp == '<=':
                m.addConstr(varsInfo['varSumLinExpr'] <= updatedLimit, name='Count %s:'%(logicMethodName))
            if limitOp == '==':
                m.addConstr(varsInfo['varSumLinExpr'] == updatedLimit, name='Count %s:'%(logicMethodName))
            
            return
        
        # ------- If creating variables representing value of OR build of provided variables
        else:
        # Build constrains
            if limitOp == '>=':
                if updatedLimit < 0:
                    if self.ifLog: self.myLogger.debug("%s created no constraint - returns True"%(logicMethodName))
                    return 1
                elif len(varsInfo['iLPVars']) < updatedLimit:
                    if self.ifLog: self.myLogger.debug("%s creates contradictory constraint - returns False"%(logicMethodName))
                    return 0
                elif len(varsInfo['iLPVars']) == 0:
                    result = int(updatedLimit < 0)
                    if self.ifLog: self.myLogger.debug("%s created no constraint - returns %s"%(logicMethodName, result))
                    return result
                else:
                    # Create new variable
                    varCOUNT = m.addVar(vtype=GRB.BINARY, name=varsInfo['varName'])
                    if m: m.update()
                    
                    m.addConstr(varsInfo['varSumLinExpr'] - BigM *varCOUNT >= updatedLimit - BigM, name='Count %s:'%(logicMethodName))
                    m.addConstr(varsInfo['varSumLinExpr'] - BigM *varCOUNT <= updatedLimit, name='Count %s:'%(logicMethodName))
                    
                    #m.addConstr(varSumLinExpr + BigM *varCOUNT <= updatedLimit + BigM, name='Count %s:'%(logicMethodName))
                    #m.addConstr(varSumLinExpr + BigM *varCOUNT >= updatedLimit + 1, name='Count %s:'%(logicMethodName))
                    #m.addConstr(varSumLinExpr - varCOUNT <= updatedLimit - 1, name='Count %s:'%(logicMethodName))
            if limitOp == '<=':
                if len(varsInfo['iLPVars']) == 0:
                    result = int(updatedLimit > 0)
                    if self.ifLog: self.myLogger.debug("%s created no constraint - returns %s"%(logicMethodName, result))
                    return result  
                else:  
                    # Create new variable
                    varCOUNT = m.addVar(vtype=GRB.BINARY, name=varsInfo['varName'])
                    if m: m.update()
                    
                    m.addConstr(varsInfo['varSumLinExpr'] + BigM *varCOUNT <= updatedLimit + BigM, name='Count %s:'%(logicMethodName))
                    m.addConstr(varsInfo['varSumLinExpr'] + BigM *varCOUNT >= updatedLimit, name='Count %s:'%(logicMethodName))
            if limitOp == '==':
                if len(varsInfo['iLPVars']) == 0:
                    result = int(updatedLimit == 0)
                    if self.ifLog: self.myLogger.debug("%s created no constraint - returns %s"%(logicMethodName, result))
                    return result  
                else:
                    # Create new variable
                    varCOUNT = m.addVar(vtype=GRB.BINARY, name=varsInfo['varName'])
                    if m: m.update()
            
                    m.addConstr(varsInfo['varSumLinExpr'] - updatedLimit <= BigM * (1 - varCOUNT), name='Count %s:'%(logicMethodName))
                    m.addConstr(varsInfo['varSumLinExpr'] - updatedLimit >= BigM * (varCOUNT - 1), name='Count %s:'%(logicMethodName))
    
            if self.ifLog: self.myLogger.debug("%s returns new variable: %s"%(logicMethodName,varsInfo['varName']))
            return varCOUNT
    
    def fixedVar(self, m, var, onlyConstrains = False): 
        logicMethodName = "FIXED"
        
        # -- Consider None
        if var is None: # not create Fixed constraint for None
            return None
        # --
                
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