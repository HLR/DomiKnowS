import logging

from domiknows.solver.ilpBooleanMethods import ilpBooleanProcessor 
from domiknows.solver.ilpConfig import ilpConfig 

from gurobipy import Var, GRB, LinExpr

USE_De_Morgan = False # For orVar nandVar methods 

# Creates ILP constraints for logical methods based on provided arguments.
#
# The method's arguments originate either directly from the candidates (dataNode) 
#                                  or are the result of the nested logical constraints evaluation.
#
# The method's arguments can be:
#   - ILP variable, 
#   - number (0 or 1) representing True or False value,
#   - None representing lack of information (when the candidate is missing in the dataNode).

class gurobiILPBooleanProcessor(ilpBooleanProcessor):
    
    def __init__(self, _ildConfig = ilpConfig) -> None:
        super().__init__()
                
        self.grad = False
        
        self.myLogger = logging.getLogger(ilpConfig['log_name'])
        self.ifLog =  ilpConfig['ifLog']
                
    def __varIsNumber(self, var):
        return not isinstance(var, Var)
    
    def preprocessLogicalMethodVar(self, var, logicMethodName, varNameConnector, minN = 2):
        # -- Check types of variables in var - gather information about them
        varsInfo = {}
        varsInfo['N'] = len(var)                # Number of all variables
        varsInfo['iLPVars'] = []                # ILP variables
        varsInfo['No_of_ilp'] = 0               # Number of IPL variables
        varsInfo['varsNames'] = []              # Names of variables
        varsInfo['varName'] = ""                # Name of the new ILP variable if created
        varsInfo['numberMul'] = 1               # Multiplication of all numbers if present
        varsInfo['numberSum'] = 0               # Summation of all numbers if present
        varsInfo['varSumLinExpr'] = LinExpr()   # Summation of all ILP variable
        varsInfo['varSumLinExprStr'] = ""       # String representation of the summation of all ILP variable
        
        for i, currentVar in enumerate(var):
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
                varsInfo['No_of_ilp'] += 1

                varsInfo['varSumLinExpr'].addTerms(1.0, currentVar)

                varsInfo['varName'] += varNameConnector
                varsInfo['varName'] += "_%s_" % (currentVar.VarName)
    
        if varsInfo['varSumLinExpr'].size() > 0:
            varsInfo['varSumLinExprStr'] = str(varsInfo['varSumLinExpr']) 
            #varsInfo['varSumLinExprStr'] = varsInfo['varSumLinExprStr'][varsInfo['varSumLinExprStr'].index(':') + 1 : varsInfo['varSumLinExprStr'].index('>')]
          
        if len(varsInfo['varName']) > 0:   
            varsInfo['varName'] = varsInfo['varName'][:-1] # Remove last '_'
            varsInfo['varName'] = '{:.200}'.format(varsInfo['varName'])
            varsInfo['varName'] = varsInfo['varName'][:254] # Limit size of the new ILP variable name
        
        if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, varsInfo['varsNames']))
        
        if varsInfo['N'] < minN: # Variables number less than min 
            raise Exception("%s has no enough variable - %i, required %i"%(logicMethodName,varsInfo['N'],minN))
            
        return varsInfo
    
    def notVar(self, m, var, onlyConstrains = False):
        logicMethodName = "NOT"
        
        # -- Consider None
        varFixed = []  
        if var is None:
            varFixed.append(0) # when None
        else:
            varFixed.append(var)
        # --
        
        varsInfo = self.preprocessLogicalMethodVar(varFixed, logicMethodName, "not", minN=1)
        
        if varsInfo['N'] > 1: # More than 1 variable
            raise Exception("%s has %i variables, accepts only 1"%(logicMethodName,varsInfo['N']))
        
        # -- Only constructing constrains forcing NOT to be True 
        if onlyConstrains:
            if varsInfo['No_of_ilp'] == 0: # Called with a number
                if varsInfo['numberSum'] == 0: # number is 0
                    # Applying not results in True
                    return 
                else: # number is 1
                    # Applying not results in False -> model is infeasible -> exception
                    raise Exception("ILP model is infeasible - %s is called with value %i, and the result of applying %s is False"%(logicMethodName,1,logicMethodName))
            else:
                # -- Create constraint as there is an ILP variable
                m.addConstr(varsInfo['iLPVars'][0] == 0, name='Not:') # ILP variable has to be 0 so applying not will result in True
                if self.ifLog: self.myLogger.debug("%s created constraint only: not %s == %i"%(logicMethodName,varsInfo['varsNames'][0],0))
    
                return
        else:
            # --- Creating ILP variable representing value of NOT build of provided method argument
            
            if varsInfo['No_of_ilp'] == 0: # Called with a number
                if varsInfo['numberSum'] == 0:
                    # Applying not results in True
                    if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,1))
                    return 1
                else:
                    # Applying not results in False
                    if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,0))
                    return 0
            else: 
                # -- One ILP variable
                
                # Create new ILP variable 
                varNOT = m.addVar(vtype=GRB.BINARY, name=varsInfo['varName'])
                if m: m.update()
        
                # Adding ILP constraint
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
                varFixed.append(1) # when None
            else:
                varFixed.append(v)
        # --
        
        varsInfo = self.preprocessLogicalMethodVar(varFixed, logicMethodName, "and")
       
        # -- Only constructing constrains forcing AND to be True 
        if onlyConstrains:    
            if varsInfo['numberMul'] == 0: # Vars numbers multiply to 0 - at least one zero present
                # Applying and results in False -> model is infeasible -> exception
                raise Exception("ILP model is infeasible - %s is called with value %i, and the result of applying %s is False"%(logicMethodName,0,logicMethodName))
            elif varsInfo['No_of_ilp'] == 0: # No ILP variables
                # Applying and to all 1 (multiply is 1) results in True
                if self.ifLog: self.myLogger.debug("%s returns: %i"%(logicMethodName,1))
                return
            elif varsInfo['No_of_ilp'] == 1: # if only one ILP variable - rest have to be 1 as there were not zeros
                # Adding ILP constraint
                m.addConstr(varsInfo['iLPVars'][0] == 1, name='Not:') # ILP variable has to be 1 so applying and will result in True
                return
            else:
                # -- Create constraint as there are at least two ILP variables and all numbers, if present are 1
                m.addConstr(varsInfo['No_of_ilp'] - varsInfo['varSumLinExpr'] <= 0, name='And:') #  varSumLinExpr >= N
                if self.ifLog: self.myLogger.debug("%s created constraint only: and %s > %i"%(logicMethodName,varsInfo['varSumLinExprStr'],varsInfo['No_of_ilp']))
                return
        else:  
            # -- If creating ILP variable representing value of AND build of provided method arguments
            
            if varsInfo['numberMul'] == 0: # Vars numbers multiply to 0 - at least one 0 present
                # Applying and results in False
                if self.ifLog: self.myLogger.debug("%s has zero, returning 0 without creating additional constraint"%(logicMethodName))
                return 0
            elif varsInfo['No_of_ilp'] == 0: # No ILP variables
                # Applying and results in True
                if self.ifLog: self.myLogger.debug("%s has no ILP variable, returning %i without creating additional constraint"%(logicMethodName, varsInfo['numberMul']))
                return 1
            elif varsInfo['No_of_ilp'] == 1: # Only single ILP variable; rest is 1 here
                # Result of and is the value of the single ILP variable
                if self.ifLog: self.myLogger.debug("%s has ones and only single variable: %s, it is returned"%(logicMethodName,varsInfo['iLPVars'][0]))
                return varsInfo['iLPVars'][0]
            else:      
                # -- More than one ILP variable and the rest is 1 
                
                # Create new variable
                varAND = m.addVar(vtype=GRB.BINARY, name=varsInfo['varName'])
                if m: m.update()
        
                # Build ILP constraints 
                for currentVar in varsInfo['iLPVars']:
                    m.addConstr(varAND - currentVar <= 0, name='And:') # varAND <= currentVar
                
                # Adding ILP constraint
                m.addConstr(varsInfo['varSumLinExpr'] - varAND <= varsInfo['No_of_ilp'] - 1, name='And:') #  varSumLinExpr <= varAND + N - 1
    
                if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varsInfo['varName']))
                return varAND
    
    def or2Var(self, m, var1, var2, onlyConstrains = False):
        return self.orVar(m, (var1, var2), onlyConstrains = onlyConstrains)
    
    def orVar(self, m, *var, onlyConstrains = False):
        if USE_De_Morgan:
            notVar = []
            for v in var:
                notVar.append(self.notVar(m, v))
            
            return self.notVar(m, self.andVar(m, *notVar), onlyConstrains=onlyConstrains) #  Negation of the conjunction of the negations
        
        logicMethodName = "OR"
        
        # -- Consider None
        varFixed = []  
        for v in var:
            if v is None:
                varFixed.append(0) # when None
            else:
                varFixed.append(v)
        # --
        
        varsInfo = self.preprocessLogicalMethodVar(varFixed, logicMethodName, "or")
        
        # If only constructing constrains forcing OR to be True 
        if onlyConstrains:
            if varsInfo['numberSum'] > 0: # Vars numbers sum is larger then 0 - at least one is present
                # Applying or results in True
                if self.ifLog: self.myLogger.debug("%s has ones, returning without creating constraint"%(logicMethodName))
                return
            elif varsInfo['No_of_ilp'] == 0: # No ILP variables
                # Applying or results in False -> model is infeasible -> exception
                raise Exception("ILP model is infeasible - %s is called with values %i, and the result of applying %s is False"%(logicMethodName,0,logicMethodName))
            elif varsInfo['No_of_ilp'] == 1: # Only one ILP variable and the rest are zeros
                # Adding ILP constraint
                m.addConstr(varsInfo['iLPVars'][0] >= 1, name='Or:') # ILP variable has to be 1 so applying or will result in True
                if self.ifLog: self.myLogger.debug("%s created constraint only: %s >= %i"%(logicMethodName,varsInfo['iLPVars'][0],1))
                return
            else:
                # -- Create constraint as there are at least two ILP variables and all numbers, if present, are 0         
                m.addConstr(varsInfo['varSumLinExpr'] >= 1, name='Or:') # varSumLinExpr >= 1
                if self.ifLog: self.myLogger.debug("%s created constraint only: %s >= %i"%(logicMethodName,varsInfo['varSumLinExprStr'],1))
                return
        else:
            # ------- Creating ILP variable representing value of OR build of provided method arguments
            
            if varsInfo['numberSum'] > 0: #  Vars numbers sum is larger then 0 - at least one present
                # Applying or results in True
                if self.ifLog: self.myLogger.debug("%s has ones, returning 1 without creating additional constraint"%(logicMethodName))
                return 1
            elif varsInfo['No_of_ilp'] == 0: # No ILP variables
                if self.ifLog: self.myLogger.debug("%s has no ILP variable, returning %i without creating additional constraint"%(logicMethodName, varsInfo['numberSum']))
                return varsInfo['numberSum']
            elif varsInfo['No_of_ilp'] == 1: # Only single ILP variable; rest has to be zeros: see first if above
                if self.ifLog: self.myLogger.debug("%s has zeros and only single variable: %s, it is returned"%(logicMethodName,varsInfo['iLPVars'][0]))
                return varsInfo['iLPVars'][0]
            else:
                # -- More than one ILP variable and the rest is 0 
                
                # Create new variable
                varOR = m.addVar(vtype=GRB.BINARY, name=varsInfo['varName'])
                if m: m.update()
        
                # Build constrains
                for currentVar in varsInfo['iLPVars']:
                    m.addConstr(currentVar - varOR <= 0, name='Or:') # currentVar <= varOR
                    if self.ifLog: self.myLogger.debug("%s created constraint: %s - %s <= %i"%(logicMethodName,currentVar.VarName,varsInfo['varName'],0))
        
                m.addConstr(varsInfo['varSumLinExpr'] - varOR >= 0, name='Or:') # varSumLinExpr >= varOR
                if self.ifLog: self.myLogger.debug("%s created constraint: %s - %s >= %i"%(logicMethodName,varsInfo['varSumLinExprStr'],varsInfo['varName'],1-1))
        
                if self.ifLog: self.myLogger.debug("%s returns new variable: %s"%(logicMethodName,varsInfo['varName']))
                return varOR
            
    def nand2Var(self, m, var1, var2, onlyConstrains = False):
        return self.nandVar(m, (var1, var2), onlyConstrains = onlyConstrains)
    
    def nandVar(self, m, *var, onlyConstrains = False):
        logicMethodName = "NAND"
       
        if USE_De_Morgan:
            notVar = []
            for v in var:
                notVar.append(self.notVar(m, v))
            
            return self.andVar(m, *notVar, onlyConstrains=onlyConstrains) # Negation of their conjunction 
        
        # -- Consider None
        varFixed = []  
        for v in var:
            if v is None:
                varFixed.append(0) # when None
            else:
                varFixed.append(v)
        
        var = varFixed
        # --
        
        varsInfo = self.preprocessLogicalMethodVar(varFixed, logicMethodName, "or")
                
        # If only constructing constrains forcing NAND to be true 
        if onlyConstrains:
            varSumLinExpr = LinExpr()
            for currentVar in var:
                varSumLinExpr.addTerms(1.0, currentVar)
        
            m.addConstr(varSumLinExpr <= varsInfo["N"] - 1, name='Nand:')     
            if self.ifLog: self.myLogger.debug("NAND created constraint only: %s <= %i"%(varsInfo["varSumLinExprStr"], varsInfo["N"]-1))
                  
            return
        
        # ------- If creating variables representing value of NAND build of provided variables
        varNAND = m.addVar(vtype=GRB.BINARY, name=varsInfo['varName'])
        for currentVar in var:
            m.addConstr(self.notVar(m, varNAND) <= currentVar, name='Nand:')

        m.addConstr(varsInfo["varSumLinExpr"]<= self.notVar(m, varNAND) + varsInfo["N"] - 1, name='Nand:')
    
        return varNAND
    
    def nor2Var(self, m, var1, var2, onlyConstrains = False):
        return self.norVar(m, (var1, var2), onlyConstrains = onlyConstrains)
    
    def norVar(self, m, *var, onlyConstrains = False):
        
        return self.notVar(m, self.orVar(m, var), onlyConstrains=onlyConstrains) # Negation of the disjunction
    
        #---------------------- No used anymore
        
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
        var = (var1, var2)
        
        # Conjunction of the disjunction and the negation of the conjunction
        return self.andVar(m, self.orVar(m, var), self.notVar(m, self.andVar(m, var)), onlyConstrains=onlyConstrains) 
                
        #---------------------- No used anymore

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
        hasNone = False
        if var1 is None: # antecedent 
            antecedent = 1 # when None
            self.myLogger.info("%s called with antecedent equals None"%(logicMethodName))
            self.myLogger.info("%s called with consequent equals %s"%(logicMethodName,var2))

            hasNone = True
        else:
            antecedent = var1

        if var2 is None: # consequent
            consequent = 0 # when None
            if not hasNone: # not yet
                self.myLogger.info("%s called with antecedent equals %s"%(logicMethodName,var1))
                self.myLogger.info("%s called with consequent equals None"%(logicMethodName))
                hasNone = True
        else:
            consequent = var2
        # --
    
        varsInfo = self.preprocessLogicalMethodVar((antecedent,consequent), logicMethodName, "if",  minN=2)
        
        if varsInfo['N'] > 2: # More than 2 variable
            raise Exception("%s has %i variables, accepts only 2"%(logicMethodName,varsInfo['N']))
            
        if onlyConstrains:
            if varsInfo['No_of_ilp'] == 0: # No ILP variables
                if (not antecedent): 
                    # Applying if results in True
                    if self.ifLog: self.myLogger.debug("%s is True - antecedent is False"%(logicMethodName))
                    if hasNone: self.myLogger.info("%s is True - antecedent is False"%(logicMethodName))
                    return 
                elif consequent: # antecedent is True
                    # Applying if results in True
                    if self.ifLog: self.myLogger.debug("%s is True - antecedent and consequent are True"%(logicMethodName))
                    if hasNone: self.myLogger.info("%s is True - antecedent and consequent are True"%(logicMethodName))
                    return 
                else: # antecedent and not consequent
                    raise Exception("ILP model is infeasible - %s is called with the antecedent True and the consequent False - the result of applying %s is False"
                                    %(logicMethodName,logicMethodName))
            elif self.__varIsNumber(antecedent): # antecedent is boolean and consequent is the ILP variable
                if not antecedent:
                    # Applying if results in True
                    if self.ifLog: self.myLogger.debug("%s is True - antecedent is False"%(logicMethodName))
                    if hasNone: self.myLogger.info("%s is True - antecedent is False"%(logicMethodName))
                    return 
                else: # antecedent is True
                    # Adding ILP constraint
                    m.addConstr(consequent >= 1, name='If:') # var2 >= 0 - consequent has to be True
                    if self.ifLog: self.myLogger.debug("%s antecedent is True - created constraint: %s >= %i"%(logicMethodName,consequent,1))
                    if hasNone: self.myLogger.info("%s antecedent is True - created constraint: %s >= %i"%(logicMethodName,consequent,1))
                    return
            elif  self.__varIsNumber(consequent): # consequent is boolean and the antecedent is the ILP variable
                if consequent:
                    # Applying if results in True
                    if self.ifLog: self.myLogger.debug("%s is True - consequent is True"%(logicMethodName))
                    if hasNone: self.myLogger.info("%s is True - consequent is True"%(logicMethodName))
                    return 
                else: # consequent is False
                    # Adding ILP constraint - antecedent ILP variable has to be False too
                    m.addConstr(antecedent <= 0, name='If:')
                    if self.ifLog: self.myLogger.debug("%s consequent is False - created constraint: %s <= %i"%(logicMethodName,antecedent,0))
                    if hasNone:  self.myLogger.info("%s consequent is False - created constraint: %s <= %i"%(logicMethodName,antecedent,0))
                    return
                        
            # Create constraint as there are two ILP variables         

            # Only constructing constrains forcing IF to be true 
            m.addConstr(antecedent - consequent <= 0, name='If:') # var1 <= var2
            if self.ifLog: self.myLogger.debug("%s created constraint only: %s <= %s"%(logicMethodName,varsInfo['varsNames'][0],varsInfo['varsNames'][1]))
            
            return
        else:
            if varsInfo['No_of_ilp'] == 0: # No ILP variable
                if not antecedent:
                    if self.ifLog: self.myLogger.debug("%s is True - antecedent is False - returning %i"%(logicMethodName,1))
                    if hasNone: self.myLogger.info("%s is True - antecedent is False - returning %i"%(logicMethodName,1))
                    return 1
                elif consequent: # antecedent is True
                    if self.ifLog: self.myLogger.debug("%s is True - antecedent and consequent are True - returning %i"%(logicMethodName,1))
                    if hasNone: self.myLogger.info("%s is True - antecedent and consequent are True - returning %i"%(logicMethodName,1))
                    return 1
                else: # antecedent and not consequent
                    if self.ifLog: self.myLogger.debug("%s is False - is called with the antecedent True and the consequent False - returning %i"%(logicMethodName,0))
                    if hasNone: self.myLogger.info("%s is False - is called with the antecedent True and the consequent False - returning %i"%(logicMethodName,0))
                    return 0
            elif self.__varIsNumber(antecedent):
                if not antecedent:
                    if self.ifLog: self.myLogger.debug("%s is True - antecedent is False - returning %i"%(logicMethodName,1))
                    if hasNone: self.myLogger.info("%s is True - antecedent is False - returning %i"%(logicMethodName,1))
                    return 1
                else: #antecedent
                    if self.ifLog: self.myLogger.debug("%s returns: %s - antecedent is True"%(logicMethodName,consequent))
                    if hasNone: self.myLogger.info("%s returns: %s - antecedent is True"%(logicMethodName,consequent))
                    return consequent
            elif  self.__varIsNumber(consequent):
                if consequent:
                    if self.ifLog: self.myLogger.debug("%s is True - consequent is True - returning %i"%(logicMethodName,1))
                    if hasNone: self.myLogger.info("%s is True - consequent is True - returning %i"%(logicMethodName,1))
                    return 1
                else: #not consequent
                    notAntecedent = self.notVar(m, antecedent)
                    if self.ifLog: self.myLogger.debug("%s returns: %s - consequent is False"%(logicMethodName,notAntecedent.VarName))
                    if hasNone: self.myLogger.info("%s returns: %s - consequent is False"%(logicMethodName,notAntecedent.VarName))
                    return notAntecedent
       
            # Create new variable
            varIF = m.addVar(vtype=GRB.BINARY, name=varsInfo["varName"])
                
            # Build constrains
            m.addConstr(1 - antecedent <= varIF, name='If:')                # 1 - var1 <= varIF
            m.addConstr(consequent <= varIF, name='If:')                    # var2 <= varIF
            m.addConstr(1 - antecedent + consequent >= varIF, name='If:')   # 1- var1 + var2 >= varIF
                
            m.update()
            
            if self.ifLog: self.myLogger.debug("IF returns : %s"%(varsInfo["varName"]))
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
                varFixed.append(0) # when None
            else:
                varFixed.append(v)
        
        var = varFixed
        # --
        
        varsInfo = self.preprocessLogicalMethodVar(varFixed, logicMethodName, logicMethodName,  minN=1)
            
        updatedLimit = limit - varsInfo['numberSum']
        
        # If only constructing constrains forcing OR to be true 
        if onlyConstrains:
            # Adding ILP constraint
            if limitOp == '>=': # ilp >= L
                if updatedLimit <= 0: # The constraint is satisfied - the limit is negative or zero
                    if self.ifLog: self.myLogger.debug("%s constraint is satisfied - the limit %i is negative or zero"%(logicMethodName,updatedLimit))
                    return
                elif updatedLimit > varsInfo['No_of_ilp']: # The limit is greater than the number of ILP variable - the constraint cannot be satisfied
                    raise Exception("ILP model is infeasible - %s limit %i is greater than the number of ILP variable %i - the constraint %s cannot be satisfied"
                                    %(logicMethodName,updatedLimit,varsInfo['No_of_ilp'],logicMethodName))
                else:
                    # Create Constraint
                    m.addConstr(varsInfo['varSumLinExpr'] >= updatedLimit, name='Count %s:'%(logicMethodName)) # varSumLinExpr >= updatedLimit
                    if self.ifLog: self.myLogger.debug("%s created ILP constraint: %s >= %i"%(logicMethodName,varsInfo['varSumLinExprStr'],updatedLimit))
                    
            # This check is common for '<=' and '=='
            elif updatedLimit < 0: # The constraint not is satisfied - the limit is negative or zero so ilp sum cannot be less than it - ilp sum is zero or more
                raise Exception("ILP model is infeasible - %s limit %i is negative or zero, ilp sum cannot be less than it - the constraint %s cannot be satisfied"
                                    %(logicMethodName,updatedLimit,logicMethodName))
                
            elif limitOp == '<=': # ilp <= L
                if varsInfo['No_of_ilp'] == 0: # sum Ilp =0 and L >= 0
                    if self.ifLog: self.myLogger.debug("%s constraint is satisfied - no ILP variable"%(logicMethodName))
                    return
                else:
                    m.addConstr(varsInfo['varSumLinExpr'] <= updatedLimit, name='Count %s:'%(logicMethodName)) # varSumLinExpr <= updatedLimit
                    if self.ifLog: self.myLogger.debug("%s created ILP constraint: %s <= %i"%(logicMethodName,varsInfo['varSumLinExprStr'],updatedLimit))

            elif limitOp == '==': # ilp == L
                if varsInfo['No_of_ilp'] == 0:
                    if updatedLimit == 0:
                        if self.ifLog: self.myLogger.debug("%s constraint is satisfied - no ILP variable"%(logicMethodName))
                        return
                    else: # updatedLimit > 0
                        raise Exception("ILP model is infeasible - %s limit %i is not zero as number of ILP variable is zero - the constraint %s cannot be satisfied"
                                    %(logicMethodName,updatedLimit,logicMethodName))
                else:  
                    m.addConstr(varsInfo['varSumLinExpr'] == updatedLimit, name='Count %s:'%(logicMethodName)) # varSumLinExpr == updatedLimit
                    if self.ifLog: self.myLogger.debug("%s created ILP constraint: %s == %i"%(logicMethodName,varsInfo['varSumLinExprStr'],updatedLimit))

            return
        
        # ------- If creating variables representing value of OR build of provided variables
        else:
        # Build constrains
            if limitOp == '>=': # ilp >= L
                if updatedLimit <= 0: # The constraint is satisfied - the limit is negative or zero - return 1
                    if self.ifLog: self.myLogger.debug("%s constraint is satisfied - the limit %i is negative or zero - return 1"%(logicMethodName,updatedLimit))
                    return 1
                elif updatedLimit > varsInfo['No_of_ilp']: # The limit is greater than the number of ILP variable - the constraint cannot be satisfied
                    if self.ifLog: self.myLogger.debug("%s limit %i is greater than the number of ILP variable %i - the constraint %s cannot be satisfied - return False"
                                    %(logicMethodName,updatedLimit,varsInfo['No_of_ilp'],logicMethodName))
                    return False
                else:
                    # Create new variable
                    varCOUNT = m.addVar(vtype=GRB.BINARY, name=varsInfo['varName'])
                    if m: m.update()
                    
                    m.addConstr(varsInfo['varSumLinExpr'] - BigM *varCOUNT >= updatedLimit - BigM, name='Count %s:'%(logicMethodName))
                    m.addConstr(varsInfo['varSumLinExpr'] - BigM *varCOUNT <= updatedLimit, name='Count %s:'%(logicMethodName))   
                                 
            # This check is common for '<=' and '=='
            elif updatedLimit < 0: # The constraint not is satisfied - the limit is negative or zero so ilp sum cannot be less than it - ilp sum is zero or more
                if self.ifLog: self.myLogger.debug("%s limit %i is negative or zero, ilp sum cannot be less than it - the constraint %s cannot be satisfied - return False"
                                    %(logicMethodName,updatedLimit,logicMethodName))
                return False
                
            elif limitOp == '<=': # ilp <= L
                if varsInfo['No_of_ilp'] == 0: # No ILP variable - sum Ilp =0 and L >= 0
                    if self.ifLog: self.myLogger.debug("%s constraint is satisfied - no ILP variable"%(logicMethodName))
                    return True
                else:  
                    # Create new variable
                    varCOUNT = m.addVar(vtype=GRB.BINARY, name=varsInfo['varName'])
                    if m: m.update()
                    
                    m.addConstr(varsInfo['varSumLinExpr'] + BigM *varCOUNT <= updatedLimit + BigM, name='Count %s:'%(logicMethodName))
                    m.addConstr(varsInfo['varSumLinExpr'] + BigM *varCOUNT >= updatedLimit, name='Count %s:'%(logicMethodName))
                    
            elif limitOp == '==': # ilp == L
                if varsInfo['No_of_ilp'] == 0:
                    if updatedLimit == 0:
                        if self.ifLog: self.myLogger.debug("%s constraint is satisfied - no ILP variable - return True"%(logicMethodName))
                        return True
                    else: # updatedLimit > 0
                        if self.ifLog: self.myLogger.debug("I%s limit %i is not zero as number of ILP variable is zero - the constraint %s cannot be satisfied - return False"
                                    %(logicMethodName,updatedLimit,logicMethodName))
                        return False
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