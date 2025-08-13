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
        S = varsInfo['varSumLinExpr']
       
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
                m.addConstr(varsInfo['No_of_ilp'] - S <= 0, name='And:') #  varSumLinExpr >= N
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
                m.addConstr(S - varAND <= varsInfo['No_of_ilp'] - 1, name='And:') #  varSumLinExpr <= varAND + N - 1
    
                if self.ifLog: self.myLogger.debug("%s returns: %s"%(logicMethodName,varsInfo['varName']))
                return varAND

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
        S = varsInfo['varSumLinExpr']
        
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
                m.addConstr(S >= 1, name='Or:') # varSumLinExpr >= 1
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
        
                m.addConstr(S - varOR >= 0, name='Or:') # varSumLinExpr >= varOR
                if self.ifLog: self.myLogger.debug("%s created constraint: %s - %s >= %i"%(logicMethodName,varsInfo['varSumLinExprStr'],varsInfo['varName'],1-1))
        
                if self.ifLog: self.myLogger.debug("%s returns new variable: %s"%(logicMethodName,varsInfo['varName']))
                return varOR
             
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
        notVat = self.notVar(m, varNAND)
        for currentVar in var:
            m.addConstr(notVat <= currentVar, name='Nand:')

        m.addConstr(varsInfo["varSumLinExpr"]<= notVat + varsInfo["N"] - 1, name='Nand:')
    
        return varNAND
    
    def norVar(self, m, *var, onlyConstrains = False):
        return self.notVar(m, self.orVar(m, var), onlyConstrains=onlyConstrains) # Negation of the disjunction
    
    def xorVar(self, m, *var, onlyConstrains = False):        
        # Conjunction of the disjunction and the negation of the conjunction
        return self.andVar(m, self.orVar(m, var), self.notVar(m, self.andVar(m, var)), onlyConstrains=onlyConstrains) 
    
    def ifVar(self, m, var1, var2, onlyConstrains=False):
        """
        Logical implication: (var1 => var2).
        - If either side is None (missing), we do NOT force anything.
            * onlyConstrains=True: no constraint added (skip).
            * onlyConstrains=False: vacuously return 1.
        - If both are numeric/bool, evaluate and return {0,1} (no constraints).
        - If one side is numeric and the other is an ILP var:
            * antecedent == 1  and consequent is ILP  -> add: consequent >= 1
            * antecedent == 0  and consequent is ILP  -> no constraint (vacuous truth)
            * antecedent is ILP and consequent == 1   -> no constraint (vacuous truth)
            * antecedent is ILP and consequent == 0   -> add: antecedent <= 0
        - If both are ILP vars:
            * onlyConstrains=True: add A - B <= 0   (A <= B)
            * onlyConstrains=False: create z = (¬A ∨ B) with standard linearization.
        """
        logicMethodName = "IF"

        # --- 1) Short-circuit on missing inputs (do NOT coerce None to 1/0) ---
        if var1 is None or var2 is None:
            if self.ifLog:
                self.myLogger.debug("%s: skipping (one side is None); %s"
                                    % (logicMethodName,
                                    "no constraints" if onlyConstrains else "return 1"))
            if onlyConstrains:
                return
            return 1  # vacuous truth when building an expression

        antecedent = var1
        consequent = var2

        # Helpers
        is_num_ante = self.__varIsNumber(antecedent)
        is_num_cons = self.__varIsNumber(consequent)

        # For logging / naming (safe even for mixed types)
        varsInfo = self.preprocessLogicalMethodVar((antecedent, consequent),
                                                logicMethodName, "if", minN=2)
        if varsInfo['N'] > 2:
            raise Exception("%s has %i variables, accepts only 2" % (logicMethodName, varsInfo['N']))

        # Normalize numeric to {0,1} for clean reasoning
        def as01(x): return 1 if bool(x) else 0

        # --- 2) Constraint-only mode: enforce A => B without creating a return var ---
        if onlyConstrains:
            if is_num_ante and is_num_cons:
                A = as01(antecedent)
                B = as01(consequent)
                # Purely numeric; implication holds unless A=1 and B=0.
                if A == 1 and B == 0:
                    # Warn and skip — do NOT force infeasibility here.
                    self.myLogger.warn("%s: antecedent=True and consequent=False (numeric); "
                                    "implication would be False; ignoring in constraint-only mode"
                                    % logicMethodName)
                else:
                    if self.ifLog: self.myLogger.debug("%s: numeric implication is True" % logicMethodName)
                return

            if is_num_ante and not is_num_cons:
                A = as01(antecedent)
                if A == 1:
                    # A => B  with A=1  -> force B=1
                    m.addConstr(consequent >= 1, name='If:')  # B >= 1
                    if self.ifLog:
                        self.myLogger.debug("%s: added constraint (A=1): %s >= 1"
                                            % (logicMethodName, varsInfo['varsNames'][1]))
                # A = 0 => vacuously true; no constraint
                return

            if not is_num_ante and is_num_cons:
                B = as01(consequent)
                if B == 0:
                    # A => 0  -> force A = 0
                    m.addConstr(antecedent <= 0, name='If:')  # A <= 0
                    if self.ifLog:
                        self.myLogger.debug("%s: added constraint (B=0): %s <= 0"
                                            % (logicMethodName, varsInfo['varsNames'][0]))
                # B = 1 => vacuously true; no constraint
                return

            # Both ILP vars: A <= B
            m.addConstr(antecedent - consequent <= 0, name='If:')  # A <= B
            if self.ifLog:
                self.myLogger.debug("%s: added constraint: %s <= %s"
                                    % (logicMethodName, varsInfo['varsNames'][0], varsInfo['varsNames'][1]))
            return

        # --- 3) Expression mode: return a {0,1}/var representing (¬A ∨ B) ---
        if is_num_ante and is_num_cons:
            A = as01(antecedent)
            B = as01(consequent)
            return 1 if (A == 0 or B == 1) else 0

        if is_num_ante and not is_num_cons:
            A = as01(antecedent)
            if A == 0:
                return 1  # vacuous truth
            else:
                # A==1 -> returns B (since (¬1 ∨ B) == B)
                if self.ifLog:
                    self.myLogger.debug("%s returns consequent (A=1): %s"
                                        % (logicMethodName, varsInfo['varsNames'][1]))
                return consequent

        if not is_num_ante and is_num_cons:
            B = as01(consequent)
            if B == 1:
                return 1  # (¬A ∨ 1) == 1
            else:
                # B==0 -> returns ¬A
                notAntecedent = self.notVar(m, antecedent)
                if self.ifLog:
                    self.myLogger.debug("%s returns NOT antecedent (B=0): %s"
                                        % (logicMethodName, notAntecedent.VarName))
                return notAntecedent

        # Both ILP vars: build z = (¬A ∨ B) with standard linearization
        varIF = m.addVar(vtype=GRB.BINARY, name=varsInfo["varName"])
        # z >= 1 - A
        m.addConstr(1 - antecedent <= varIF, name='If:')
        # z >= B
        m.addConstr(consequent <= varIF, name='If:')
        # z <= 1 - A + B
        m.addConstr(1 - antecedent + consequent >= varIF, name='If:')

        m.update()
        if self.ifLog:
            self.myLogger.debug("%s returns : %s" % (logicMethodName, varsInfo["varName"]))
        return varIF

           
    def equivalenceVar(self, m, *var, onlyConstrains = False):
        logicMethodName = "EQUIVALENCE"
        
        # -- Consider None
        varFixed = []  
        for v in var:
            if v is None:
                varFixed.append(0) # when None
            else:
                varFixed.append(v)
        # --
        
        if len(varFixed) == 0:
            # Equivalence of no variables is True (vacuous truth)
            if self.ifLog: self.myLogger.debug("%s returns: %i (no variables)"%(logicMethodName, 1))
            return 1
        elif len(varFixed) == 1:
            # Equivalence of single variable is True (always equivalent to itself)
            if self.ifLog: self.myLogger.debug("%s returns: %i (single variable)"%(logicMethodName, 1))
            return 1
        else:
            # Multi-variable equivalence using existing methods:
            # equiv(a, b, c, ...) = AND(a, b, c, ...) OR AND(NOT(a), NOT(b), NOT(c), ...)
            
            if self.ifLog: self.myLogger.debug("%s called with: %s"%(logicMethodName, [v if self.__varIsNumber(v) else v.VarName for v in varFixed]))
            
            # All true case: AND of all variables
            all_true = self.andVar(m, *varFixed)
            
            # All false case: AND of all negated variables
            negated_vars = [self.notVar(m, v) for v in varFixed]
            all_false = self.andVar(m, *negated_vars)
            
            # Equivalence = (all true) OR (all false)
            return self.orVar(m, all_true, all_false, onlyConstrains=onlyConstrains)
        
    def countVar(self, m, *var, onlyConstrains = False, limitOp = 'None', limit = 1, logicMethodName = "COUNT"):
        BigM = 1000         # set limit on number of variables possible to be used in the counting constraint
        LittleM = 0.0001    # set the precision of equality constraint
                
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
        S = varsInfo['varSumLinExpr']
        updatedLimit = limit - varsInfo['numberSum']
        
        # If only constructing constrains forcing OR to be true 
        if onlyConstrains:
            # Adding ILP constraint
            if limitOp == '>=': # ilp >= L atLeast
                if updatedLimit <= 0: # The constraint is satisfied - the limit is negative or zero
                    if self.ifLog: self.myLogger.debug("%s constraint is satisfied - the limit %i is negative or zero"%(logicMethodName,updatedLimit))
                    return
                elif updatedLimit > varsInfo['No_of_ilp']: # The limit is greater than the number of ILP variable - the constraint cannot be satisfied
                    raise Exception("ILP model is infeasible - %s limit %i is greater than the number of ILP variable %i - the constraint %s cannot be satisfied"
                                    %(logicMethodName,updatedLimit,varsInfo['No_of_ilp'],logicMethodName))
                else:
                    # Create Constraint
                    m.addConstr(S >= updatedLimit, name='Count %s:'%(logicMethodName)) # varSumLinExpr >= updatedLimit
                    if self.ifLog: self.myLogger.debug("%s created ILP constraint: %s >= %i"%(logicMethodName,varsInfo['varSumLinExprStr'],updatedLimit))
                    
            # This check is common for '<=' and '==' atNost and Equal
            elif updatedLimit < 0: # The constraint not is satisfied - the limit is negative or zero so ilp sum cannot be less than it - ilp sum is zero or more
                raise Exception("ILP model is infeasible - %s limit %i is negative or zero, ilp sum cannot be less than it - the constraint %s cannot be satisfied"
                                    %(logicMethodName,updatedLimit,logicMethodName))
                
            elif limitOp == '<=': # ilp <= L atMost
                if varsInfo['No_of_ilp'] == 0: # sum Ilp =0 and L >= 0
                    if self.ifLog: self.myLogger.debug("%s constraint is satisfied - no ILP variable"%(logicMethodName))
                    return
                else:
                    m.addConstr(S <= updatedLimit, name='Count %s:'%(logicMethodName)) # varSumLinExpr <= updatedLimit
                    if self.ifLog: self.myLogger.debug("%s created ILP constraint: %s <= %i"%(logicMethodName,varsInfo['varSumLinExprStr'],updatedLimit))

            elif limitOp == '==': # ilp == L Equal
                if varsInfo['No_of_ilp'] == 0:
                    if updatedLimit == 0:
                        if self.ifLog: self.myLogger.debug("%s constraint is satisfied - no ILP variable"%(logicMethodName))
                        return
                    else: # updatedLimit > 0
                        raise Exception("ILP model is infeasible - %s limit %i is not zero as number of ILP variable is zero - the constraint %s cannot be satisfied"
                                    %(logicMethodName,updatedLimit,logicMethodName))
                else:  
                    m.addConstr(S == updatedLimit, name='Count %s:'%(logicMethodName)) # varSumLinExpr == updatedLimit
                    if self.ifLog: self.myLogger.debug("%s created ILP constraint: %s == %i"%(logicMethodName,varsInfo['varSumLinExprStr'],updatedLimit))

            return
        
        # ------- If creating variables representing value of result of comparison of the sum of ILP variables and limit
        else:
        # Build constrains
            if limitOp == '>=': # atLeast S >= L 
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
                    
                    m.addConstr(S  >= updatedLimit - BigM * (1 - varCOUNT), name='Count %s:'%(logicMethodName))
                    m.addConstr(S <= updatedLimit - 1 + BigM * varCOUNT, name='Count %s:'%(logicMethodName))  
                                 
            # This check is common for '<=' and '=='
            elif updatedLimit < 0: # The constraint not is satisfied - the limit is negative or zero so ilp sum cannot be less than it - ilp sum is zero or more
                if self.ifLog: self.myLogger.debug("%s limit %i is negative or zero, ilp sum cannot be less than it - the constraint %s cannot be satisfied - return False"
                                    %(logicMethodName,updatedLimit,logicMethodName))
                return False
                
            elif limitOp == '<=': # atMost S <= L 
                if varsInfo['No_of_ilp'] == 0: # No ILP variable - sum Ilp = 0 and L >= 0
                    if self.ifLog: self.myLogger.debug("%s constraint is satisfied - no ILP variable"%(logicMethodName))
                    return True
                else:  
                    # Create new variable
                    varCOUNT = m.addVar(vtype=GRB.BINARY, name=varsInfo['varName'])
                    if m: m.update()
                    
                    m.addConstr(S <= updatedLimit + BigM * (1 - varCOUNT), name='Count %s:'%(logicMethodName))
                    m.addConstr(S >= updatedLimit + 1 - BigM * varCOUNT, name='Count %s:'%(logicMethodName))
                    
            elif limitOp == '==': # Exact S == L 
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
                    z_le = m.addVar(vtype=GRB.BINARY, name=varsInfo['varName'] + '_le')
                    z_ge = m.addVar(vtype=GRB.BINARY, name=varsInfo['varName'] + '_ge')
                    if m: m.update()

                    # z_le : S ≤ L
                    m.addConstr(S <= updatedLimit + BigM*(1 - z_le))
                    m.addConstr(S >= updatedLimit + 1 - BigM*z_le)

                    # z_ge : S ≥ L
                    m.addConstr(S >= updatedLimit - BigM*(1 - z_ge))
                    m.addConstr(S <= updatedLimit - 1 + BigM*z_ge)

                    # varCOUNT = z_le ∧ z_ge
                    m.addConstr(varCOUNT <= z_le)
                    m.addConstr(varCOUNT <= z_ge)
                    m.addConstr(varCOUNT >= z_le + z_ge - 1)

            if self.ifLog: self.myLogger.debug("%s returns new variable: %s"%(logicMethodName,varsInfo['varName']))
            return varCOUNT
    
    def compareCountsVar(
        self,
        m,
        varsA,               # iterable of literals forming “left” count
        varsB,               # iterable of literals forming “right” count
        *,                    # force kwargs for clarity
        compareOp='>',        # one of '>', '>=', '<', '<=', '==', '!='
        diff = 0,             # optional constant offset: count(A) - count(B) ∘ diff
        onlyConstrains=False,
        logicMethodName="COUNT_CMP",
    ):
   
        if compareOp not in ('>', '>=', '<', '<=', '==', '!='):
            raise ValueError(f"{logicMethodName}: unsupported operator {compareOp}")

        # --- preprocess each side (re‑use your helper) ---------------------------
        infoA = self.preprocessLogicalMethodVar(
                    list(varsA), f"{logicMethodName}_A", "cntA", minN=1)
        infoB = self.preprocessLogicalMethodVar(
                    list(varsB), f"{logicMethodName}_B", "cntB", minN=1)

        # Constant parts (0/1 literals encountered)
        constA = infoA['numberSum']
        constB = infoB['numberSum']

        # Symbolic sums over (binary) ILP vars
        sumA   = infoA['varSumLinExpr']
        sumB   = infoB['varSumLinExpr']

        # Upper bound for |ΣA − ΣB|  → use total number of ILP vars + |diff|
        BigM = infoA['No_of_ilp'] + infoB['No_of_ilp'] + abs(diff)

        # ------------------------------------------------------------------------
        #            ONLY CONSTRAINTS  (no indicator variable returned)
        # ------------------------------------------------------------------------
        if onlyConstrains:

            expr = sumA - sumB + (constA - constB)    # linear expr  ΣA - ΣB
            rhs  = diff                               # compare to diff

            if   compareOp == '>':  m.addConstr(expr >= rhs + 1, name=logicMethodName)
            elif compareOp == '>=': m.addConstr(expr >= rhs,     name=logicMethodName)
            elif compareOp == '<':  m.addConstr(expr <= rhs - 1, name=logicMethodName)
            elif compareOp == '<=': m.addConstr(expr <= rhs,     name=logicMethodName)
            elif compareOp == '==': m.addConstr(expr == rhs,     name=logicMethodName)
            elif compareOp == '!=':
                # (expr <= rhs-1) OR (expr >= rhs+1)   → two constraints & one aux‑binary
                z = m.addVar(vtype=GRB.BINARY, name=f"{logicMethodName}_neq")
                m.addConstr(expr <= rhs - 1 + BigM * z)
                m.addConstr(expr >= rhs + 1 - BigM * (1 - z))
            return

        # ------------------------------------------------------------------------
        #            WITH INDICATOR VARIABLE  (returned)
        # ------------------------------------------------------------------------
        # quick‑return if the relation is already decided by constants
        exprConst = constA - constB
        if   compareOp in ('>', '>=') and exprConst - diff >= (1 if compareOp == '>' else 0) \
            and infoA['No_of_ilp'] == 0 and infoB['No_of_ilp'] == 0:
            return 1
        if   compareOp in ('<', '<=') and exprConst - diff <= (-1 if compareOp == '<' else 0) \
            and infoA['No_of_ilp'] == 0 and infoB['No_of_ilp'] == 0:
            return 1
        if   compareOp == '==' and exprConst - diff == 0 \
            and infoA['No_of_ilp'] == 0 and infoB['No_of_ilp'] == 0:
            return 1
        if   compareOp == '!=' and exprConst - diff != 0 \
            and infoA['No_of_ilp'] == 0 and infoB['No_of_ilp'] == 0:
            return 1
        # (symmetrically, you could return 0 for impossible cases)

        # Build indicator
        varCMP = m.addVar(vtype=GRB.BINARY,
                        name=f"{logicMethodName}_{compareOp}_{infoA['varName']}_{infoB['varName']}")

        expr = sumA - sumB + (constA - constB)        # ΣA - ΣB + c
        rhs  = diff

        if compareOp in ('>', '>='):

            strict = (compareOp == '>')

            # expr >= rhs + (strict ? 1 : 0)
            m.addConstr(expr >= rhs + (1 if strict else 0) - BigM * (1 - varCMP),
                        name=f"{logicMethodName}_lb")
            # expr <= rhs + (strict ? 0 : -1) + BigM
            m.addConstr(expr <= rhs - (1 if strict else 0) + BigM * varCMP,
                        name=f"{logicMethodName}_ub")

        elif compareOp in ('<', '<='):

            strict = (compareOp == '<')

            # expr <= rhs - (strict ? 1 : 0)
            m.addConstr(expr <= rhs - (1 if strict else 0) + BigM * (1 - varCMP),
                        name=f"{logicMethodName}_ub")
            # expr >= rhs - (strict ? 0 : -1) - BigM
            m.addConstr(expr >= rhs + (1 if strict else 0) - BigM * varCMP,
                        name=f"{logicMethodName}_lb")

        elif compareOp == '==':

            # Two sided
            m.addConstr(expr - rhs <=  BigM * (1 - varCMP))
            m.addConstr(expr - rhs >= -BigM * (1 - varCMP))
            # If varCMP = 0, push expr at least 1 away from rhs
            m.addConstr(expr - rhs >= 1 - BigM * varCMP)
            m.addConstr(expr - rhs <= -1 + BigM * varCMP)

        elif compareOp == '!=':

            # expr differs from rhs by ≥1
            m.addConstr(expr - rhs >= 1 - BigM * (1 - varCMP))
            m.addConstr(expr - rhs <= -1 + BigM * (1 - varCMP))
            # varCMP = 0  ⇒  expr == rhs
            m.addConstr(expr - rhs <=  BigM * varCMP)
            m.addConstr(expr - rhs >= -BigM * varCMP)

        return varCMP
  
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