from time import perf_counter, perf_counter_ns
import torch
from collections import OrderedDict

from domiknows.graph import fixedL, ifL, forAllL

class LogicalConstraintVerifier:
    """
    Helper class for verifying logical constraint results.
    
    This class verifies whether logical constraints are satisfied by checking the model's
    predictions against the constraint definitions. It evaluates constraints on the predicted
    outputs (typically argmax results) rather than on probability distributions or ILP variables.
    
    The verifier:
    - Checks overall satisfaction rate for each logical constraint
    - For conditional constraints (ifL, forAllL), also checks satisfaction rate when the
      antecedent (condition) is true
    - Reports detailed statistics including processing time and satisfaction percentages
    
    Verification is performed post-inference to assess constraint compliance without requiring
    the ILP solver. This is useful for:
    - Evaluating model predictions against domain knowledge
    - Debugging constraint definitions
    - Analyzing which constraints are frequently violated
    - Comparing different inference methods (ML-only vs ILP-enhanced)
    """
    
    def __init__(self, solver):
        """
        Initialize verifier with reference to main solver.
        
        Args:
            solver: Reference to gurobiILPOntSolver instance that provides access to:
                   - myGraph: Graph structure with logical constraints
                   - booleanMethodsCalculator: Calculator for boolean operations
                   - constructLogicalConstrains: Method to construct constraint checks
                   - myLogger/myLoggerTime: Logging facilities
        """
        self.solver = solver
    
    def verifySingleConstraint(self, lc, myBooleanMethods, dn, key="/argmax"):
        """
        Verify a single logical constraint against model predictions.
        
        This method evaluates how well the model's predictions satisfy a single logical
        constraint. It operates on discrete predictions (e.g., argmax results) rather than
        probability distributions, providing binary satisfaction metrics.
        
        Args:
            lc: The logical constraint to verify
            myBooleanMethods: Boolean methods calculator instance
            dn: Data node containing the predictions to verify
            key: Attribute key for accessing predictions in datanodes.
                 Default: "/argmax" for discrete predicted labels
            
        Returns:
            dict: Verification result for the constraint with structure:
                {
                    'verifyList': [[bool, ...], ...],  # Satisfaction per instance
                    'satisfied': float,                 # Overall satisfaction % (0-100)
                    'ifVerifyList': [[bool, ...], ...], # (ifL/forAllL only) Filtered list
                    'ifSatisfied': float,               # (ifL/forAllL only) Conditional satisfaction % (0-100)
                    'elapsedInMsLC': float              # Processing time in milliseconds
                }
        """
        startLC = perf_counter_ns()
        
        m = None
        p = 0
        result = {}
        
        self.solver.myLogger.info('\n')
        self.solver.myLogger.info('Processing %r - %s' % (lc, lc.strEs()))
        
        self.solver.constraintConstructor.current_device = dn.current_device
        self.solver.constraintConstructor.myGraph = self.solver.myGraph
        verifyList, lcVariables = self.solver.constraintConstructor.constructLogicalConstrains(
            lc, myBooleanMethods, m, dn, p, key=key, headLC=True, verify=True)
        result['verifyList'] = verifyList
        
        verifyListLen = 0
        verifyListSatisfied = 0
        for vl in verifyList:
            verifyListLen += len(vl)
            verifyListSatisfied += sum(vl)
        
        if verifyListLen:
            result['satisfied'] = (verifyListSatisfied / verifyListLen) * 100
        else:
            result['satisfied'] = 0
            
        # If this is an if logical constraint
        if type(lc) is ifL or type(lc) is forAllL:
            firstKey = next(iter(lcVariables))
            firstLcV = lcVariables[firstKey]
            
            ifVerifyList = []
            ifVerifyListLen = 0
            ifVerifyListSatisfied = 0
            
            for i, v in enumerate(verifyList):
                ifVi = []
                if len(v) != len(firstLcV[i]):
                    continue
                
                for j, w in enumerate(v):
                    if torch.is_tensor(firstLcV[i][j]):
                        currentAntecedent = firstLcV[i][j].item()
                    else: 
                        currentAntecedent = firstLcV[i][j] 
                        
                    if currentAntecedent == 1:
                        ifVi.append(w)
                            
                ifVerifyList.append(ifVi)
                
                ifVerifyListLen += len(ifVi)
                ifVerifyListSatisfied += sum(ifVi)
            
            result['ifVerifyList'] = ifVerifyList
            if ifVerifyListLen:
                result['ifSatisfied'] = (ifVerifyListSatisfied / ifVerifyListLen) * 100
            else:
                result['ifSatisfied'] = 0

        endLC = perf_counter_ns()
        elapsedInNsLC = endLC - startLC
        elapsedInMsLC = elapsedInNsLC / 1000000
        result['elapsedInMsLC'] = elapsedInMsLC
        
        return result
        
    def verifyResults(self, dn, key="/argmax"):
        """
        Verify results of logical constraints against model predictions.
        
        This method evaluates how well the model's predictions satisfy the defined logical
        constraints. It operates on discrete predictions (e.g., argmax results) rather than
        probability distributions, providing binary satisfaction metrics.
        
        For each active logical constraint, the verifier:
        1. Constructs the constraint check using the boolean calculator
        2. Evaluates the constraint across all applicable instances
        3. Computes overall satisfaction rate (% of instances satisfying the constraint)
        4. For conditional constraints (ifL/forAllL), also computes conditional satisfaction
           rate (% satisfaction when the antecedent/condition is true)
        
        Args:
            dn: Data node containing the predictions to verify. Should have predictions
                stored with the specified key (e.g., 'local/argmax' for discrete predictions)
            key: Attribute key for accessing predictions in datanodes. 
                 Default: "/argmax" for discrete predicted labels
                 Common alternatives: "/local/argmax", "/ILP/x"
            
        Returns:
            OrderedDict: Verification results for each logical constraint with structure:
                {
                    'constraint_name': {
                        'verifyList': [[bool, ...], ...],  # Satisfaction per instance
                        'satisfied': float,                 # Overall satisfaction % (0-100)
                        'ifVerifyList': [[bool, ...], ...], # (ifL/forAllL only) Filtered list
                        'ifSatisfied': float,               # (ifL/forAllL only) Conditional satisfaction % (0-100)
                        'elapsedInMsLC': float              # Processing time in milliseconds
                    },
                    ...
                }
            
        Notes:
            - Only active head constraints are verified (nested constraints are skipped)
            - Fixed constraints (fixedL) are skipped as they don't need verification
            - For ifL/forAllL constraints, 'ifSatisfied' measures satisfaction when the
              condition is true, which is often more meaningful than overall satisfaction
            - A satisfaction rate of 100% means all instances satisfy the constraint
            - Lower satisfaction rates indicate constraint violations that may need attention
        """
        start = perf_counter()
        
        myBooleanMethods = self.solver.booleanMethodsCalculator
        myBooleanMethods.current_device = dn.current_device

        self.solver.myLogger.info('Calculating verify ')
        self.solver.myLoggerTime.info('Calculating verify ')
        
        lcCounter = 0
        lcVerifyResult = OrderedDict()
        
        for graph in self.solver.myGraph:
            for _, lc in graph.logicalConstrains.items():
                if not lc.headLC or not lc.active:
                    continue
                    
                if type(lc) is fixedL:
                    continue
                    
                lcCounter += 1
                lcName = lc.lcName
                lcVerifyResult[lcName] = self.verifySingleConstraint(lc, myBooleanMethods, dn, key)
        
        self.solver.myLogger.info('Processed %i logical constraints' % (lcCounter))
        self.solver.myLoggerTime.info('Processed %i logical constraints' % (lcCounter))
            
        end = perf_counter()
        elapsedInS = end - start
        
        if elapsedInS > 1:
            self.solver.myLogger.info('End of Verify Calculation - total time: %fs' % (elapsedInS))
            self.solver.myLoggerTime.info('End of Verify Calculation - total time: %fs' % (elapsedInS))
        else:
            elapsedInMs = (end - start) * 1000
            self.solver.myLogger.info('End of Verify Calculation - total time: %ims' % (elapsedInMs))
            self.solver.myLoggerTime.info('End of Verify Calculation - total time: %ims' % (elapsedInMs))
            
        self.solver.myLogger.info('')
        self.solver.myLoggerTime.info('')

        [h.flush() for h in self.solver.myLoggerTime.handlers]
        return lcVerifyResult