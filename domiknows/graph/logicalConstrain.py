from collections import namedtuple
from domiknows.solver.ilpConfig import ilpConfig 
from domiknows.graph import Concept
from domiknows.solver.lcLossSampleBooleanMethods import lcLossSampleBooleanMethods
import logging
import torch
myLogger = logging.getLogger(ilpConfig['log_name'])
ifLog =  ilpConfig['ifLog']
        
V = namedtuple("V", ['name', 'v', 'relVarInfo'], defaults= [None, None, None])

class LcElement:
    def __init__(self, *e,  name = None):
        from .relation import Relation

        self.typeName = "Logical Element"
        if isinstance(self, LogicalConstrain):
            self.typeName = "Logical Constraint"
            
        if not e:
            raise LcElement.LcElementError(f"{self.typeName} is empty")
        
        updatedE = []
        lenE = len(e)
        self.cardinalityException = ""
        
        if isinstance(self, eqL):
            expected_types = {
                0: (Concept, Relation, tuple),
                1: str,
                2: set
            }
            
            for i, eItem in enumerate(e):
                expected_type = expected_types.get(i)
                if expected_type and isinstance(eItem, expected_type):
                    updatedE.append(eItem)
                else:
                    error_message = f"{self.typeName} is incorrect - {eItem} is not a valid {i} element of the {self.typeName}"
                    raise LcElement.LcElementError(error_message)
        else:
            for i, eItem in enumerate(e):
                if isinstance(eItem, (LcElement, Concept, Relation, tuple)): # lc element without path
                    updatedE.append(eItem)
                elif callable(eItem): # multiclass label
                    newEItem = eItem.__call__() # generated label
                    updatedE.extend(newEItem)
                elif isinstance(eItem, list): # lc element with path
                    updatedE.extend(eItem)
                elif isinstance(eItem, int):
                    if i<lenE-1:
                        self.cardinalityException = eItem
    
                    updatedE.append(eItem)
                else:
                    raise LcElement.LcElementError(f"{self.typeName} is incorrect - {eItem} is not a valid element of the {self.typeName}")
                
        self.e = updatedE
        
        updatedE = []
        for _, eItem in enumerate(self.e):
            if eItem == None:
                continue
        
            if isinstance(eItem, Concept): # binary concept mapping to tuple representation
                updatedE.append((eItem, eItem.name, None, 1))
            elif isinstance(eItem, V) and eItem.relVarInfo is not None: # Concept with relation variable mapping to tuple representation
                    updatedArgs = {}
                    for arg, v in eItem.relVarInfo.items():
                        concept = v.relVarInfo
                        if isinstance(concept, Concept):
                            # Concept path takes precedence. 
                            updatedArgs[arg] = V(name=v.name, v=v.v, relVarInfo=(concept, concept.name, None, 1))
                        elif callable(concept): # multiclass label
                            conceptEncoding = concept.__call__() # generated label
                            updatedArgs[arg] = V(name=v.name, v=v.v, relVarInfo=conceptEncoding)
                            
                    eItem = V(name=eItem.name, v=eItem.v, relVarInfo=updatedArgs)
                    updatedE.append(eItem)
            else:
                updatedE.append(eItem)
                
        self.e = updatedE

        # -- Find the graph for this Logical Element - based on context defined in the concept used in constraint definition
        self.graph = None
       
        conceptOrLc = None
        
        for _, eItem in enumerate(self.e):
            if isinstance(eItem, (LcElement, Relation)):
                conceptOrLc = eItem
                break
            elif isinstance(eItem, tuple): # Concept
                if isinstance(eItem[0], Concept):
                    conceptOrLc = eItem[0]
                    break
    
        if conceptOrLc is None:
            myLogger.error("Logical Element is incorrect")
            raise LcElement.LcElementError(f"{self.typeName} {self.name} is incorrect - no concept or logical constrain found")
            
        if isinstance(conceptOrLc, Concept):
            if self.__getContext(conceptOrLc):
                self.graph = self.__getContext(conceptOrLc)[-1]
        elif isinstance(conceptOrLc, (Relation,)):
            self.graph = conceptOrLc.graph
        elif isinstance(conceptOrLc, str):
            self.graph = None
        else:
            self.graph = conceptOrLc.graph
                
        if self.graph == None:
            raise LcElement.LcElementError(f"{self.typeName} is incorrect - no graph found for the {self.typeName}")
        
        # Create Logical Element id based on number of existing Logical Element in the graph
        self.lcName = "LC%i"%(len(self.graph.logicalConstrains))
        
        if name is not None:
            self.name = name
        else:
            self.name = self.lcName            
     
    class LcElementError(Exception):
        pass
    
    def __str__(self):
        return self.lcName
    
    def __repr__(self):
        return  self.lcName + '(' + self.__class__.__name__ + ')'
            
    def __getContext(self, e):
        if isinstance(e, LcElement):
            return self.__getContext(e.e[0])
        else:
            return e._context

class LogicalConstrain(LcElement):
    def __init__(self, *e, p=100, active = True, sampleEntries  = False, name = None):
        super().__init__(*e, name = name)
        
        self.headLC = True # Indicate that it is head constraint and should be process individually
        self.active = active
        self.sampleEntries = sampleEntries
        
        # Add the constraint to the graph
        assert self.graph is not None
        self.graph.logicalConstrains[self.lcName] = self
        
        # Go though constraint, find nested constrains and change their property headLC to indicate that their are nested and should not be process individually
        for e_item in self.e:
            if isinstance(e_item, LogicalConstrain):
                e_item.headLC = False
                
        # Check soft constraint is activated though p - if certainty in the validity of the constraint or the user preference is provided by p
        if p < 0:
            self.p = 0
            myLogger.warning("%s Logical Constraint created with p equal %i sets it to 0"%(self.lcName,p))
        elif p > 100:
            self.p = 100
            myLogger.warning("%s Logical Constraint created with p equal %i sets it to 100"%(self.lcName,p))
        else:
            self.p = p
        
    def __call__(self, model, myConstraintVarProcessor, v): 
        pass 
    
    def getLcConcepts(self):
        lcConcepts = set()
        
        for _, eItem in enumerate(self.e):
            if isinstance(eItem, V):
                if eItem[1] and isinstance(eItem[1], (list, tuple)) and len(eItem[1]) > 1:
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
        
    # Get string representation of  Logical Constraint
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
                        new_v = []
                        for v in eItem[1]:
                            if isinstance(v, (str, LogicalConstrain)):
                                new_v.append(v)
                            elif isinstance(v, (tuple)):
                                v_tuple = [w if isinstance(w, (str, tuple, LogicalConstrain)) else w.name for w in v]
                                new_v.append(v_tuple)
                            else:
                                new_v.append(v.name)

                        #new_v = [v if isinstance(v, (str, tuple, LogicalConstrain)) else v.name for v in eItem[1]]
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
    
    # ------------Method building logical constraints
    
    def createSingleVarLogicalConstrains(self, lcName, lcFun, model, v, headConstrain):  
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
            if ifLog: myLogger.debug("%s Logical Constraint is the head constraint - only ILP constraint created"%(lcName))
        
        if model is not None:
            model.update()
        
        return singleV
    
    # Collects setups of variables for logical methods calls for the created Logical Constraint - recursive method
    def _collectVariableSetups(self, lcVariableName, lcVariableNames, v, lcVars=[]): 
        """
        Collects setups of variables for logical methods calls - recursive method.
        
        Handles alignment of variables with different structures:
        - Simple variables: [N] rows, 1 element each
        - Relation variables: [N] rows, [M] elements each (e.g., front(b,y) has N rows for b, M elements for y)
        - Nested constraint results: [M] rows, 1 element each (indexed by second variable)
        
        When combining relation[b][y] with nested_result[y], we need element-based lookup,
        not row-based lookup. The element index within a row corresponds to the y variable.
        """
        # Get set of ILP variables lists for the current variable name
        cLcVariables = v[lcVariableName]
        
        # List of lists containing sets of ILP variables for particular position 
        newLcVars = []
        
        # --- Update the lcVars setup with ILP variables from this iteration
        
        if not lcVars:  # First iteration - initialize
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
                    
        elif cLcVariables is None or len(cLcVariables) == 0:
            # No variables to add - just append None
            for indexLcV, lcV in enumerate(lcVars):
                newV = []
                for lcVelement in lcV:
                    newElement = lcVelement.copy()
                    newElement.append(None)
                    newV.append(newElement)
                newLcVars.append(newV)
                    
        elif len(cLcVariables) == 1:  # Single variable - broadcast to all
            for indexLcV, lcV in enumerate(lcVars):
                newV = []
                for lcVelement in lcV:
                    newElement = lcVelement.copy()
                    if cLcVariables[0]:
                        newElement.append(cLcVariables[0][0])
                    else:
                        newElement.append(None)
                    newV.append(newElement)
                newLcVars.append(newV)
                                    
        else:  # Many ILP variables in the current set
            # Check structure characteristics
            num_lcVars_rows = len(lcVars)
            num_cLc_rows = len(cLcVariables)
            
            # Is this a nested constraint result? (shape [N,1] - N rows, 1 element each)
            is_nested_constraint_result = all(
                len(row) == 1 for row in cLcVariables if row
            )
            
            # Check if accumulated vars have been expanded (have multiple elements per row)
            # This happens when a relation variable was added (e.g., front(b,y) adds 6 elements per row)
            accumulated_is_expanded = any(len(row) > 1 for row in lcVars if row)
            
            # Get max elements per row in accumulated vars for alignment detection
            max_elements_per_row = max((len(row) for row in lcVars if row), default=1)
            
            # Primary alignment strategy - when row counts match, use row-based indexing
            rows_match = (num_lcVars_rows == num_cLc_rows)
            
            for indexLcV, lcV in enumerate(lcVars):
                newV = []
                for indexElement, lcVelement in enumerate(lcV):
                    newLcVelement = lcVelement.copy()
                    
                    if indexLcV >= len(cLcVariables):
                        # Out of bounds - append None
                        newLcVelement.append(None)
                        newV.append(newLcVelement)
                        continue
                    
                    cLcRow = cLcVariables[indexLcV]
                    
                    if cLcRow is None or len(cLcRow) == 0:
                        newLcVelement.append(None)
                        newV.append(newLcVelement)
                        
                    elif len(lcV) == len(cLcRow):
                        # Element counts match within row - use element-based indexing
                        cV = cLcRow[indexElement] if indexElement < len(cLcRow) else None
                        newLcVelement.append(cV)
                        newV.append(newLcVelement)
                        
                    elif is_nested_constraint_result and accumulated_is_expanded and indexElement < num_cLc_rows:
                        # When accumulated vars are expanded by a relation (e.g., front(b,y))
                        # and we have a nested constraint result (e.g., inner_AND(y)),
                        # use ELEMENT-INDEX-based lookup: element index corresponds to the y variable
                        # So for accumulated row b, element y, we want nested_result[y], not nested_result[b]
                        cV = cLcVariables[indexElement][0] if cLcVariables[indexElement] else None
                        newLcVelement.append(cV)
                        newV.append(newLcVelement)
                        
                    elif is_nested_constraint_result and rows_match:
                        # Row counts match, not expanded - use ROW-based alignment
                        # This is for cases where both variables are indexed by the same variable
                        cV = cLcRow[0] if cLcRow else None
                        newLcVelement.append(cV)
                        newV.append(newLcVelement)
                        
                    elif is_nested_constraint_result and indexElement < num_cLc_rows:
                        # Rows don't match but nested result - use element-based lookup
                        cV = cLcVariables[indexElement][0] if cLcVariables[indexElement] else None
                        newLcVelement.append(cV)
                        newV.append(newLcVelement)
                        
                    elif len(cLcRow) == 1:
                        # Single element in row - broadcast to all elements
                        cV = cLcRow[0]
                        newLcVelement.append(cV)
                        newV.append(newLcVelement)
                        
                    else:
                        # Fallback: expand by iterating through all values in the row
                        for cV in cLcRow:
                            newLcVelement_copy = lcVelement.copy()
                            newLcVelement_copy.append(cV)
                            newV.append(newLcVelement_copy)
                            
                newLcVars.append(newV)                
                            
        if lcVariableNames:
            # Recursive call
            return self._collectVariableSetups(
                lcVariableNames[0], lcVariableNames[1:], v, lcVars=newLcVars
            )
        else:
            return newLcVars
    # Helper to recursively flatten nested lists to scalar values
    def flatten_to_scalars(self, item):
        """Recursively flatten nested lists to a flat list of scalar values."""
        result = []
        if isinstance(item, (list, tuple)):
            for sub_item in item:
                result.extend(self.flatten_to_scalars(sub_item))
        elif item is not None:
            result.append(item)
        return result
        
    def createLogicalConstrains(self, lcName, lcFun, model, v, headConstrain):
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
        sVar = self._collectVariableSetups(lcVariableName0, lcVariableNames[1:], v)
        
        # Apply collected setups and create ILP constraint
        for z in sVar:
            tVars = [] # Collect ILP constraints results
            for t in z:
                # Check if 'onlyConstrains' is already in t (as a dict or by position)
                if isinstance(t, dict) and 'onlyConstrains' in t:
                    tVars.append(lcFun(model, *t))
                elif isinstance(t, (list, tuple)):
                    # Try to detect if onlyConstrains is already present by argument name
                    import inspect
                    sig = inspect.signature(lcFun)
                    param_names = list(sig.parameters.keys())
                    if 'onlyConstrains' in param_names and len(t) >= param_names.index('onlyConstrains') + 1:
                        tVars.append(lcFun(model, *t))
                    else:
                        tVars.append(lcFun(model, *t, onlyConstrains = headConstrain))
                else:
                    tVars.append(lcFun(model, *t, onlyConstrains = headConstrain))
            rVars.append(tVars)
        
        # Return results from created ILP constraints - 
        # None if headConstrain is True or no ILP constraint created, ILP variable representing the value of ILP constraint, loss calculated
        return rVars

    def createCountConstraints(self, model, myConstraintVarProcessor, v, headConstrain, cOperation, cLimit, integrate, logicMethodName="COUNT"):
        """
        Build count constraints for existsL, atMostL, atLeastL, exactL.
        
        Fixed: Properly handles nested constraint results which may have nested list structure.
        """
        try:
            lcVariableNames = [e for e in iter(v)]
        except StopIteration:
            pass
        if cLimit is None:
            cLimit = 1

        if not lcVariableNames:
            return [[None]]
            
        lcVariableName0 = lcVariableNames[0]
        lcVariableSet0 = v[lcVariableName0]

        if not lcVariableSet0:
            return [[None]]

        # -----------------------------
        # Build flattened var list(s)
        # -----------------------------
        batch_vars = []   # list of lists (per-row), unless headConstrain/integrate
        
        for i in range(len(lcVariableSet0)):
            row = []
            for currentV in lcVariableNames:
                current_val = v[currentV][i] if i < len(v[currentV]) else None
                # Flatten any nested structure to scalar values
                flattened = self.flatten_to_scalars(current_val)
                row.extend(flattened)

            if headConstrain or integrate:
                # accumulate globally
                batch_vars.extend(row)
            else:
                # element-wise
                if len(row) == 0:
                    batch_vars.append([])     # <-- mark empty row; we'll return None for it
                else:
                    batch_vars.append(row)

        # -----------------------------
        # Build constraints
        # -----------------------------
        zVars = []

        if headConstrain or integrate:
            # Global mode: if we have *no* buildable candidates, propagate None
            if len(batch_vars) == 0:
                zVars.append([None])
            else:
                # Filter out None values before passing to countVar
                valid_vars = [v for v in batch_vars if v is not None]
                if len(valid_vars) == 0:
                    zVars.append([None])
                else:
                    r = myConstraintVarProcessor.countVar(
                        model,
                        *valid_vars,
                        onlyConstrains=headConstrain,
                        limitOp=cOperation,
                        limit=cLimit,
                        logicMethodName=logicMethodName,
                    )
                    zVars.append([r])
        else:
            # Element-wise mode
            for row in batch_vars:
                if len(row) == 0:
                    zVars.append([None])
                else:
                    # Filter out None values
                    valid_vars = [v for v in row if v is not None]
                    if len(valid_vars) == 0:
                        zVars.append([None])
                    else:
                        r = myConstraintVarProcessor.countVar(
                            model,
                            *valid_vars,
                            onlyConstrains=headConstrain,
                            limitOp=cOperation,
                            limit=cLimit,
                            logicMethodName=logicMethodName,
                        )
                        zVars.append([r])

        if model is not None:
            model.update()

        return zVars

    def createAccumulatedCountConstraints(self, model, myConstraintVarProcessor, v, headConstrain, cOperation, cLimit, integrate, logicMethodName = "COUNT"):  
        
        # Ignore value of integrate
        integrate = True
               
        try:
            lcVariableNames = [e for e in iter(v)]
        except StopIteration:
            pass
        
        if cLimit == None:
            cLimit = 1
            
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
            r = myConstraintVarProcessor.countVar(model, *varsSetup, onlyConstrains = headConstrain, limitOp = cOperation, limit=cLimit, 
                                                     logicMethodName = logicMethodName)
            for _ in lcVariableSet0:
                zVars.append([r])
        else:
            for current_var in varsSetup:
                zVars.append([myConstraintVarProcessor.countVar(model, *current_var, onlyConstrains = headConstrain, limitOp = cOperation, limit=cLimit, 
                                                         logicMethodName = logicMethodName)])
       
        if model is not None:
            model.update()
            
        return zVars

    def createCompareCountsConstraints(self, model, myConstraintVarProcessor,v, headConstrain, compareOp, diff, integrate, *, logicMethodName="COUNT_CMP"):
        """
        Build ILP constraints (and optionally return indicator vars) enforcing
        compareOp between the **counts** of two variable sets.

        compareOp : one of '>', '>=', '<', '<=', '==', '!='
        diff      : constant offset  (we enforce  count(A) - count(B) ∘ diff)
        """
        try:
            lcVariableNames = [name for name in iter(v)]
        except StopIteration:
            return []

        if len(lcVariableNames) != 2:
            myLogger.error(
                "%s Comparative Logical Constraint created with %i sets of "
                "variables – need exactly two",
                logicMethodName, len(lcVariableNames)
            )
            return []

        nameA, nameB = lcVariableNames[:2]
        setA, setB   = v[nameA], v[nameB]

        if len(setA) != len(setB):
            myLogger.error(
                "%s has mismatching numbers of variable-tuples: %s=%i, %s=%i",
                logicMethodName, nameA, len(setA), nameB, len(setB)
            )
            return []

        zVars = []

        # ---------------------------------------------------------------
        # integrate / headConstrain ➜ single global constraint
        # ---------------------------------------------------------------
        if headConstrain or integrate:
            varsA_acc = [lit for tupleA in setA for lit in tupleA]
            varsB_acc = [lit for tupleB in setB for lit in tupleB]

            r = myConstraintVarProcessor.compareCountsVar(
                model,
                varsA_acc,
                varsB_acc,
                compareOp=compareOp,
                diff=diff,
                onlyConstrains=headConstrain,
                logicMethodName=logicMethodName,
            )
            zVars = [[r] for _ in setA]        # replicate to keep shape
        # ---------------------------------------------------------------
        # element-wise constraints
        # ---------------------------------------------------------------
        else:
            for tupleA, tupleB in zip(setA, setB):
                r = myConstraintVarProcessor.compareCountsVar(
                    model,
                    tupleA,
                    tupleB,
                    compareOp=compareOp,
                    diff=diff,
                    onlyConstrains=headConstrain,
                    logicMethodName=logicMethodName,
                )
                zVars.append([r])

        if model is not None:
            model.update()
        return zVars
    
    def createSummation(self, model, myConstraintVarProcessor, v, headConstrain, integrate, label=None, logicMethodName="SUMMATION"):
        """
        Build summation constraints for sumL.
        
        Fixed: Properly handles nested constraint results that may have 
        different shapes/indexing than the primary iteration variable.
        """
        try:
            lcVariableNames = [name for name in iter(v)]
        except StopIteration:
            return []

        if not lcVariableNames:
            return []
            
        lcVariableSet0 = v[lcVariableNames[0]]
        
        if not lcVariableSet0:
            return []
            
        zVars = []
        num_rows = len(lcVariableSet0)

        if headConstrain or integrate:
            # Global sum across all variables and all rows
            # For global mode, collect ALL values from ALL variables
            all_values = []
            for name in lcVariableNames:
                var_data = v[name]
                if var_data is not None:
                    all_values.extend(self.flatten_to_scalars(var_data))
            
            if len(all_values) == 0:
                zVars.append([0])
            else:
                S = myConstraintVarProcessor.summationVar(model, *all_values, onlyConstrains = headConstrain, label=label)
                zVars.append([S])
        else:
            # Element-wise per-row sums
            for i in range(num_rows):
                row = []
                for name in lcVariableNames:
                    var_data = v[name]
                    
                    if var_data is None:
                        continue
                    
                    # Check if this variable has the same number of rows
                    if len(var_data) == num_rows:
                        # Aligned - use direct index
                        if i < len(var_data):
                            row.extend(self.flatten_to_scalars(var_data[i]))
                    elif len(var_data) == 1:
                        # Single row - broadcast to all iterations
                        row.extend(self.flatten_to_scalars(var_data[0]))
                    else:
                        # Different number of rows - nested constraint result
                        # with different indexing. For element-wise sum,
                        # we can't properly align, so include all values
                        # in first row only to avoid double-counting
                        if i == 0:
                            row.extend(self.flatten_to_scalars(var_data))
                            myLogger.warning(f"{logicMethodName}: variable '{name}' has mismatched row count; ")
                            
                if len(row) == 0:
                    zVars.append([0])
                else:
                    S = myConstraintVarProcessor.summationVar(model, *row, onlyConstrains = headConstrain)
                    zVars.append([S])

        if model is not None:
            model.update()
        return zVars

    def createIotaSelection(self, model, myConstraintVarProcessor, v, headConstrain, integrate, temperature, logicMethodName):
        """
        Build ILP constraints / loss for definite description selection.
        
        The iota operator:
        1. Collects all variables representing condition satisfaction
        2. Enforces exactly one entity satisfies (uniqueness presupposition)
        3. Returns selection distribution over entities
        """
        try:
            lcVariableNames = [name for name in iter(v)]
        except StopIteration:
            return []
        
        if not lcVariableNames:
            myLogger.error(f"{logicMethodName} has no variables")
            return []
        
        lcVariableName0 = lcVariableNames[0]
        lcVariableSet0 = v[lcVariableName0]
        
        zVars = []
        
        condition_vars = []
        for i, _ in enumerate(lcVariableSet0):
            # Collect all condition variables for this grounding
            condition_vars.extend(lcVariableSet0[i])
            
        # Call the boolean processor's iotaVar method
        result = myConstraintVarProcessor.iotaVar(
            model,
            *condition_vars,
            onlyConstrains=headConstrain,
            temperature=temperature,
            logicMethodName=logicMethodName,
        )
        
        if len(condition_vars) == 0:
            zVars.append([None])
    
        if isinstance(result, (list, tuple)):
            zVars.append(list(result))
        else:
            zVars.append([result])
        
        if model is not None:
            model.update()
        
        return zVars
   
    def createQuerySelection(self, model, concept, subclasses, myConstraintVarProcessor, v, headConstrain, integrate, temperature, logicMethodName):
            """Build query selection over attribute subclasses.

            Variables in *v* fall into two groups:

            - **Selection variables** (from ``iotaL`` etc.) — entity selection weights.
              These do NOT start with ``_ql_``.
            - **Subclass-data variables** (added by ``queryL.__init__``) — per-entity
              subclass predictions.  Their names start with ``_ql_``.

            The method collects both groups and passes them to ``queryVar``.
            """
            try:
                lcVariableNames = [name for name in iter(v)]
            except StopIteration:
                return []

            if not lcVariableNames:
                myLogger.error(f"{logicMethodName} has no variables")
                return []

            # -- Separate selection vars from subclass-data vars --------
            sel_var_names = [n for n in lcVariableNames if not n.startswith('_ql_')]
            sub_var_names = [n for n in lcVariableNames if n.startswith('_ql_')]

            if not sel_var_names:
                myLogger.error(f"{logicMethodName} has no selection variables")
                return []

            # Iterate over the *selection* variables' row count
            sel_var_0 = v[sel_var_names[0]]
            num_iterations = len(sel_var_0)

            zVars = []

            for i in range(num_iterations):
                # Collect selection values for this row
                selection_vars = []
                for name in sel_var_names:
                    if i < len(v[name]):
                        selection_vars.extend(v[name][i])

                if len(selection_vars) == 0:
                    zVars.append([None])
                    continue

                # Collect subclass data if _ql_* variables are present
                subclass_data = None
                if sub_var_names:
                    subclass_data = self._collect_query_subclass_data(
                        v, sub_var_names, i, num_iterations, len(subclasses))

                result = myConstraintVarProcessor.queryVar(
                    model,
                    concept,
                    subclasses,
                    selection_vars,
                    subclass_data=subclass_data,
                    onlyConstrains=headConstrain,
                    temperature=temperature,
                    logicMethodName=logicMethodName,
                )

                zVars.append([result])

            return zVars

    @staticmethod
    def _collect_query_subclass_data(v, sub_var_names, iteration, num_sel_iterations, num_subclasses):
        """Gather per-entity subclass prediction data from ``_ql_*`` variables.

        Returns ``subclass_data[entity_idx]`` = list of *num_subclasses* values,
        one per subclass, representing the model's prediction for that entity.

        Two layouts are handled:

        * **EnumConcept** (single ``_ql_attr`` variable): each row already
          contains *K* values (one per enum member).
        * **is_a subtypes** (multiple ``_ql_sub_N`` variables): each variable
          has one value per entity per row.

        In verification mode (``num_sel_iterations == 1``) the selection
        variables have a single row while the subclass variables have *N*
        rows (one per entity).  We return all *N* entities' data so that
        ``queryVar`` can look up the selected entity.

        In loss mode (after the constructor's split, ``num_sel_iterations > 1``)
        every variable has *N* rows; we return only row *iteration*.
        """
        if num_sel_iterations == 1:
            # Verification / ILP mode — collect ALL entity rows
            if len(sub_var_names) == 1:
                # EnumConcept: single var, K values per entity row
                return [list(row) for row in v[sub_var_names[0]]]
            else:
                # is_a subtypes: K vars, 1 value per entity per var
                num_entities = len(v[sub_var_names[0]])
                result = []
                for entity_idx in range(num_entities):
                    entity_subs = []
                    for name in sub_var_names:
                        if entity_idx < len(v[name]):
                            entity_subs.extend(v[name][entity_idx])
                    result.append(entity_subs)
                return result
        else:
            # Loss mode (after split) — one entity per iteration
            if len(sub_var_names) == 1:
                var_data = v[sub_var_names[0]]
                if iteration < len(var_data):
                    return [list(var_data[iteration])]
                return None
            else:
                entity_subs = []
                for name in sub_var_names:
                    if iteration < len(v[name]):
                        entity_subs.extend(v[name][iteration])
                return [entity_subs] if entity_subs else None

    def createSameSelection(self, model, concept, subclasses, myConstraintVarProcessor, v, headConstrain, logicMethodName):
        """
        Build constraints checking whether all entities share the same subclass.

        Each variable name in v corresponds to one entity's subclass indicator
        variables (resolved from concept variable bindings like ``color('x')``).
        Collects per-entity groups and delegates to sameVar for comparison.
        """
        try:
            lcVariableNames = [name for name in iter(v)]
        except StopIteration:
            return []

        if not lcVariableNames:
            myLogger.error(f"{logicMethodName} has no variables")
            return []

        lcVariableName0 = lcVariableNames[0]
        lcVariableSet0 = v[lcVariableName0]

        zVars = []

        for i, _ in enumerate(lcVariableSet0):
            entity_var_groups = []
            for currentV in lcVariableNames:
                group = list(v[currentV][i]) if i < len(v[currentV]) else []
                entity_var_groups.append(group)

            if len(entity_var_groups) < 2:
                zVars.append([None])
                continue

            result = myConstraintVarProcessor.sameVar(
                model,
                concept,
                subclasses,
                *entity_var_groups,
                onlyConstrains=headConstrain,
                logicMethodName=logicMethodName,
            )

            zVars.append([result])

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
        
    def __call__(self, model, myConstraintVarProcessor, v, headConstrain = False, integrate = False): 
        with torch.set_grad_enabled(myConstraintVarProcessor.grad):
            return self.createSingleVarLogicalConstrains("Not", myConstraintVarProcessor.notVar, model, v, headConstrain)

# ----------------- Logical

class andL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myConstraintVarProcessor, v, headConstrain = False, integrate = False): 
        with torch.set_grad_enabled(myConstraintVarProcessor.grad):
            return self.createLogicalConstrains('And', myConstraintVarProcessor.andVar, model, v, headConstrain)        

class orL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myConstraintVarProcessor, v, headConstrain = False, integrate = False):
        with torch.set_grad_enabled(myConstraintVarProcessor.grad):
            return self.createLogicalConstrains('Or', myConstraintVarProcessor.orVar, model, v, headConstrain)
    
class nandL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myConstraintVarProcessor, v, headConstrain = False, integrate = False): 
        with torch.set_grad_enabled(myConstraintVarProcessor.grad):
            return self.createLogicalConstrains('Nand', myConstraintVarProcessor.nandVar, model, v, headConstrain)
        
class ifL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
    
    def __call__(self, model, myConstraintVarProcessor, v, headConstrain = False, integrate = False): 
        with torch.set_grad_enabled(myConstraintVarProcessor.grad): #use_grad(myConstraintVarProcessor.grad):
            return self.createLogicalConstrains('If', myConstraintVarProcessor.ifVar, model, v, headConstrain)
    
class norL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
    
    def __call__(self, model, myConstraintVarProcessor, v, headConstrain = False, integrate = False): 
        with torch.set_grad_enabled(myConstraintVarProcessor.grad):
            return self.createLogicalConstrains('Nor', myConstraintVarProcessor.norVar, model, v, headConstrain)

class xorL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
    
    def __call__(self, model, myConstraintVarProcessor, v, headConstrain = False, integrate = False): 
        with torch.set_grad_enabled(myConstraintVarProcessor.grad):
            return self.createLogicalConstrains('Xor', myConstraintVarProcessor.xorVar, model, v, headConstrain)

class equivalenceL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
    
    def __call__(self, model, myConstraintVarProcessor, v, headConstrain = False, integrate = False):
        with torch.set_grad_enabled(myConstraintVarProcessor.grad): 
            return self.createLogicalConstrains('Equivalence', myConstraintVarProcessor.equivalenceVar, model, v, headConstrain)


class iffL(equivalenceL):
    """Bi-conditional logical constraint (A ↔ B).

    This is a user-facing alias for `equivalenceL`.
    Semantics: true when all provided operands have the same truth value.
    """
    pass

# ----------------- Counting

class _CountBaseL(LogicalConstrain):
    """
    Element-wise counting constraint.
    Sub-classes set `limitOp` ('<=', '>=', '==').
    Optionally set `fixedLimit` (int) to hard-code a limit that
    *cannot* be overridden by a trailing integer.
    """
    limitOp: str = None            # must be provided by subclass
    fixedLimit: int | None = None  # override in subclass for 'exists'-style LCs

    def __init__(self, *e, p=100, active=True, sampleEntries=False, name=None):
        super().__init__(*e, p=p, active=active,
                         sampleEntries=sampleEntries, name=name)

    def __call__(self,
                 model,
                 myConstraintVarProcessor,
                 v,
                 headConstrain=False,
                 integrate=False):

        # ── decide the numeric limit ───────────────────────────────────────
        if self.fixedLimit is not None:
            limit = self.fixedLimit
        else:
            limit = (
                self.e[-1] if (self.e and isinstance(self.e[-1], int)) else 1
            )

        with torch.set_grad_enabled(myConstraintVarProcessor.grad):
            return self.createCountConstraints(
                model,
                myConstraintVarProcessor,
                v,
                headConstrain,
                self.limitOp,
                limit,
                integrate,
                logicMethodName=str(self),
            )
      
class atMostL(_CountBaseL):      limitOp = "<="
class atLeastL(_CountBaseL):     limitOp = ">="
class exactL(_CountBaseL):       limitOp = "=="
class existsL(_CountBaseL):
    limitOp = ">="
    fixedLimit = 1

class oneOfL(_CountBaseL):
    """
    Mutual-exclusion + covering constraint between concepts (issue #371).

    Requires **exactly one** of the supplied concept applications to be true
    per candidate. Semantically equivalent to ``exactL(c1, c2, ..., 1)`` but:

    * the limit is pinned to ``1`` (``fixedLimit = 1``) so it cannot be
      accidentally overridden by a trailing integer, and
    * the name makes the "exactly one of these concepts" intent explicit —
      directly answering the question in the issue ("Defining ExactL
      between various concepts"). ``atMostL(c1, c2, c3)`` would allow zero
      of them to be true; ``oneOfL`` does not.

    Example — for every ``(step, entity)`` pair, exactly one of
    ``action_create``, ``action_destroy`` and ``action_move`` must hold::

        forAllL(
            combinationC(step, entity)('i', 'e'),
            ifL(
                action('x', path=(('i', action_step.reversed),
                                  ('e', action_entity.reversed))),
                oneOfL(action_create(path='x'),
                       action_destroy(path='x'),
                       action_move(path='x')),
            ),
        )
    """
    limitOp = "=="
    fixedLimit = 1

# ----------------- Accumulated Counting

class _AccumulatedCountBaseL(LogicalConstrain):
    """
    Global (accumulated) counting constraint.
    Same parameters as _CountBaseL.
    """
    limitOp: str = None
    fixedLimit: int | None = None

    def __init__(self, *e, p=100, active=True, sampleEntries=False, name=None):
        super().__init__(*e, p=p, active=active,
                         sampleEntries=sampleEntries, name=name)

    def __call__(self,
                 model,
                 myConstraintVarProcessor,
                 v,
                 headConstrain=False,
                 integrate=False):

        if self.fixedLimit is not None:
            limit = self.fixedLimit
        else:
            limit = (
                self.e[-1] if (self.e and isinstance(self.e[-1], int)) else 1
            )

        with torch.set_grad_enabled(myConstraintVarProcessor.grad):
            return self.createAccumulatedCountConstraints(
                model,
                myConstraintVarProcessor,
                v,
                headConstrain,
                self.limitOp,
                limit,
                integrate,
                logicMethodName=str(self),
            )
       
class atMostAL(_AccumulatedCountBaseL):   limitOp = "<="
class atLeastAL(_AccumulatedCountBaseL):  limitOp = ">="
class exactAL(_AccumulatedCountBaseL):    limitOp = "=="
class existsAL(_AccumulatedCountBaseL):   
    limitOp = ">="
    fixedLimit = 1 

class oneOfAL(_AccumulatedCountBaseL):
    """Accumulated-counting sibling of :class:`oneOfL` (issue #371).

    Enforces that exactly one of the supplied concept applications is true
    across all accumulated candidates (global scope), rather than per-row.
    """
    limitOp = "=="
    fixedLimit = 1

# -----------------  Comparative counting constraints (count(A) ∘ count(B)+diff)

class _CompareCountsBaseL(LogicalConstrain):
    """Base class – subclasses only need to set `compareOp`."""
    compareOp = None  # MUST be overridden: '>', '>=', '<', '<=', '==', '!='

    def __init__(self, *e, p=100, active=True, sampleEntries=False, name=None):
        super().__init__(*e, p=p, active=active,
                         sampleEntries=sampleEntries, name=name)

    def __call__(self,
                 model,
                 myConstraintVarProcessor,
                 v,
                 headConstrain=False,
                 integrate=False):
        # optional trailing int is the diff
        diff = self.e[-1] if (self.e and isinstance(self.e[-1], int)) else 0

        with torch.set_grad_enabled(myConstraintVarProcessor.grad):
            return self.createCompareCountsConstraints(
                model,
                myConstraintVarProcessor,
                v,
                headConstrain,
                self.compareOp,
                diff,
                integrate,
                logicMethodName=str(self),
            )

class greaterL(_CompareCountsBaseL):       compareOp = '>'
class greaterEqL(_CompareCountsBaseL):     compareOp = '>='
class lessL(_CompareCountsBaseL):          compareOp = '<'
class lessEqL(_CompareCountsBaseL):        compareOp = '<='
class equalCountsL(_CompareCountsBaseL):   compareOp = '=='     # not to confuse with eqL (path-equality)
class notEqualCountsL(_CompareCountsBaseL): compareOp = '!='

# ----------------- forall
class forAllL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myConstraintVarProcessor, v, headConstrain = False, integrate = False): 
        with torch.set_grad_enabled(myConstraintVarProcessor.grad):
            return self.createLogicalConstrains('If', myConstraintVarProcessor.ifVar, model, v, headConstrain)     
        
# ----------------- Auxiliary
     
class eqL(LogicalConstrain):
    def __init__(self, *e, active = True, sampleEntries = False, name = None):
        #if e is len 2 and element index 1 is of type String
        if len(e) == 2 and isinstance(e[1], str):
            e = (e[0],  "instanceID", e[1])  
        LogicalConstrain.__init__(self, *e, p=100)
        self.headLC = False
    
class fixedL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
        
    def __call__(self, model, myConstraintVarProcessor, v, headConstrain = False, integrate = False):
        with torch.set_grad_enabled(myConstraintVarProcessor.grad): 
            return self.createSingleVarLogicalConstrains("Fixed", myConstraintVarProcessor.fixedVar, model, v, headConstrain)        
        
# ----------------- Summation

class sumL(LogicalConstrain):
    def __init__(self, *e, p=100, active = True, sampleEntries = False, name = None):
        LogicalConstrain.__init__(self, *e, p=p, active=active, sampleEntries  = sampleEntries, name=name)
    
    def __call__(self, model, myConstraintVarProcessor, v, headConstrain = False, integrate = False, label=None):
        with torch.set_grad_enabled(myConstraintVarProcessor.grad): 
            return self.createSummation(model, myConstraintVarProcessor, v, headConstrain, integrate, label=label, logicMethodName='Summation')
        
#----------------- Definite Description
class iotaL(LogicalConstrain):
    """
    Definite description operator - selects THE unique entity satisfying a condition.
    
    From Russell's theory of definite descriptions:
    iota(var, expr) returns the entity that uniquely satisfies expr.
    
    Semantics:
        - Returns a probability distribution over entities (soft selection via softmax)
        - In ILP: enforces exactly one entity satisfies, returns selection variables
        - Presupposes existence and uniqueness of satisfying entity
    
    Usage:
        # Select THE person who works for Microsoft
        iotaL(person(V.x), path=(V.x, work_for, eqL(organization, 'name', 'Microsoft')))
        
        # Select THE sphere in the scene (assuming exactly one)
        iotaL(sphere(V.x))
        
        # Can be nested in other constraints
        # "Is there something left of THE blue sphere?"
        existsL(left(V.x, iotaL(andL(blue(V.y), sphere(V.y)))))
    
    Parameters:
        *e: Constraint elements defining the selection condition
        p: Priority (0-100, higher = more important)
        temperature: Softmax temperature for differentiable selection (lower = harder)
        active: Enable/disable constraint
        sampleEntries: Use sampling for large groundings
        name: Constraint name (auto-generated if None)
    
    Returns:
        - ILP: Selection indicator variables (one-hot among satisfying entities)
        - Loss: Entity distribution tensor [N] via softmax over satisfaction scores
        - Sample: Selected entity indices
        - Verify: Index of selected entity or -1 if violation
    
    Notes:
        - Unlike existsL which returns boolean, iotaL returns entity selection
        - Implicitly enforces uniqueness (exactly one should satisfy)
        - For soft selection during training, uses temperature-scaled softmax
        - Gradient flows through softmax for differentiable entity selection
    """
    
    def __init__(self, *e, p=100, temperature=1.0, active=True, 
                 sampleEntries=False, name=None):
        super().__init__(*e, p=p, active=active, 
                        sampleEntries=sampleEntries, name=name)
        self.temperature = temperature
        # Mark as returning entity selection rather than boolean
        self._returns_selection = True
        
    def __call__(self, model, myConstraintVarProcessor, v, 
                 headConstrain=False, integrate=False):
        with torch.set_grad_enabled(myConstraintVarProcessor.grad):
            return self.createIotaSelection(
                model,
                myConstraintVarProcessor,
                v,
                headConstrain,
                integrate,
                temperature=self.temperature,
                logicMethodName=str(self),
            )
    
class queryL(LogicalConstrain):
        """
        Query constraint for multiclass attribute concepts.
        
        Given a multiclass concept (parent with subclasses via is_a, or EnumConcept)
        and an entity selection (typically from iotaL or other constraints), returns the most probable
        subclass for the selected entity.
        
        Usage:
            # Define material as parent concept with subclasses
            material = object_node(name='material')
            metal = object_node(name='metal')
            rubber = object_node(name='rubber')
            metal.is_a(material)
            rubber.is_a(material)
            
            # Query: "What material is THE big sphere?"
            answer = queryL(
                material,
                iotaL(andL(big('x'), sphere(path='x')))
            )
        """
        
        def __init__(self, concept, *e, p=100, temperature=1.0, active=True,
                    sampleEntries=False, name=None):
            from domiknows.graph.concept import EnumConcept

            # Build concept variable bindings so the constraint constructor
            # collects per-entity subclass predictions into v.  This lets
            # queryVar see which subclass each entity actually has.
            attr_elements = []
            if isinstance(concept, EnumConcept):
                binding = concept('_ql_attr')
                if isinstance(binding, list):
                    attr_elements = [b for b in binding if b is not None]
            else:
                sub_idx = 0
                for rel in concept._in.get('is_a', []):
                    sub = rel.src
                    sub_binding = sub(f'_ql_sub_{sub_idx}')
                    sub_idx += 1
                    if isinstance(sub_binding, list):
                        attr_elements.extend(b for b in sub_binding if b is not None)

            super().__init__(*e, *attr_elements, p=p, active=active,
                            sampleEntries=sampleEntries, name=name)
            self.concept = concept
            self.temperature = temperature
            self._returns_value = True
            self._subclasses = None
            self._subclass_names = None
            self._init_subclasses()
    
        def _init_subclasses(self):
            """Initialize subclass information from concept."""
            from domiknows.graph.concept import EnumConcept
            
            if isinstance(self.concept, EnumConcept):
                self._subclass_names = list(self.concept.enum)
                self._subclasses = [(self.concept, name, i) 
                                for i, name in enumerate(self._subclass_names)]
            else:
                self._subclasses = []
                self._subclass_names = []
                
                for rel in self.concept._in.get('is_a', []):
                    subclass = rel.src
                    self._subclasses.append((subclass, subclass.name, len(self._subclasses)))
                    self._subclass_names.append(subclass.name)
                    
            if not self._subclasses:
                raise ValueError(f"Concept '{self.concept.name}' has no subclasses.")
        
        @property
        def num_subclasses(self):
            return len(self._subclasses)
        
        def get_subclass_name(self, index):
            if 0 <= index < len(self._subclass_names):
                return self._subclass_names[index]
            return None
        
        def __call__(self, model, myConstraintVarProcessor, v, 
                    headConstrain=False, integrate=False):
            with torch.set_grad_enabled(myConstraintVarProcessor.grad):
                return self.createQuerySelection(
                    model,
                    self.concept,        
                    self._subclasses,    
                    myConstraintVarProcessor, 
                    v, 
                    headConstrain, 
                    integrate,
                    temperature=self.temperature,
                    logicMethodName=str(self),
                )

class sameL(LogicalConstrain):
        """
        Same-attribute constraint for multiclass concepts.

        Given a multiclass concept (parent with subclasses via is_a, or EnumConcept)
        and entity variable names, returns true iff all entities share the same
        subclass value.

        Semantics:
            result = OR_j( AND_i( entity_i has subclass_j ) )

        Usage:
            color = EnumConcept(name='color', values=['red', 'blue', 'green'])

            # Check: "Do entities x and y have the same color?"
            sameL(color, 'x', 'y')

            # Typically used within a pair/relation context:
            ifL(right_of('x', 'y'), sameL(color, 'x', 'y'))
        """

        def __init__(self, concept, *e, p=100, active=True,
                    sampleEntries=False, name=None):

            # Convert string variable names to concept variable bindings
            converted = []
            for var in e:
                if isinstance(var, str):
                    binding = concept(var)  # e.g. color('x') -> [tuple, V, error_or_None]
                    if isinstance(binding, list):
                        converted.extend(b for b in binding if b is not None)
                    else:
                        converted.append(binding)
                else:
                    converted.append(var)

            super().__init__(*converted, p=p, active=active,
                            sampleEntries=sampleEntries, name=name)
            self.concept = concept
            self._returns_value = False
            self._subclasses = None
            self._subclass_names = None
            self._init_subclasses()

        def _init_subclasses(self):
            """Initialize subclass information from concept."""
            from domiknows.graph.concept import EnumConcept

            if isinstance(self.concept, EnumConcept):
                self._subclass_names = list(self.concept.enum)
                self._subclasses = [(self.concept, name, i)
                                for i, name in enumerate(self._subclass_names)]
            else:
                self._subclasses = []
                self._subclass_names = []

                for rel in self.concept._in.get('is_a', []):
                    subclass = rel.src
                    self._subclasses.append((subclass, subclass.name, len(self._subclasses)))
                    self._subclass_names.append(subclass.name)

            if not self._subclasses:
                raise ValueError(f"Concept '{self.concept.name}' has no subclasses.")

        def __call__(self, model, myConstraintVarProcessor, v,
                    headConstrain=False, integrate=False):
            with torch.set_grad_enabled(myConstraintVarProcessor.grad):
                return self.createSameSelection(
                    model,
                    self.concept,
                    self._subclasses,
                    myConstraintVarProcessor,
                    v,
                    headConstrain,
                    logicMethodName=str(self),
                )

class differentL(LogicalConstrain):
        """
        Different-attribute constraint for multiclass concepts.

        Negation of sameL: returns true iff NOT all entities share the same
        subclass value (i.e., at least one entity differs).

        Semantics:
            result = NOT( OR_j( AND_i( entity_i has subclass_j ) ) )

        Usage:
            color = EnumConcept(name='color', values=['red', 'blue', 'green'])

            # Check: "Do entities x and y have different colors?"
            differentL(color, 'x', 'y')
        """

        def __init__(self, concept, *e, p=100, active=True,
                    sampleEntries=False, name=None):

            # Convert string variable names to concept variable bindings
            converted = []
            for var in e:
                if isinstance(var, str):
                    binding = concept(var)
                    if isinstance(binding, list):
                        converted.extend(b for b in binding if b is not None)
                    else:
                        converted.append(binding)
                else:
                    converted.append(var)

            super().__init__(*converted, p=p, active=active,
                            sampleEntries=sampleEntries, name=name)
            self.concept = concept
            self._returns_value = False
            self._subclasses = None
            self._subclass_names = None
            self._init_subclasses()

        def _init_subclasses(self):
            """Initialize subclass information from concept."""
            from domiknows.graph.concept import EnumConcept

            if isinstance(self.concept, EnumConcept):
                self._subclass_names = list(self.concept.enum)
                self._subclasses = [(self.concept, name, i)
                                for i, name in enumerate(self._subclass_names)]
            else:
                self._subclasses = []
                self._subclass_names = []

                for rel in self.concept._in.get('is_a', []):
                    subclass = rel.src
                    self._subclasses.append((subclass, subclass.name, len(self._subclasses)))
                    self._subclass_names.append(subclass.name)

            if not self._subclasses:
                raise ValueError(f"Concept '{self.concept.name}' has no subclasses.")

        def __call__(self, model, myConstraintVarProcessor, v,
                    headConstrain=False, integrate=False):
            with torch.set_grad_enabled(myConstraintVarProcessor.grad):
                same_result = self.createSameSelection(
                    model,
                    self.concept,
                    self._subclasses,
                    myConstraintVarProcessor,
                    v,
                    headConstrain,
                    logicMethodName=str(self),
                )
                # Negate each result: differentL = NOT(sameL)
                negated = []
                for row in same_result:
                    neg_row = []
                    for val in row:
                        if val is None:
                            neg_row.append(None)
                        else:
                            neg_row.append(
                                myConstraintVarProcessor.notVar(model, val))
                    negated.append(neg_row)
                return negated

class execute:
    """
    Wrapper for logical constraints that marks them as executable rather than standard constraints.
    
    When a LogicalConstrain is wrapped with execute(), it is moved from graph.logicalConstrains
    to graph.executableLCs. This allows for different processing of executable constraints
    vs standard logical constraints.
    
    Usage:
        constraint1 = execute(andL(p1("x"), p2("y")))
        
    The wrapped constraint will be accessible via graph.executableLCs instead of 
    graph.logicalConstrains.
    """
    
    def __init__(self, lc, name=None):
        """
        Initialize an executable constraint wrapper.
        
        Args:
            lc: LogicalConstrain instance to wrap
            name: Optional name for this executable constraint
        """
        if not isinstance(lc, LogicalConstrain):
            raise TypeError(f"execute() requires a LogicalConstrain instance, got {type(lc).__name__}")
        
        self.innerLC = lc
        self.graph = lc.graph
        
        # Remove from logicalConstrains (it was added there during LogicalConstrain.__init__)
        if lc.lcName in self.graph.logicalConstrains:
            del self.graph.logicalConstrains[lc.lcName]
        
        # Generate name for executable constraint
        self.lcName = "ELC%i" % (len(self.graph.executableLCs))
        
        if name is not None:
            self.name = name
        else:
            self.name = self.lcName
        
        # Update the wrapped constraint's name reference
        self.innerLC.lcName = self.lcName
        self.innerLC.name = self.name
        
        # Add to executableLCs
        self.graph.executableLCs[self.lcName] = self
        
        # Copy relevant attributes from wrapped constraint
        self.headLC = self.innerLC.headLC
        self.active = self.innerLC.active
        self.p = self.innerLC.p
        self.e = self.innerLC.e
        self.sampleEntries = self.innerLC.sampleEntries
    
    def __str__(self):
        return self.lcName
    
    def __repr__(self):
        return f"{self.lcName}(execute<{self.innerLC.__class__.__name__}>)"
    
    def __call__(self, model, myConstraintVarProcessor, v, headConstrain=False, integrate=False):
        """Delegate to wrapped constraint's __call__ method."""
        return self.innerLC(model, myConstraintVarProcessor, v, headConstrain, integrate)
    
    def strEs(self):
        """Delegate to wrapped constraint's strEs method."""
        return self.innerLC.strEs()
    
    def getLcConcepts(self):
        """Delegate to wrapped constraint's getLcConcepts method."""
        return self.innerLC.getLcConcepts()