import math
from collections import OrderedDict
import torch

from domiknows.graph.concept import Concept, EnumConcept
from domiknows.graph import LcElement, LogicalConstrain, V
from domiknows.graph import CandidateSelection
from domiknows.graph.candidates import getCandidates
from domiknows.graph.logicalConstrain import sumL


class LogicalConstraintConstructor:
    """
    Helper class for constructing logical constraints.
    
    This class handles the construction of logical constraints by processing
    concepts, variables, and nested constraints. It's independent of the ILP
    solver and can be used by various functionalities (ILP inference, loss
    calculation, verification, etc.).
    """
    
    def __init__(self, logger):
        """
        Initialize the constraint constructor.
        
        Args:
            logger: Logger instance for debugging/info messages
        """
        self.myLogger = logger
        self.current_device = None
        self.current_dtype = None
        
    def _get_dtype(self):
        """Get current dtype, defaulting to float32 if not yet detected."""
        if self.current_dtype is not None:
            return self.current_dtype
        return torch.float32  # Default to float32 (standard neural network output)
    
    def _detect_dtype_from_datanode(self, dn, xPkey):
        """
        Detect and set dtype from datanode.
        
        First checks if datanode has current_dtype attribute set (from builder).
        If not, tries to detect from attribute values.
        
        Args:
            dn: Datanode to detect dtype from
            xPkey: Key for accessing predictions
        """
        # First try to use dtype from datanode if available
        if hasattr(dn, 'current_dtype') and dn.current_dtype is not None:
            self.current_dtype = dn.current_dtype
            self.myLogger.info(f'Using dtype {self.current_dtype} from datanode')
            return
        
        # Fallback: try to detect from attributes
        if dn.getAttribute(xPkey) is not None:
            attr_value = dn.getAttribute(xPkey)
            if torch.is_tensor(attr_value):
                self.current_dtype = attr_value.dtype
                self.myLogger.info(f'Detected dtype {self.current_dtype} from datanode attribute')
                return
        
        # If still not detected, keep None (will use default in _get_dtype)
        self.myLogger.debug('Could not detect dtype from datanode, will use default')
        
    def getConcept(self, concept):
        return concept[0]
    
    def getConceptName(self, concept):
        return concept[0].name
    
    def conceptIsBinary(self, concept):
        return concept[2] is None
    
    def conceptIsMultiClass(self, concept):
        return concept[2] is not None
    
    def valueToBeSkipped(self, x):
        """Check if value is NaN or Inf and should be skipped"""
        # Detach tensor if needed to avoid autograd warnings
        if isinstance(x, torch.Tensor):
            x = x.detach().item()
        return math.isnan(x) or math.isinf(x)
    
    def getLabel(self, dn, conceptRelation):
        """Get Ground Truth for provided concept"""
        value = dn.getAttribute(conceptRelation, 'label')
        return value
    
    def getDatanodesForConcept(self, rootDn, currentName, conceptToDNSCash=None):
        if conceptToDNSCash is None or currentName is None:
            conceptToDNSCash = {}
            if currentName is None:
                return  # Just reset cash
            
        if currentName in conceptToDNSCash:
            dns = conceptToDNSCash[currentName]
        else:
            rootConcept = rootDn.findRootConceptOrRelation(currentName)
            dns = rootDn.findDatanodes(select=rootConcept)
            conceptToDNSCash[currentName] = dns
            
        return dns
    
    @staticmethod
    def _circuit_probability(value, class_index):
        """Return one scalar class probability without detaching autograd."""
        if not torch.is_tensor(value):
            return torch.as_tensor(value, dtype=torch.get_default_dtype()).reshape(())
        squeezed = value.squeeze()
        if squeezed.numel() == 1:
            return squeezed.reshape(())
        return squeezed.reshape(-1)[int(class_index)]

    def _circuit_leaf(self, dn, xPkey, e, concept):
        """Create a stable, categorical-aware circuit leaf handle."""
        from domiknows.solver.circuitBooleanMethods import CircuitLeaf

        concept_name, label_index, class_index = e
        instance_id = dn.getInstanceID()
        raw = dn.getAttribute(xPkey)
        fixed_value = self.isVariableFixed(dn, concept_name, e)

        if isinstance(concept, EnumConcept):
            values = raw.squeeze().reshape(-1)
            domain_size = len(concept.enum)
            if values.numel() < domain_size:
                raise ValueError(
                    f"EnumConcept {concept_name!r} exposes {values.numel()} probabilities "
                    f"for a {domain_size}-value domain"
                )
            probabilities = tuple(values[index] for index in range(domain_size))
            variable_key = ("categorical", concept_name, instance_id)
            value_index = int(label_index)
            probability = probabilities[value_index]
            categorical = True
        else:
            # is_a sibling concepts are one categorical variable when a parent
            # has multiple subclasses.  Gather all sibling true-probabilities
            # now so a formula mentioning only one sibling still has a complete
            # categorical distribution for WMC.
            categorical_parent = None
            siblings = None
            for relation in getattr(concept, "_out", {}).get("is_a", []):
                parent = relation.dst
                candidates = [rel.src for rel in getattr(parent, "_in", {}).get("is_a", [])]
                # A property-family parent (e.g. object -> material ->
                # {metal,rubber}) is itself an is_a child.  A root/container
                # concept (e.g. entity -> {selected,material}) merely owns
                # independent predicates and must not make them one-hot.
                parent_is_property_family = bool(
                    getattr(parent, "_out", {}).get("is_a", [])
                )
                if len(candidates) > 1 and parent_is_property_family:
                    categorical_parent = parent
                    siblings = candidates
                    break

            if categorical_parent is not None:
                probabilities_list = []
                for sibling in siblings:
                    sibling_key = f"<{sibling.name}>" + xPkey[xPkey.index("/") :]
                    sibling_raw = dn.getAttribute(sibling_key)
                    if sibling_raw is None:
                        raise ValueError(
                            f"Missing probability for categorical sibling {sibling.name!r}"
                        )
                    sibling_values = sibling_raw.squeeze().reshape(-1)
                    sibling_probability = sibling_values[-1]
                    probabilities_list.append(sibling_probability)
                probabilities = tuple(probabilities_list)
                value_index = siblings.index(concept)
                probability = probabilities[value_index]
                variable_key = ("categorical", categorical_parent.name, instance_id)
                categorical = True
            else:
                probability = self._circuit_probability(raw, label_index)
                probabilities = (1.0 - probability, probability)
                variable_key = ("binary", (concept_name, instance_id, int(class_index)))
                value_index = 1
                categorical = False

        if fixed_value is not None:
            # isVariableFixed historically compares binary labels with the
            # storage index (0), whereas this handle denotes the positive
            # class (label 1). Recover the literal truth from the label value.
            if not isinstance(concept, EnumConcept):
                label = self.getLabel(dn, concept_name)
                if label is not None:
                    label_value = int(label.detach().reshape(-1)[0].item())
                    fixed_value = int(label_value == int(label_index))
            fixed_value = int(fixed_value)

        return CircuitLeaf(
            key=(concept_name, instance_id, int(class_index)),
            probability=probability,
            variable_key=variable_key,
            value_index=value_index,
            probabilities=probabilities,
            categorical=categorical,
            fixed_value=fixed_value,
        )

    def getMLResult(self, dn, xPkey, e, p, loss=False, sample=False,
                    circuit=False, concept=None):
        """
        Get ML result for a datanode and concept.
        
        Args:
            dn: Datanode
            xPkey: Key for accessing predictions
            e: Concept tuple (concept_name, label, index)
            p: Sample size (for sampling) or priority (for ILP)
            loss: Whether calculating loss
            sample: Whether generating samples
            circuit: Whether returning a stable exact-circuit leaf handle
            concept: Concrete Concept object used to recover categorical groups
            
        Returns:
            For ILP: ILP variable
            For loss without sample: Tensor value
            For loss with sample: Tuple of (sample, (probability, sample, variable_name))
        """
        if dn == None:
            raise Exception("No datanode provided")
        
        # Detect dtype early from datanode before creating any tensors
        if loss and self.current_dtype is None:
            self._detect_dtype_from_datanode(dn, xPkey)
                
        conceptName = e[0]
        
        sampleKey = '<' + conceptName + ">/sample" 
        if sample and sampleKey not in dn.getAttributes():
            dn.getAttributes()[sampleKey] = {}
        
        if dn.ontologyNode.name == conceptName:
            if circuit:
                return True
            if not sample:
                if "xP" in xPkey:
                    return 1
                elif loss:
                    tOne = torch.ones(1, device=self.current_device, requires_grad=True, dtype=self._get_dtype())
                else:
                    tOne = torch.ones(1, device=self.current_device, requires_grad=False, dtype=self._get_dtype())
                    
                tOneSqueezed = torch.squeeze(tOne)
                return tOneSqueezed
            else:
                sampleSize = p

                # Semantic sampling stores the complete assignment table under
                # the ``-1`` key.  Reuse it instead of trying to allocate a
                # tensor with a negative dimension.
                if sampleSize == -1:
                    sample_values = dn.getAttributes().get(sampleKey, {}).get(-1, {}).get(e[1])
                    if sample_values is None:
                        semantic_sample_size = getattr(self, 'semantic_sample_size', None)
                        if semantic_sample_size is None:
                            raise RuntimeError(
                                'Semantic sample size is unavailable for a structural node.'
                            )
                        sample_values = torch.ones(
                            semantic_sample_size,
                            dtype=torch.bool,
                            device=self.current_device,
                        )
                    xVarName = "%s_%s_is_%s" % (e[0], dn.getInstanceID(), e[1])
                    xP = torch.ones(
                        sample_values.shape[0],
                        device=self.current_device,
                        dtype=self._get_dtype(),
                    )
                    return (sample_values, (xP, sample_values, xVarName))

                if sampleSize not in dn.getAttributes()[sampleKey]: 
                    dn.getAttributes()[sampleKey][sampleSize] = {}
                    
                xVarName = "%s_%s_is_%s"%(e[0], dn.getInstanceID(), e[1])

                dn.getAttributes()[sampleKey][sampleSize][e[1]] = torch.ones(sampleSize, dtype=torch.bool, device=self.current_device)
                xP = torch.ones(sampleSize, device=self.current_device, dtype=self._get_dtype())
                
                return (dn.getAttributes()[sampleKey][sampleSize][e[1]], (xP, dn.getAttributes()[sampleKey][sampleSize][e[1]], xVarName))
        
        if dn.getAttribute(xPkey) == None:
            if not sample:
                return None
            else:   
                return ([None], (None, [None]))

        if circuit:
            if concept is None:
                raise ValueError("Circuit leaf construction requires the Concept object")
            return self._circuit_leaf(dn, xPkey, e, concept)
        
        if not loss:
            if "xP" in xPkey:
                vDn = dn.getAttribute(xPkey)[p][e[2]]
            elif "local/argmax" in xPkey:
                vDn = dn.getAttribute(xPkey)[e[1]]
            else:
                vDn = dn.getAttribute(xPkey)[e[2]]
                
            return vDn
        
        # Loss calculation
        isFiexd = self.isVariableFixed(dn, conceptName, e)

        if isFiexd != None:
            if isFiexd == 1:
                vDn = torch.tensor(1.0, device=self.current_device, requires_grad=True, dtype=self._get_dtype())
            else:
                vDn = torch.tensor(0.0, device=self.current_device, requires_grad=True, dtype=self._get_dtype())
        else:
            try:
                vDn = dn.getAttribute(xPkey)[e[1]]
                if torch.is_tensor(vDn):
                    self.current_dtype = vDn.dtype
            except IndexError: 
                vDn = None
    
        if not sample:
            return vDn
        
        if torch.is_tensor(vDn) and (len(vDn.shape) == 0 or len(vDn.shape) == 1 and vDn.shape[0] == 1):
            vDn = vDn.item()  
             
        sampleSize = p

        xVarName = "%s_%s_is_%s"%(e[0], dn.getInstanceID(), e[1])
                
        usedSampleSize = sampleSize
        if sampleSize == -1:
            usedSampleSize = dn.getAttributes()[sampleKey][-1][e[1]].shape[0]
        if isFiexd != None:  
            if isFiexd == 1:
                xP = torch.ones(usedSampleSize, device=self.current_device, requires_grad=True, dtype=self._get_dtype())
            else:
                xP = torch.zeros(usedSampleSize, device=self.current_device, requires_grad=True, dtype=self._get_dtype())
        else:
            xV = dn.getAttribute(xPkey)
            xEp = dn.getAttribute(xPkey).expand(usedSampleSize, len(xV.squeeze(0)))
            xP = xEp[:,e[1]]
          
        if sampleSize > -1: 
            if sampleSize not in dn.getAttributes()[sampleKey]: 
                dn.getAttributes()[sampleKey][sampleSize] = {}
                
            if e[1] not in dn.getAttributes()[sampleKey][sampleSize]:
                if vDn == None or vDn != vDn:
                    dn.getAttributes()[sampleKey][sampleSize][e[1]] = [None]
                else:
                    dn.getAttributes()[sampleKey][sampleSize][e[1]] = torch.bernoulli(xP)
            
        return (dn.getAttributes()[sampleKey][sampleSize][e[1]], (xP, dn.getAttributes()[sampleKey][sampleSize][e[1]], xVarName))
    
    def isVariableFixed(self, dn, conceptName, e):
        """Check if a variable is fixed by fixedL constraint"""
        fixedAttribute = None
        fixedValue = None
        
        if not hasattr(self, 'myGraph'):
            return None
            
        from domiknows.graph import fixedL
        
        for graph in self.myGraph:
            for _, lc in graph.allLogicalConstrains:
                if not lc.headLC or not lc.active:
                    continue
                    
                if type(lc) is not fixedL:
                    continue
                
                if not lc.e:
                    continue
                
                if lc.e[0][1] != conceptName:
                    continue
                
                fixedAttribute = lc.e[1].v[1].e[1]
                fixedValue = lc.e[1].v[1].e[2]
                break
                
        if fixedAttribute == None or fixedValue == None:
            return None
                      
        if fixedAttribute not in dn.getAttributes():
            return None
        
        attributeValue = dn.getAttribute(fixedAttribute).item()
        
        if attributeValue in fixedValue:
            pass
        elif (True in fixedValue) and attributeValue == 1:
            pass
        elif (False in fixedValue) and attributeValue == 0:
            pass
        else:
            return None
       
        vDnLabel = self.getLabel(dn, conceptName).item()

        if vDnLabel == e[2]:
            return 1
        else:
            return 0
    
    def fixedLSupport(self, _dn, conceptName, vDn, i, m):
        """Support for fixed constraints"""
        from gurobipy import Var
        
        vDnLabel = self.getLabel(_dn, conceptName).item()

        if isinstance(vDn, Var):                                 
            if vDnLabel == -100:
                vDn.VTag = "None" + vDn.VarName
            elif vDnLabel == i:
                vDn.VTag = "True" + vDn.VarName
            else:
                vDn.VTag = "False" + vDn.VarName
                
            m.update()
            return vDn
        elif torch.is_tensor(vDn):
            if vDnLabel == -100:
                return None
            elif vDnLabel == i:
                ones = torch.ones(vDn.shape[0])
                return ones
            else:
                zeros = torch.zeros(vDn.shape[0])
                return zeros
        else:
            if vDnLabel == -100:
                return None
            elif vDnLabel == i:
                return 1
            else:
                return 0
    
    @staticmethod
    def lossVariableWidth(e):
        """How many values a concept element contributes per candidate.

        A bare ``EnumConcept`` reference (``e[2] is None``) contributes one
        value per class; everything else contributes a single value.
        """
        concept = e[0]
        if isinstance(concept, EnumConcept) and e[2] is None:
            return len(concept.enum)
        return 1

    @staticmethod
    def stackLossColumns(vDns, width=1):
        """Batch N candidate groups into one group of ``width`` ``[N]`` tensors.

        Loss mode batches a variable across all of its groundings. A binary or
        class-indexed concept contributes one value per group, which yields the
        familiar single ``[[tensor[N]]]`` layout. A *bare* ``EnumConcept``
        contributes K values per group (one per class) and all K have to
        survive as separate entries, because consumers index them by class —
        ``sameVar`` reads ``group[j]`` per subclass and
        ``_collect_query_subclass_data`` documents "each row already contains K
        values". Keeping only ``v[0]`` silently reduces every such constraint to
        "is the first class", which is what the non-loss backends (ILP, verify,
        exact circuit) never did.

        Returns None when the groups cannot be stacked (a ``None`` candidate or
        a group narrower than ``width``), so the caller can fall back to the
        per-group layout.
        """
        if width < 1:
            return None
        try:
            columns = [torch.stack([v[j] for v in vDns], dim=0) for j in range(width)]
        except (TypeError, IndexError):
            return None
        return [columns]

    @classmethod
    def fillPathBindings(cls, useLcVariables, variableVs, lcVariablesDns, bindings):
        """Give path-derived variables the grounding of the variable they walk from.

        ``big(path=('right_of_0', arg1))`` is enumerated row-for-row alongside
        ``right_of_0``, so it shares that variable's grounding. It only varies
        along the argument it projects, which ``reduceToCommonGrounding``
        discovers numerically (its rows are constant along the other axis).
        Runs to a fixpoint so chained paths resolve.
        """
        for _ in range(len(useLcVariables) + 1):
            progressed = False
            for name in useLcVariables:
                if name in bindings or name not in lcVariablesDns:
                    continue
                variable = variableVs.get(name)
                path = getattr(variable, 'v', None) if variable is not None else None
                if not (isinstance(path, tuple) and path and isinstance(path[0], str)):
                    continue
                source = bindings.get(path[0])
                if source is None:
                    continue
                if len(lcVariablesDns[name]) != len(source[1]):
                    continue
                bindings[name] = source
                progressed = True
            if not progressed:
                break
        return bindings

    @staticmethod
    def groundingBinding(variable, dnsList, lcVariablesDns):
        """Which logical variables a candidate set ranges over, and per-row indices.

        Returns ``(names, keys)`` where ``names`` are the constraint's logical
        variable names (e.g. ``('z', 'x')`` for ``right_of('z','x')``) and
        ``keys[r]`` holds that row's index into each of those variables'
        candidate lists. Returns None when the grounding cannot be determined,
        in which case callers must leave the variable alone.

        A relation variable is enumerated as a nested loop over its source and
        destination candidates (see ``getDatanoteForVariable``), so row ``r``
        decomposes arithmetically into ``(r // n_dest, r % n_dest)``.
        """
        if variable is None:
            return None
        rows = len(dnsList)
        if rows == 0:
            return None

        relVarInfo = getattr(variable, 'relVarInfo', None)
        if relVarInfo:
            names = tuple(relVarInfo.keys())
            if len(names) != 2 or any(n not in lcVariablesDns for n in names):
                return None
            n_src = len(lcVariablesDns[names[0]])
            n_dest = len(lcVariablesDns[names[1]])
            if not n_dest or rows != n_src * n_dest:
                return None
            return names, [(r // n_dest, r % n_dest) for r in range(rows)]

        # A plain logical variable ranges over itself, one row per candidate.
        if getattr(variable, 'name', None) and getattr(variable, 'v', None) is None:
            return (variable.name,), [(r,) for r in range(rows)]

        return None

    @staticmethod
    def reduceToCommonGrounding(useLcVariables, bindings, booleanProcessor):
        """Existentially quantify each operand down to the shared variables.

        Two relations that share a variable — ``andL(right_of('z','x'),
        left_of('z','y'))`` — are enumerated over different tuples, ``(z,x)``
        and ``(z,y)``. Multiplying them row-by-row silently forces ``x == y``,
        because both enumerations happen to place ``z`` on the same axis. The
        correct reading quantifies the unshared variables away first::

            phi(z) = (exists x. right_of(z,x)) and (exists y. left_of(z,y))

        so every operand is reduced onto the variables common to all of them and
        the conjunction is then well-posed. Existential quantification is OR over
        the quantified axis, using the active t-norm. When an operand does not
        actually vary along the axis (its rows are duplicates, as for a predicate
        on ``z`` that expansion replicated across ``x``) the group is collapsed by
        taking one representative instead — OR-ing identical values would inflate
        them under a non-idempotent t-norm.

        No-ops unless at least two operands have known, *differing* variable
        sets, so co-grounded constraints keep their exact current behaviour.
        """
        bound = {n: bindings[n] for n in useLcVariables if n in bindings}
        varSets = {n: set(b[0]) for n, b in bound.items()}
        if len(bound) < 2 or len({frozenset(s) for s in varSets.values()}) < 2:
            return useLcVariables

        common = set.intersection(*varSets.values())
        if not common:
            return useLcVariables

        # Any operand we cannot place in the shared frame would be left at the
        # wrong length; rather than guess, decline the whole reduction.
        for name, groups in useLcVariables.items():
            if name in bound:
                continue
            if not (groups and len(groups) == 1 and len(groups[0]) >= 1
                    and torch.is_tensor(groups[0][0]) and groups[0][0].numel() == 1):
                return useLcVariables

        reduced = OrderedDict()
        commonOrder = None
        for name, groups in useLcVariables.items():
            if name not in bound:
                reduced[name] = groups  # scalar: broadcasts, nothing to align
                continue

            names, keys = bound[name]
            keepIdx = [i for i, v in enumerate(names) if v in common]
            buckets = OrderedDict()
            for row, key in enumerate(keys):
                buckets.setdefault(tuple(key[i] for i in keepIdx), []).append(row)

            if commonOrder is None:
                commonOrder = list(buckets.keys())

            columns = []
            for column in groups[0]:
                values = []
                for bucketKey in commonOrder:
                    rowsInBucket = buckets.get(bucketKey, [])
                    if not rowsInBucket:
                        values.append(torch.zeros((), dtype=column.dtype, device=column.device))
                        continue
                    picked = column[rowsInBucket]
                    if picked.numel() == 1 or bool(torch.allclose(picked, picked[:1].expand_as(picked))):
                        values.append(picked[0])  # constant along the axis
                    else:
                        values.append(booleanProcessor.orVar(
                            None, *[p.reshape(1) for p in picked]).reshape(()))
                columns.append(torch.stack(values, dim=0))
            reduced[name] = [columns]

        return reduced

    @staticmethod
    def splitLossColumns(variable):
        """Tear a single batched group back into one group per row.

        Loss mode keeps a variable as one group of ``K`` ``[N]`` tensors. When a
        sibling variable ends up with multiple groups, every single-group
        variable has to be split along the row axis so all operands agree on the
        row count. All K columns must be split *in parallel* — splitting only
        column 0 would silently re-collapse a bare ``EnumConcept`` to its first
        class, which is the very defect this layout exists to avoid.
        """
        group = variable[0]
        splits = [torch.split(column, 1) for column in group]
        rows = len(splits[0]) if splits else 0
        return [[column_split[row] for column_split in splits] for row in range(rows)]

    def addLossTovDns(self, loss, vDns):
        """Add loss tensor to vDns.
        
        Handles tensors of different sizes that occur when processing
        counting constraints (atLeastL, atMostL, exactL) wrapping sumL.
        These constraints produce scalar results while nested element-wise
        constraints produce multi-element tensors.
        """
        if loss and vDns:
            vDnsList = [
                v[0] for v in vDns
                if v and len(v) > 0 and v[0] is not None
            ]
            if not vDnsList:
                return vDns
            
            updatedVDns = []
            try:
                if len(vDnsList) > 1:
                    # Check if all tensors have the same size
                    sizes = []
                    for v in vDnsList:
                        if torch.is_tensor(v):
                            sizes.append(v.numel())
                        else:
                            sizes.append(1)
                    
                    if len(set(sizes)) > 1:
                        # Mixed sizes - flatten and concatenate instead of stacking
                        flat_tensors = []
                        for v in vDnsList:
                            if torch.is_tensor(v):
                                flat_tensors.append(v.flatten())
                            elif v is not None:
                                flat_tensors.append(torch.tensor([v], 
                                    device=self.current_device, 
                                    dtype=self._get_dtype(), 
                                    requires_grad=True))
                        if flat_tensors:
                            tsqueezed = torch.cat(flat_tensors, dim=0)
                        else:
                            return vDns
                    else:
                        # Same sizes - use original stacking logic
                        tStack = torch.stack(vDnsList, dim=1)
                        tsqueezed = torch.squeeze(tStack, dim=0)
                else:
                    tStack = vDnsList[0]
                    tsqueezed = torch.squeeze(tStack, dim=0) if torch.is_tensor(tStack) else tStack

            except (IndexError, RuntimeError, TypeError):
                # Fallback: try to concatenate flattened tensors
                flat_tensors = []
                for v in vDnsList:
                    if torch.is_tensor(v):
                        flat_tensors.append(v.flatten())
                    elif v is not None:
                        flat_tensors.append(torch.tensor([v], 
                            device=self.current_device, 
                            dtype=self._get_dtype(), 
                            requires_grad=True))
                if flat_tensors:
                    tsqueezed = torch.cat(flat_tensors, dim=0)
                else:
                    return vDns
        
            if torch.is_tensor(tsqueezed) and not len(tsqueezed.shape):
                tsqueezed = torch.unsqueeze(tsqueezed, 0)
                
            tList = [tsqueezed]
            updatedVDns.append(tList)
            
            return updatedVDns
        else:
            return vDns
    
    def eliminate_duplicate_columns(self, data_dict, rows_to_consider, data_dict_target):
        """Eliminates columns that have identical elements across specified rows."""
        if not rows_to_consider or not data_dict or not data_dict_target:
            return data_dict_target
        
        first_row = list(data_dict.values())[0]
        num_columns = len(first_row)
        
        columns_to_keep = []
        
        for col_idx in range(num_columns):
            column_values = []
            for row_name in rows_to_consider:
                if row_name in data_dict:
                    if col_idx >= len(data_dict[row_name]):
                        continue
                    column_values.append(data_dict[row_name][col_idx])
            
            unique_values = set(str(val) for val in column_values)
            if len(unique_values) < len(column_values):
                pass
            else:
                columns_to_keep.append(col_idx)
        
        result = OrderedDict()
        for row_name, row_data in data_dict_target.items():
            if len(row_data) == 1:
                try:
                    # A batched group may hold several class columns; filter the
                    # kept rows out of every one of them, not just the first.
                    result[row_name] = [[column[columns_to_keep] for column in row_data[0]]]
                except (TypeError, IndexError):
                    result[row_name] = [row_data[i] for i in columns_to_keep]
            else:
                result[row_name] = [row_data[i] for i in columns_to_keep]
        
        return result

    def constructLogicalConstrains(self, lc, booleanProcessor, m, dn, p, key=None,
                                   lcVariablesDns=None, lcVariables=None, headLC=False, 
                                   loss=False, sample=False, vNo=None, verify=False, label=None,
                                   circuit=False):
        """
        Construct logical constraints by processing concepts and variables.
        
        Args:
            lc: Logical constraint to construct
            booleanProcessor: Boolean processor for constraint operations
            m: Model (ILP model or None for loss/verify)
            dn: Root datanode
            p: Sample size (for sampling) or priority (for ILP)
            key: Key for accessing predictions
            lcVariablesDns: Dictionary mapping variable names to datanodes
            lcVariables: Dictionary mapping variable names to values/variables
            headLC: Whether this is a head constraint
            loss: Whether calculating loss
            sample: Whether generating samples
            vNo: Variable numbering counter [concept_counter, lc_counter]
            verify: Whether verifying constraints
            labels: Optional labels for the constraint
            circuit: Return stable leaf handles for an exact circuit backend
            
        Returns:
            For sample=True: (result, sampleInfo, lcVariablesSet, lcVariables)
            For verify=True and headLC=True: (result, lcVariables)
            Otherwise: (result, lcVariables)
        """
        if key == None:
            key = ""
            
        lcRepr = f'{lc.__class__.__name__} {lc.strEs()}'

        if lcVariablesDns == None:
            lcVariablesDns = OrderedDict()

        if lcVariables == None:
            lcVariables = OrderedDict()
            
        usedVariablesNames = set()
        # Which logical variables each candidate set ranges over, used to align
        # operands that were enumerated over different tuples (see
        # reduceToCommonGrounding).
        lcVariableBindings = OrderedDict()
        lcVariableVs = OrderedDict()

        if sample:
            sampleInfo = OrderedDict()
            lcVariablesSet = OrderedDict()
            
        if vNo == None:
            vNo = [1, 1]
        
        firstV = None
        integrate = False
        newVariables = {}

        iter_es = lc.e

        for eIndex, e in enumerate(iter_es):
            if isinstance(e, V):
                continue
            
            if isinstance(e, (Concept, LcElement, tuple)): 
                # Look ahead for variable names
                if eIndex + 1 < len(iter_es) and isinstance(iter_es[eIndex+1], V):
                    variable = iter_es[eIndex+1]
                else:
                    if isinstance(e, LogicalConstrain):
                        variable = V(name="_lc" + str(vNo[1]))
                        vNo[1] += 1
                    elif isinstance(e, tuple) and isinstance(e[0], CandidateSelection):
                        e[0].CandidateSelectionVariable = e[1]
                        e = e[0]
                        variable = V(name="_cs" + str(vNo[1]))
                        vNo[1] += 1
                    else:
                        if firstV == None:
                            variable = V(name="_x" + str(vNo[0]))
                            if not isinstance(lc, CandidateSelection):
                                firstV = variable.name
                            vNo[0] += 1
                        else:
                            variable = V(name="_x" + str(vNo[0]), v=(firstV,))
                            vNo[0] += 1
                    
                if variable.name:
                    variableName = variable.name
                else:
                    variableName = "V" + str(vNo[0])
                    vNo[0] += 1
                    
                if variableName in lcVariables:
                    newVariableName = "_x" + str(vNo[0])
                    vNo[0] += 1

                    lcVariablesDns[newVariableName] = lcVariablesDns[variableName]

                    # When the current element is a Concept/tuple with a different concept
                    # than what the variable was originally bound to, we need to look up
                    # the NEW concept's ILP variables on the same datanodes rather than
                    # copying the old concept's values.
                    # E.g., in ifL(word('x'), exactL(people('x'), org('x'), ...)),
                    # people('x') should get <people>/ILP/x, not <word>/ILP/x values.
                    is_concept_tuple = isinstance(e, tuple) and len(e) >= 1 and isinstance(e[0], Concept)
                    if is_concept_tuple:
                        conceptName = e[0].name
                        xPkey = '<' + conceptName + ">" + key
                        dnsList = lcVariablesDns[variableName]
                        vDns = []
                        if sample:
                            sampleInfoForVariable = []

                        for dns in dnsList:
                            _vDns = []
                            if sample:
                                _sampleInfoForVariable = []

                            for _dn in dns:
                                if not _dn:
                                    _vDns.append(None)
                                    continue

                                if isinstance(e[0], EnumConcept) and e[2] == None:
                                    eList = e[0].enum
                                    for i, _ in enumerate(eList):
                                        eT = (e[0].name, i, i)
                                        if sample:
                                            vDn, vDnSampleInfo = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample, circuit=circuit, concept=e[0])
                                            _sampleInfoForVariable.append(vDnSampleInfo)
                                        else:
                                            vDn = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample, circuit=circuit, concept=e[0])
                                        _vDns.append(vDn)
                                elif isinstance(e[0], EnumConcept) and e[2] != None:
                                    eT = (e[0].name, e[2], e[2])
                                    if sample:
                                        vDn, vDnSampleInfo = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample, circuit=circuit, concept=e[0])
                                        _sampleInfoForVariable.append(vDnSampleInfo)
                                    else:
                                        vDn = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample, circuit=circuit, concept=e[0])
                                    _vDns.append(vDn)
                                else:
                                    eT = (conceptName, 1, 0)
                                    if sample:
                                        vDn, vDnSampleInfo = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample, circuit=circuit, concept=e[0])
                                        _sampleInfoForVariable.append(vDnSampleInfo)
                                    else:
                                        vDn = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample, circuit=circuit, concept=e[0])
                                    _vDns.append(vDn)

                            vDns.append(_vDns)
                            if sample:
                                sampleInfoForVariable.append(_sampleInfoForVariable)

                        if vDns and loss and not sample:
                            columns = self.stackLossColumns(vDns, self.lossVariableWidth(e))
                            if columns is not None:
                                lcVariables[newVariableName] = columns
                            else:
                                for v in vDns:
                                    if v[0] != None and torch.is_tensor(v[0]):
                                        v[0] = torch.unsqueeze(v[0], 0)
                                lcVariables[newVariableName] = vDns
                        else:
                            lcVariables[newVariableName] = vDns

                        if sample:
                            sampleInfo[newVariableName] = sampleInfoForVariable
                    else:
                        lcVariables[newVariableName] = lcVariables[variableName]
                    usedVariablesNames.add(newVariableName)

                elif isinstance(e, (Concept, tuple)):
                    # Get dataNode candidates 
                    result = getCandidates(dn, e, variable, lcVariablesDns, lc, self.myLogger, integrate=integrate)
                    
                    # Handle None result
                    if result is None or result[0] is None:
                        continue
                    
                    # Unpack result - now returns 3 values with expansion info
                    dnsList, referedVariables, expansionInfo = result
                    
                    lcVariablesDns[variableName] = dnsList

                    lcVariableVs[variableName] = variable
                    binding = self.groundingBinding(variable, dnsList, lcVariablesDns)
                    if binding is not None:
                        lcVariableBindings[variableName] = binding

                    # Apply expansion to lcVariables if expansion occurred
                    if expansionInfo is not None:
                        mapping = expansionInfo['mapping']
                        expanded_vars = expansionInfo['expanded_vars']
                        
                        # Expand all lcVariables entries that match the pre-expansion
                        # group count — not only those from lcVariablesDns.
                        # Nested constraint results (e.g. existsL output stored as _lc1)
                        # live only in lcVariables and must also be realigned when
                        # a sibling variable triggers expansion (issue #377).
                        pre_expansion_len = max(idx for idx, _ in mapping) + 1 if mapping else 0
                        vars_to_expand = set(expanded_vars)
                        for var_name in list(lcVariables.keys()):
                            if var_name in vars_to_expand:
                                continue
                            old_structure = lcVariables[var_name]
                            if old_structure and len(old_structure) == pre_expansion_len:
                                vars_to_expand.add(var_name)
                        
                        self.myLogger.info(f"Applying expansion to lcVariables for: {vars_to_expand}")

                        # Expansion re-grounds earlier variables onto this one's
                        # candidate rows, so they inherit its grounding too.
                        if binding is not None:
                            for var_name in vars_to_expand:
                                lcVariableBindings[var_name] = binding

                        for var_name in vars_to_expand:
                            if var_name not in lcVariables:
                                continue
                            
                            old_structure = lcVariables[var_name]
                            if not old_structure:
                                continue
                            
                            new_structure = []
                            for orig_group_idx, item_idx in mapping:
                                if orig_group_idx < len(old_structure):
                                    old_group = old_structure[orig_group_idx]
                                    if old_group:
                                        # When replicating a variable during expansion,
                                        # handle cases where source group size differs from expansion.
                                        # If source group has only 1 item, replicate it.
                                        # Otherwise try to use item_idx if in bounds.
                                        if isinstance(old_group, list):
                                            if len(old_group) == 1:
                                                # Single item - replicate for all expanded positions
                                                new_structure.append([old_group[0]])
                                            elif item_idx < len(old_group):
                                                # Item exists at this index - use it
                                                new_structure.append([old_group[item_idx]])
                                            else:
                                                # Index out of bounds - replicate first item
                                                new_structure.append([old_group[0]])
                                        else:
                                            # Not a list - wrap it
                                            new_structure.append([old_group])
                                    else:
                                        new_structure.append([None])
                                else:
                                    new_structure.append([None])
                            
                            lcVariables[var_name] = new_structure
                            self.myLogger.info(f"  {var_name}: {len(old_structure)} → {len(new_structure)} entries")
                                
                    if isinstance(lc, CandidateSelection):
                        continue
                    
                    if len(referedVariables) == 1:
                        referedVariable = referedVariables.pop()
                        
                        if referedVariable.startswith('p'):
                            if referedVariable not in newVariables:
                                newVariables[referedVariable] = set()
                            newVariables[referedVariable].add(variableName)

                    # Get ILP variables/values from collected DataNodes
                    conceptName = e[0].name
                    vDns = []
                    if sample:
                        sampleInfoForVariable = []
                    xPkey = '<' + conceptName + ">" + key

                    for dns in dnsList:
                        _vDns = []
                        if sample:
                            _sampleInfoForVariable = []
                            
                        for _dn in dns:
                            if not _dn:
                                vDn = None
                                _vDns.append(vDn)
                                continue

                            if isinstance(e[0], EnumConcept) and e[2] == None:
                                eList = e[0].enum
                                for i, _ in enumerate(eList):
                                    eT = (e[0].name, i, i)
                                    if sample:
                                        vDn, vDnSampleInfo = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample, circuit=circuit, concept=e[0])
                                        _sampleInfoForVariable.append(vDnSampleInfo)
                                    else:
                                        vDn = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample, circuit=circuit, concept=e[0])
                                    
                                    if lc.__str__() == "fixedL":
                                        vDn = self.fixedLSupport(_dn, conceptName, vDn, i, m)
                                        
                                    _vDns.append(vDn)
                            elif isinstance(e[0], EnumConcept) and e[2] != None:
                                eT = (e[0].name, e[2], e[2])
                                
                                if sample:
                                    vDn, vDnSampleInfo = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample, circuit=circuit, concept=e[0])
                                    _sampleInfoForVariable.append(vDnSampleInfo)
                                else:
                                    vDn = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample, circuit=circuit, concept=e[0])
                                
                                if lc.__str__() == "fixedL":
                                    self.fixedLSupport(_dn, conceptName, vDn, e[2], m)
                                    
                                vDn = _vDns.append(vDn)
                            else:
                                eT = (conceptName, 1, 0)
                                if sample:
                                    vDn, vDnSampleInfo = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample, circuit=circuit, concept=e[0])
                                    _sampleInfoForVariable.append(vDnSampleInfo)
                                else:
                                    vDn = self.getMLResult(_dn, xPkey, eT, p, loss=loss, sample=sample, circuit=circuit, concept=e[0])
                                
                                if lc.__str__() == "fixedL":
                                    self.fixedLSupport(_dn, conceptName, vDn, 1, m)
                                        
                                vDn = _vDns.append(vDn)
                        
                        vDns.append(_vDns)
                        
                        if sample:
                            sampleInfoForVariable.append(_sampleInfoForVariable)
                        
                    # Store values/variables
                    if vDns and loss and not sample:
                        columns = self.stackLossColumns(vDns, self.lossVariableWidth(e))
                        if columns is not None:
                            lcVariables[variableName] = columns
                        else:
                            for v in vDns:
                                if v[0] != None and torch.is_tensor(v[0]):
                                    v[0] = torch.unsqueeze(v[0], 0)

                            lcVariables[variableName] = vDns
                    else:
                        lcVariables[variableName] = vDns
                    
                    if sample:
                        sampleInfo[variableName] = sampleInfoForVariable
                        
                    usedVariablesNames.add(variableName)
                
                if isinstance(e, LcElement):

                    if isinstance(e, CandidateSelection):
                        lcVariablesDnsNew = self.constructLogicalConstrains(
                            e, booleanProcessor, m, dn, p, key=key, 
                            lcVariablesDns=lcVariablesDns, lcVariables=lcVariables, 
                            headLC=False, loss=loss, sample=sample, vNo=vNo, verify=verify, label=label,
                            circuit=circuit)
                         
                        lcVariablesDns = lcVariablesDnsNew
                        vDns = None
                        if lcVariablesDns:
                            length_of_list = len(next(iter(lcVariablesDns.values())))

                            if sample:
                                vDns = [[torch.ones(p, device=self.current_device, requires_grad=False, dtype=torch.bool)] for _ in range(length_of_list)]
                            elif loss:
                                vDns = [[torch.zeros(length_of_list, device=self.current_device, requires_grad=True, dtype=self._get_dtype())]]
                                vDns = self.addLossTovDns(loss, vDns)
                            else:
                                vDns = [[1] for _ in range(length_of_list)]
                                   
                    if isinstance(e, LogicalConstrain):
                        self.myLogger.info('Processing Nested %r - %s'%(e, e.strEs()))

                        if sample:
                            vDns, sampleInfoLC, lcVariablesLC, lcVariableUpdated = self.constructLogicalConstrains(
                                e, booleanProcessor, m, dn, p, key=key, 
                                lcVariablesDns=lcVariablesDns, lcVariables=lcVariables, 
                                headLC=False, loss=loss, sample=sample, vNo=vNo, verify=verify,
                                circuit=circuit)
                            sampleInfo = {**sampleInfo, **sampleInfoLC}
                            lcVariablesSet = {**lcVariablesSet, **lcVariablesLC}
                            lcVariables = lcVariableUpdated 
                        else:
                            vDns, lcVariableUpdated = self.constructLogicalConstrains(
                                e, booleanProcessor, m, dn, p, key=key, 
                                lcVariablesDns=lcVariablesDns, lcVariables=lcVariables,
                                headLC=False, loss=loss, sample=sample, vNo=vNo, verify=verify,
                                circuit=circuit)
                            
                            # Ensure vDns has the correct structure
                            if verify and not loss and not sample:
                                # Flatten the nested structure for counting
                                flattened_vDns = []
                                for row in vDns:
                                    if isinstance(row, list):
                                        for item in row:
                                            if isinstance(item, list):
                                                for subitem in item:
                                                    flattened_vDns.append([subitem])
                                            else:
                                                flattened_vDns.append([item])
                                    else:
                                        flattened_vDns.append([row])
                                if flattened_vDns:
                                    vDns = flattened_vDns
                            
                            vDns = self.addLossTovDns(loss, vDns)
                            lcVariables = lcVariableUpdated

                    if vDns == None:
                        self.myLogger.warning('Not found data for %s(%s) nested Logical Constraint required to build %s(%s) - skipping it'%(e.lcName,e,lc.lcName,lc))
                        return None
                        
                    countValid = sum(1 for sublist in vDns if sublist and any(elem is not None for elem in sublist))
                    self.myLogger.info('Size of candidate list returned by %s(%s) nested Logical Constraint is %i of which %i is not None'%(e.lcName,e,len(vDns),countValid))
                    lcVariables[variableName] = vDns   
                    usedVariablesNames.add(variableName)    
            elif isinstance(e, (int, str)):
                pass
            else:
                self.myLogger.error('Logical Constraint %s has incorrect element %s'%(lc,e))
                return None

        for referedVariable in newVariables:
            refVarSet = newVariables[referedVariable]
            refVarSet.add(referedVariable)  
            lcVariables = self.eliminate_duplicate_columns(lcVariablesDns, refVarSet, lcVariables)

        useLcVariables = {k: v for k, v in lcVariables.items() if k in usedVariablesNames}

        if isinstance(lc, CandidateSelection):
            return lc(lcVariablesDns, keys=lc.CandidateSelectionVariable)
        elif sample:
            lcVariablesSet[lc] = useLcVariables
            return lc(m, booleanProcessor, useLcVariables, headConstrain=headLC, integrate=integrate, **({"label": label} if isinstance(lc, sumL) else {})), sampleInfo, lcVariablesSet, lcVariables
        elif verify and headLC:
            return lc(m, booleanProcessor, useLcVariables, headConstrain=headLC, integrate=integrate, **({"label": label} if isinstance(lc, sumL) else {})), lcVariables
        else:
            if loss:
                # Align operands enumerated over different variable tuples
                # before combining them (no-op when they are co-grounded).
                self.fillPathBindings(useLcVariables, lcVariableVs,
                                      lcVariablesDns, lcVariableBindings)
                useLcVariables = self.reduceToCommonGrounding(
                    useLcVariables, lcVariableBindings, booleanProcessor)

                slpitT = False
                for v in useLcVariables:
                    if useLcVariables[v] and len(useLcVariables[v]) > 1:
                        slpitT = True
                        break
                    
                if slpitT:
                    for v in useLcVariables:
                        if useLcVariables[v] and len(useLcVariables[v]) > 1:
                            continue
                         
                        useLcVariables[v] = self.splitLossColumns(useLcVariables[v])

            return lc(m, booleanProcessor, useLcVariables, headConstrain=headLC, integrate=integrate, **({"label": label} if isinstance(lc, sumL) else {})), lcVariables
