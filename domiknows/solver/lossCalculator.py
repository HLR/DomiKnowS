from time import perf_counter_ns
import torch

from domiknows.graph import fixedL
from domiknows.graph.logicalConstrain import sumL

class LossCalculator:
    """Helper class for calculating regular (non-sample) loss for logical constraints."""
    
    def __init__(self, solver):
        """
        Initialize loss calculator with reference to main solver.
        
        Args:
            solver: Reference to gurobiILPOntSolver instance
        """
        self.solver = solver
        
    def calculateLoss(self, dn, tnorm='L', counting_tnorm=None):
        """
        Calculate regular (non-sample) loss for logical constraints.
        
        Args:
            dn: Data node
            tnorm: T-norm to use for loss calculation
            counting_tnorm: Optional counting t-norm
            
        Returns:
            Dictionary of loss values per logical constraint
        """
        myBooleanMethods = self.solver.myLcLossBooleanMethods
        myBooleanMethods.setTNorm(tnorm)
        if counting_tnorm:
            myBooleanMethods.setCountingTNorm(counting_tnorm)
        
        myBooleanMethods.current_device = dn.current_device
        
        self.solver.myLogger.info('Calculating loss')
        self.solver.myLoggerTime.info('Calculating loss')
        
        key = "/local/softmax"
        lcCounter = 0
        lcLosses = {}
        
        # First pass: construct logical constraints and collect losses
        for graph in self.solver.myGraph:
            for _, lc in graph.logicalConstrains.items():
                startLC = perf_counter_ns()
                
                if not lc.headLC or not lc.active:
                    continue
                
                if type(lc) is fixedL:
                    continue
                
                lcCounter += 1
                self.solver.myLogger.info('\n')
                self.solver.myLogger.info('Processing %r - %s' % (lc, lc.strEs()))
                
                lcName = lc.lcName
                lcLosses[lcName] = {}
                current_lcLosses = lcLosses[lcName]
                
                current_lcLosses['lc'] = lc
                
                lossList = self.solver.constructLogicalConstrains(
                    lc, myBooleanMethods, None, dn, 0, 
                    key=key, headLC=True, loss=True, sample=False)
                current_lcLosses['lossList'] = lossList
                
                endLC = perf_counter_ns()
                elapsedInNsLC = endLC - startLC
                elapsedInMsLC = elapsedInNsLC / 1000000
                current_lcLosses['elapsedInMsLC'] = elapsedInMsLC
        
        # Second pass: calculate final loss values
        for currentLcName in lcLosses:
            startLC = perf_counter_ns()
            
            current_lcLosses = lcLosses[currentLcName]
            lossList = current_lcLosses['lossList']
            lossTensor = None
            
            seperateTensorsUsed = False
            if lossList and lossList[0]:
                if torch.is_tensor(lossList[0][0]) and (len(lossList[0][0].shape) == 0 or 
                                                        len(lossList[0][0].shape) == 1 and lossList[0][0].shape[0] == 1):
                    seperateTensorsUsed = True
                
                if seperateTensorsUsed:
                    lossTensor = torch.zeros(len(lossList), device=self.solver.current_device)
                    for i, l in enumerate(lossList):
                        lossTensor[i] = float("nan")
                        for entry in l:
                            if entry is not None:
                                if lossTensor[i] != lossTensor[i]:
                                    lossTensor[i] = entry
                                else:
                                    lossTensor[i] += entry.item()
                else:
                    for entry in lossList[0]:
                        if entry is not None:
                            if lossTensor is None:
                                lossTensor = entry[0]
                            else:
                                lossTensor += entry[0]
                                
            lc = current_lcLosses.get('lc', None)
            if isinstance(lc, sumL) and lossTensor is not None and torch.is_tensor(lossTensor):
                # reduce over any entries; for headLC sumL this is just a single element
                expected_count = torch.nansum(lossTensor)
                current_lcLosses['expectedCount'] = expected_count
            
            current_lcLosses['lossTensor'] = lossTensor
            current_lcLosses['conversionTensor'] = 1 - lossTensor
            
            if lossTensor is not None and torch.is_tensor(lossTensor):
                current_lcLosses['loss'] = torch.nansum(lossTensor)
                current_lcLosses['conversionSigmoid'] = torch.sigmoid(-current_lcLosses['loss'])
                current_lcLosses['conversionClamp'] = torch.clamp(1 - current_lcLosses['loss'], min=0.0, max=1.0)
                current_lcLosses['conversion'] = 1 - current_lcLosses['loss']
            else:
                current_lcLosses['loss'] = None
                current_lcLosses['conversion'] = None
                current_lcLosses['conversionSigmoid'] = None
                current_lcLosses['conversionClamp'] = None
            
            endLC = perf_counter_ns()
            elapsedInNsLC = endLC - startLC
            elapsedInMsLC = elapsedInNsLC / 1000000
            current_lcLosses['elapsedInMsLC'] += elapsedInMsLC
            
            if lossList is not None:
                self.solver.myLoggerTime.info('Processing time for %s with %i entries is: %ims' % 
                                    (currentLcName, len(lossList), current_lcLosses['elapsedInMsLC']))
            else:
                self.solver.myLoggerTime.info('Processing time for %s with %i entries is: %ims' % 
                                    (currentLcName, 0, current_lcLosses['elapsedInMsLC']))
            
            [h.flush() for h in self.solver.myLoggerTime.handlers]
        
        self.solver.myLogger.info('')
        self.solver.myLogger.info('Processed %i logical constraints' % (lcCounter))
        self.solver.myLoggerTime.info('Processed %i logical constraints' % (lcCounter))
        
        return lcLosses