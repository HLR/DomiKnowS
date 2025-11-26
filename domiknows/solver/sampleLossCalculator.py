from time import perf_counter, perf_counter_ns
import torch
from collections import OrderedDict
from itertools import product

from domiknows.graph import fixedL


class SampleLossCalculator:
    """Helper class for calculating sample-based loss for logical constraints."""
    
    def __init__(self, solver):
        """
        Initialize sample loss calculator with reference to main solver.
        
        Args:
            solver: Reference to gurobiILPOntSolver instance
        """
        self.solver = solver
        
    def calculateSampleLoss(self, dn, sampleSize, sampleGlobalLoss, conceptsRelations):
        """
        Calculate sample-based loss for logical constraints.
        
        Args:
            dn: Data node
            sampleSize: Size of sample to generate (-1 for semantic sample)
            sampleGlobalLoss: Whether to use global loss calculation
            conceptsRelations: Tuple of concept relations
            
        Returns:
            Dictionary of sample loss values per logical constraint
        """
        p = sampleSize
        if sampleSize == -1:  # Semantic Sample
            sampleSize = self._generateSemanticSample(dn, conceptsRelations)
        if sampleSize < -1:
            raise Exception("Sample size is not incorrect - %i" % (sampleSize))
        
        myBooleanMethods = self.solver.myLcLossSampleBooleanMethods
        myBooleanMethods.sampleSize = sampleSize
        myBooleanMethods.current_device = dn.current_device
        
        self.solver.myLogger.info('Calculating sample loss with sample size: %i' % (p))
        self.solver.myLoggerTime.info('Calculating sample loss with sample size: %i' % (p))
        
        key = "/local/softmax"
        lcCounter = 0
        lcLosses = {}
        
        # First pass: construct logical constraints and collect sample data
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
                
                self.solver.constraintConstructor.current_device = dn.current_device
                self.solver.constraintConstructor.myGraph = self.solver.myGraph
                lossList, sampleInfo, inputLc, _ = self.solver.constraintConstructor.constructLogicalConstrains(
                    lc, myBooleanMethods, None, dn, p, key=key, headLC=True, loss=True, sample=True)
                
                current_lcLosses['lossList'] = lossList
                current_lcLosses['sampleInfo'] = sampleInfo
                current_lcLosses['input'] = inputLc
                current_lcLosses['lossRate'] = []
                
                for li in lossList:
                    liList = []
                    for l in li:
                        if l is not None:
                            liList.append(torch.sum(l) / l.shape[0])
                        else:
                            liList.append(None)
                    current_lcLosses['lossRate'].append(liList)
                
                endLC = perf_counter_ns()
                elapsedInNsLC = endLC - startLC
                elapsedInMsLC = elapsedInNsLC / 1000000
                current_lcLosses['elapsedInMsLC'] = elapsedInMsLC
        
        # Second pass: calculate successes
        globalSuccesses = torch.ones(sampleSize, device=self.solver.current_device)
        
        for currentLcName in lcLosses:
            startLC = perf_counter_ns()
            
            current_lcLosses = lcLosses[currentLcName]
            lossList = current_lcLosses['lossList']
            sampleInfo = current_lcLosses['sampleInfo']
            
            successesList = []
            lcSuccesses = torch.ones(sampleSize, device=self.solver.current_device)
            lcVariables = OrderedDict()
            countSuccesses = torch.zeros(sampleSize, device=self.solver.current_device)
            oneT = torch.ones(sampleSize, device=self.solver.current_device)
            
            # Prepare data
            if len(lossList) == 1:
                for currentFailures in lossList[0]:
                    if currentFailures is None:
                        successesList.append(None)
                        continue
                    
                    currentSuccesses = torch.sub(oneT, currentFailures.float())
                    successesList.append(currentSuccesses)
                    
                    lcSuccesses.mul_(currentSuccesses)
                    globalSuccesses.mul_(currentSuccesses)
                    countSuccesses.add_(currentSuccesses)
                
                # Collect lc variable
                for k in sampleInfo.keys():
                    for c in sampleInfo[k]:
                        if not c:
                            continue
                        c = c[0]
                        if len(c) > 2:
                            if c[2] not in lcVariables:
                                lcVariables[c[2]] = c
            else:
                for i, l in enumerate(lossList):
                    for currentFailures in l:
                        if currentFailures is None:
                            successesList.append(None)
                            continue
                        
                        currentSuccesses = torch.sub(oneT, currentFailures.float())
                        successesList.append(currentSuccesses)
                        
                        lcSuccesses.mul_(currentSuccesses)
                        globalSuccesses.mul_(currentSuccesses)
                        countSuccesses.add_(currentSuccesses)
                    
                    # Collect lc variable
                    for k in sampleInfo.keys():
                        for c in sampleInfo[k][i]:
                            if len(c) > 2:
                                if c[2] not in lcVariables:
                                    lcVariables[c[2]] = c
            
            current_lcLosses['successesList'] = successesList
            current_lcLosses['lcSuccesses'] = lcSuccesses
            current_lcLosses['lcVariables'] = lcVariables
            current_lcLosses['countSuccesses'] = countSuccesses
            
            endLC = perf_counter_ns()
            elapsedInNsLC = endLC - startLC
            elapsedInMsLC = elapsedInNsLC / 1000000
            current_lcLosses['elapsedInMsLC'] += elapsedInMsLC
        
        lcLosses["globalSuccesses"] = globalSuccesses
        lcLosses["globalSuccessCounter"] = torch.nansum(globalSuccesses).item()
        self.solver.myLoggerTime.info('Global success counter is %i' % (lcLosses["globalSuccessCounter"]))
        
        # Third pass: calculate sample loss for lc variables
        for currentLcName in lcLosses:
            if currentLcName in ["globalSuccessCounter", "globalSuccesses"]:
                continue
            
            startLC = perf_counter_ns()
            
            current_lcLosses = lcLosses[currentLcName]
            lossList = current_lcLosses['lossList']
            successesList = current_lcLosses['successesList']
            lcSuccesses = current_lcLosses['lcSuccesses']
            lcVariables = current_lcLosses['lcVariables']
            sampleInfo = current_lcLosses['sampleInfo']
            
            eliminateDuplicateSamples = False
            
            current_lcLosses['lossTensor'] = []
            current_lcLosses['conversionTensor'] = []
            current_lcLosses['lcSuccesses'] = []
            current_lcLosses['lcVariables'] = []
            current_lcLosses['loss'] = []
            current_lcLosses['conversion'] = []
            current_lcLosses['conversionSigmoid'] = []
            current_lcLosses['conversionClamp'] = []
            
            # Get the lc object to check sampleEntries
            lc = None
            for graph in self.solver.myGraph:
                if currentLcName in graph.logicalConstrains:
                    lc = graph.logicalConstrains[currentLcName]
                    break
            
            # Per each lc entry separately
            if lc and lc.sampleEntries:
                current_lcLosses['lcSuccesses'] = successesList
                
                for i, l in enumerate(lossList):
                    currentLcVariables = OrderedDict()
                    for k in sampleInfo.keys():
                        for c in sampleInfo[k][i]:
                            if len(c) > 2:
                                if c[2] not in currentLcVariables:
                                    currentLcVariables[c[2]] = c
                    
                    usedLcSuccesses = successesList[i]
                    if sampleGlobalLoss:
                        usedLcSuccesses = globalSuccesses
                    
                    currentLossTensor, _ = self._calculateSampleLossForVariable(
                        currentLcVariables, usedLcSuccesses, sampleSize, eliminateDuplicateSamples)
                    
                    current_lcLosses['lossTensor'].append(currentLossTensor)
                    current_lcLosses['conversionTensor'].append(1 - currentLossTensor)
                    current_lcLosses['lcVariables'].append(currentLcVariables)
                    
                    currentLoss = torch.nansum(currentLossTensor).item()
                    current_lcLosses['loss'].append(currentLoss)
                    current_lcLosses['conversion'].append(1 - currentLoss)
                    current_lcLosses['conversionSigmoid'].append(torch.sigmoid(-currentLoss))
                    current_lcLosses['conversionClamp'].append(torch.clamp(1 - currentLoss, min=0.0, max=1.0))
            
            else:  # Regular calculation for all lc entries at once
                usedLcSuccesses = lcSuccesses
                if sampleGlobalLoss:
                    usedLcSuccesses = globalSuccesses
                
                lossTensor, lcSampleSize = self._calculateSampleLossForVariable(
                    currentLcName, lcVariables, usedLcSuccesses, sampleSize, eliminateDuplicateSamples)
                
                loss_val = torch.nansum(lossTensor).item()
                
                current_lcLosses['loss'].append(loss_val)
                current_lcLosses['conversion'].append(1.0 - loss_val)
                current_lcLosses['conversionSigmoid'].append(torch.sigmoid(torch.tensor(-loss_val)))
                current_lcLosses['conversionClamp'].append(
                    torch.clamp(torch.tensor(1.0 - loss_val), min=0.0, max=1.0))
                
                current_lcLosses['lossTensor'].append(lossTensor)
                current_lcLosses['conversionTensor'].append(1.0 - lossTensor)
                current_lcLosses['lcSuccesses'].append(lcSuccesses)
                current_lcLosses['lcVariables'].append(lcVariables)
            
            endLC = perf_counter_ns()
            elapsedInNsLC = endLC - startLC
            elapsedInMsLC = elapsedInNsLC / 1000000
            current_lcLosses['elapsedInMsLC'] += elapsedInMsLC
            
            if lc and lc.sampleEntries:
                self.solver.myLoggerTime.info('Processing time for %s with %i entries and %i variables is: %ims' %
                                    (currentLcName, len(lossList), len(lcVariables), current_lcLosses['elapsedInMsLC']))
            elif eliminateDuplicateSamples:
                self.solver.myLoggerTime.info('Processing time for %s with %i entries, %i variables and %i unique samples is: %ims' %
                                    (currentLcName, len(lossList), len(lcVariables), lcSampleSize, current_lcLosses['elapsedInMsLC']))
            else:
                self.solver.myLoggerTime.info('Processing time for %s with %i entries and %i variables is: %ims' %
                                    (currentLcName, len(lossList), len(lcVariables), current_lcLosses['elapsedInMsLC']))
        
        self.solver.myLogger.info('')
        self.solver.myLogger.info('Processed %i logical constraints' % (lcCounter))
        self.solver.myLoggerTime.info('Processed %i logical constraints' % (lcCounter))
        
        return lcLosses
    
    def _eliminateDuplicateSamples(self, lcVariables, sampleSize):
        """Eliminate duplicate samples from variable samples."""
        variablesSamples = [lcVariables[v][1] for v in lcVariables]
        variablesSamplesT = torch.stack(variablesSamples)
        
        uniqueSampleIndex = OrderedDict()
        
        for i in range(sampleSize):
            currentS = variablesSamplesT[:,i]
            currentSHash = hash(currentS.cpu().detach().numpy().tobytes())
            
            if currentSHash in uniqueSampleIndex:
                continue
                
                if torch.equal(currentS, variablesSamplesT[:,uniqueSampleIndex[currentSHash]]):
                    continue
                else:
                    raise Exception("HashWrong")
            else:
                uniqueSampleIndex[currentSHash] = i
        
        va = list(uniqueSampleIndex.values())
        newSampleSize = len(va)
        
        indices = torch.tensor(va, device=self.solver.current_device)
        Vs = torch.index_select(variablesSamplesT, dim=1, index=indices)
        
        return newSampleSize, indices, Vs

    def _calculateSampleLossForVariable(self, currentLcName, lcVariables, lcSuccesses, sampleSize, 
                                       eliminateDuplicateSamples, replace_mul=False):
        """Calculate sample loss for a specific variable."""
        lcSampleSize = sampleSize
        if eliminateDuplicateSamples:
            lcSampleSize, indices, Vs = self._eliminateDuplicateSamples(lcVariables, sampleSize)

        if eliminateDuplicateSamples: 
            lossTensor = torch.index_select(lcSuccesses, dim=0, index=indices)
        else:
            if replace_mul:
                lossTensor = torch.zeros(lcSuccesses.shape).to(self.solver.current_device)
            else:
                lossTensor = torch.ones(lcSuccesses.shape).to(self.solver.current_device)
            
        for i, v in enumerate(lcVariables):
            currentV = lcVariables[v]
            
            if eliminateDuplicateSamples:
                P = currentV[0][:lcSampleSize]
            else:
                P = currentV[0]
            oneMinusP = torch.sub(torch.ones(lcSampleSize, device=self.solver.current_device), P)
            
            if eliminateDuplicateSamples:
                S = Vs[i, :]
            else:
                S = currentV[1]
                
            if isinstance(S, list):
                continue
            
            notS = torch.sub(torch.ones(lcSampleSize, device=self.solver.current_device), S.float())
            
            pS = torch.mul(P, S)
            oneMinusPS = torch.mul(oneMinusP, notS)
            
            cLoss = torch.add(pS, oneMinusPS)
                                            
            if replace_mul:
                lossTensor = lossTensor + torch.log(cLoss)
            else:
                lossTensor.mul_(cLoss)
        
        return lossTensor, lcSampleSize
            
    def _generateSemanticSample(self, rootDn, conceptsRelations):
        """Generate semantic sample for logical constraints."""
        sampleSize = -1
        
        masterConcepts = OrderedDict()
        productConcepts = []
        productSize = 1
        productArgs = []
        
        for currentConceptRelation in conceptsRelations:
            currentConceptName = self.solver.getConceptName(currentConceptRelation)
            
            if currentConceptName not in masterConcepts:
                masterConcepts[currentConceptName] = OrderedDict()
                
                masterConcepts[currentConceptName]['e'] = []
                masterConcepts[currentConceptName]['e'].append(currentConceptRelation)
                
                rootConcept = rootDn.findRootConceptOrRelation(currentConceptName)
                dns = rootDn.findDatanodes(select=rootConcept)
                masterConcepts[currentConceptName]["dns"] = dns
                
                conceptRange = 2
                if self.solver.conceptIsMultiClass(currentConceptRelation):
                    conceptRange = currentConceptRelation[3]
                masterConcepts[currentConceptName]["range"] = [x for x in range(conceptRange)]
                
                masterConcepts[currentConceptName]['xkey'] = '<' + currentConceptName + '>/sample'
                 
                if self.isConceptFixed(currentConceptName):
                    continue
                
                for _ in dns:
                    productArgs.append(masterConcepts[currentConceptName]["range"])
                    productSize *= conceptRange
                    
                productConcepts.append(currentConceptName)
            else:
                masterConcepts[currentConceptName]['e'].append(currentConceptRelation)

        # Init sample 
        for mConcept in masterConcepts:
            mConceptInfo = masterConcepts[mConcept]
            
            if len(mConceptInfo["e"]) == 1:
                mConceptInfo["binary"] = True
            else:
                mConceptInfo["binary"] = False
            
            for dn in mConceptInfo['dns']:
                if mConceptInfo['xkey'] not in dn.getAttributes():
                    dn.getAttributes()[mConceptInfo['xkey']] = OrderedDict()
                                                                    
                dn.getAttributes()[mConceptInfo['xkey']][sampleSize] = OrderedDict()
                
                for i, e in enumerate(mConceptInfo["e"]):
                    isFiexd = self.solver.isVariableFixed(dn, mConcept, e)
                    
                    if mConceptInfo["binary"]:
                        i = 1
                        
                    if isFiexd != None:
                        if isFiexd == 1:
                            dn.getAttributes()[mConceptInfo['xkey']][sampleSize][i] = torch.ones(
                                (productSize,), device=self.solver.current_device)
                            continue
                        
                    dn.getAttributes()[mConceptInfo['xkey']][sampleSize][i] = torch.zeros(
                        (productSize,), device=self.solver.current_device)
             
        for j, p in enumerate(product(*productArgs, repeat=1)):
            index = 0
            if mConceptInfo["binary"]:
                index = 1
            for pConcept in productConcepts:
                pConceptInfo = masterConcepts[pConcept]

                for dn in pConceptInfo['dns']:
                    try:
                        dn.getAttributes()[pConceptInfo['xkey']][sampleSize][p[index]][j] = 1
                    except:
                        pass
                    index += 1
        
        return productSize
    
    def isConceptFixed(self, conceptName):
        for graph in self.myGraph: # Loop through graphs
            for _, lc in graph.logicalConstrains.items(): # loop trough lcs in the graph
                if not lc.headLC or not lc.active: # Process only active and head lcs
                    continue
                    
                if type(lc) is not fixedL: # Skip not fixedL lc
                    continue
                
                if not lc.e:
                    continue
                
                if lc.e[0][1] == conceptName:
                    return True
            
        return False