from collections import OrderedDict
from time import perf_counter, perf_counter_ns
from typing import Dict, List, Optional, Tuple
from domiknows.solver import gurobiILPOntSolver

import torch

class LossCalculator:
    def calculateLcLoss(
        self,
        dn,
        tnorm: str = 'L',
        counting_tnorm: Optional[str] = None,
        sample: bool = False,
        sampleSize: int = 0,
        sampleGlobalLoss: bool = False,
        conceptsRelations=None,
    ) -> Dict[str, Dict]:
        start = perf_counter()
        self.current_device = dn.current_device

        myBooleanMethods, resolvedSampleSize = self._prepare_boolean_methods(
            dn=dn,
            sample=sample,
            tnorm=tnorm,
            counting_tnorm=counting_tnorm,
            sampleSize=sampleSize,
            conceptsRelations=conceptsRelations,
        )

        if sample:
            lcLosses = self._calculate_sample_loss(
                dn=dn,
                myBooleanMethods=myBooleanMethods,
                sampleSize=resolvedSampleSize,
                sampleGlobalLoss=sampleGlobalLoss,
            )
        else:
            lcLosses = self._calculate_normal_loss(
                dn=dn,
                myBooleanMethods=myBooleanMethods,
            )

        gurobiILPOntSolver.myLogger.info('')
        gurobiILPOntSolver.myLogger.info('Processed %i logical constraints' % (lcLosses.get('_lcCounter', 0)))
        gurobiILPOntSolver.myLoggerTime.info('Processed %i logical constraints' % (lcLosses.get('_lcCounter', 0)))

        elapsedS = perf_counter() - start
        if elapsedS > 1:
            gurobiILPOntSolver.myLogger.info('End of Loss Calculation - total internl time: %fs' % elapsedS)
            gurobiILPOntSolver.myLoggerTime.info('End of Loss Calculation - total internl time: %fs' % elapsedS)
        else:
            gurobiILPOntSolver.myLogger.info('End of Loss Calculation - total internl time: %ims' % int(elapsedS * 1000))
            gurobiILPOntSolver.myLoggerTime.info('End of Loss Calculation - total internl time: %ims' % int(elapsedS * 1000))

        gurobiILPOntSolver.myLogger.info('')
        [h.flush() for h in gurobiILPOntSolver.myLoggerTime.handlers]
        lcLosses.pop('_lcCounter', None)
        return lcLosses

    def _ms_since(self, start_ns: int) -> float:
        """
        Returns elapsed time in milliseconds given a start time from perf_counter_ns().
        """
        return (perf_counter_ns() - start_ns) / 1_000_000.0

    def _active_head_lcs(self):
        """
        Yield only active, head logical constraints across all graphs,
        skipping fixedL types.
        """
        try:
            from domiknows.program.logicalConstrains import fixedL  # type: ignore
        except Exception:
            class fixedL:  # fallback placeholder if import fails
                pass

        for graph in gurobiILPOntSolver.myGraph:
            for _, lc in graph.logicalConstrains.items():
                if not getattr(lc, "headLC", False) or not getattr(lc, "active", False):
                    continue
                if isinstance(lc, fixedL):
                    continue
                yield lc


    def _prepare_boolean_methods(
        self,
        dn,
        sample: bool,
        tnorm: str,
        counting_tnorm: Optional[str],
        sampleSize: int,
        conceptsRelations,
    ):
        if sample:
            if sampleSize == -1:
                sampleSize = gurobiILPOntSolver.generateSemanticSample(dn, conceptsRelations)
            if sampleSize < -1:
                raise Exception(f"Sample size is not incorrect - {sampleSize}")

            bm = gurobiILPOntSolver.myLcLossSampleBooleanMethods
            bm.sampleSize = sampleSize
            gurobiILPOntSolver.myLogger.info('Calculating sample loss with sample size: %i' % sampleSize)
            gurobiILPOntSolver.myLoggerTime.info('Calculating sample loss with sample size: %i' % sampleSize)
        else:
            bm = gurobiILPOntSolver.myLcLossBooleanMethods
            bm.setTNorm(tnorm)
            if counting_tnorm:
                bm.setCountingTNorm(counting_tnorm)
            gurobiILPOntSolver.myLogger.info('Calculating loss ')
            gurobiILPOntSolver.myLoggerTime.info('Calculating loss ')

        bm.current_device = dn.current_device
        return bm, sampleSize

    def _calculate_normal_loss(self, dn, myBooleanMethods) -> Dict[str, Dict]:
        key = "/local/softmax"
        lcCounter = 0
        lcLosses: Dict[str, Dict] = {}

        # Phase 1: build lossList
        for lc in self._active_head_lcs():
            start_ns = perf_counter_ns()
            lcCounter += 1

            gurobiILPOntSolver.myLogger.info('\n')
            gurobiILPOntSolver.myLogger.info('Processing %r - %s' % (lc, lc.strEs()))

            lcName = lc.lcName
            current = lcLosses.setdefault(lcName, {})
            lossList = self.constructLogicalConstrains(
                lc, myBooleanMethods, None, dn, 0,
                key=key, headLC=True, loss=True, sample=False
            )
            current['lossList'] = lossList
            current['elapsedInMsLC'] = self._ms_since(start_ns)

        # Phase 2: aggregate to tensors
        for lcName, current in lcLosses.items():
            start_ns = perf_counter_ns()
            lossList = current.get('lossList')
            lossTensor: Optional[torch.Tensor] = None

            # Guards for empty lists
            if isinstance(lossList, list) and len(lossList) > 0 and isinstance(lossList[0], list) and len(lossList[0]) > 0:
                first = lossList[0][0]
                separateTensorsUsed = (torch.is_tensor(first) and (first.ndim == 0 or (first.ndim == 1 and first.shape[0] == 1)))

                if separateTensorsUsed:
                    lossTensor = torch.zeros(len(lossList), device=self.current_device)
                    for i, l in enumerate(lossList):
                        # init to NaN; then sum entries present
                        lossTensor[i] = float("nan")
                        for entry in l:
                            if entry is None:
                                continue
                            if not torch.is_tensor(entry):
                                continue
                            if torch.isnan(lossTensor[i]):
                                lossTensor[i] = entry if entry.ndim == 0 else entry.squeeze()
                            else:
                                lossTensor[i] = lossTensor[i] + (entry if entry.ndim == 0 else entry.squeeze())
                else:
                    for entry in lossList[0]:   
                        if entry is None or not torch.is_tensor(entry):
                            continue
                        lossTensor = entry if lossTensor is None else lossTensor + entry

            current['lossTensor'] = lossTensor
            current['conversionTensor'] = None if lossTensor is None else (1 - lossTensor)
            current['loss'] = None if (lossTensor is None or not torch.is_tensor(lossTensor)) else torch.nansum(lossTensor).item()
            current['conversion'] = None if current['loss'] is None else (1 - current['loss'])
            current['conversionSigmoid'] = None if current['loss'] is None else (torch.sigmoid(-current['loss']))
            current['conversionClamp'] = None if current['loss'] is None else (torch.clamp(1 - current['loss'], min=0.0, max=1.0))

            current['elapsedInMsLC'] += self._ms_since(start_ns)
            n_entries = len(lossList) if isinstance(lossList, list) else 0
            gurobiILPOntSolver.myLoggerTime.info('Processing time for %s with %i entries is: %ims' %
                                (lcName, n_entries, current['elapsedInMsLC']))
            [h.flush() for h in gurobiILPOntSolver.myLoggerTime.handlers]

        lcLosses['_lcCounter'] = lcCounter
        return lcLosses

    def _calculate_sample_loss(self, dn, myBooleanMethods, sampleSize: int, sampleGlobalLoss: bool) -> Dict[str, Dict]:
        key = "/local/softmax"
        lcCounter = 0
        lcLosses: Dict[str, Dict] = {}

        # Phase 1: gather lossList + sampleInfo
        for lc in self._active_head_lcs():
            start_ns = perf_counter_ns()
            lcCounter += 1

            gurobiILPOntSolver.myLogger.info('\n')
            gurobiILPOntSolver.myLogger.info('Processing %r - %s' % (lc, lc.strEs()))

            lcName = lc.lcName
            current = lcLosses.setdefault(lcName, {})

            lossList, sampleInfo, inputLc = gurobiILPOntSolver.constructLogicalConstrains(
                lc, myBooleanMethods, None, dn, sampleSize,
                key=key, headLC=True, loss=True, sample=True
            )
            current['lossList'] = lossList
            current['sampleInfo'] = sampleInfo
            current['input'] = inputLc

            # Diagnostics: average failure per entry (guarded)
            lossRate: List[List[Optional[torch.Tensor]]] = []
            for li in lossList:
                row = []
                for l in li:
                    if l is None or not torch.is_tensor(l) or l.numel() == 0:
                        row.append(None)
                    else:
                        # shape guard: expect vector; if scalar, keep as is
                        v = l if l.ndim == 1 else l.view(-1)
                        row.append(torch.sum(v) / v.shape[0])
                lossRate.append(row)
            current['lossRate'] = lossRate

            current['elapsedInMsLC'] = self._ms_since(start_ns)

        # Phase 2: successes + variable collection
        globalSuccesses = torch.ones(sampleSize, device=self.current_device)

        for lcName, current in lcLosses.items():
            # Skip meta keys
            # (none at this moment)
            start_ns = perf_counter_ns()

            lossList = current['lossList']
            sampleInfo = current['sampleInfo']

            successesList: List[Optional[torch.Tensor]] = []
            lcSuccesses = torch.ones(sampleSize, device=self.current_device)
            lcVariables: "OrderedDict[str, Tuple]" = OrderedDict()
            countSuccesses = torch.zeros(sampleSize, device=self.current_device)
            oneT = torch.ones(sampleSize, device=self.current_device)

            if len(lossList) == 1:
                for currentFailures in lossList[0]:
                    if (currentFailures is None) or (not torch.is_tensor(currentFailures)):
                        successesList.append(None)
                        continue
                    f = currentFailures.float().view(-1)  # guard to vector
                    currentSuccesses = torch.sub(oneT, f)
                    successesList.append(currentSuccesses)
                    lcSuccesses.mul_(currentSuccesses)
                    globalSuccesses.mul_(currentSuccesses)
                    countSuccesses.add_(currentSuccesses)

                # collect variables
                for k in sampleInfo.keys():
                    for c in sampleInfo[k]:
                        if not c:
                            continue
                        c0 = c[0]
                        if len(c0) > 2:
                            keyVar = c0[2]
                            if keyVar not in lcVariables:
                                lcVariables[keyVar] = c0
            else:
                for i, l in enumerate(lossList):
                    for currentFailures in l:
                        if (currentFailures is None) or (not torch.is_tensor(currentFailures)):
                            successesList.append(None)
                            continue
                        f = currentFailures.float().view(-1)
                        currentSuccesses = torch.sub(oneT, f)
                        successesList.append(currentSuccesses)
                        lcSuccesses.mul_(currentSuccesses)
                        globalSuccesses.mul_(currentSuccesses)
                        countSuccesses.add_(currentSuccesses)
                    # collect variables per entry
                    for k in sampleInfo.keys():
                        for c in sampleInfo[k][i]:
                            if len(c) > 2 and c[2] not in lcVariables:
                                lcVariables[c[2]] = c

            current['successesList'] = successesList
            current['lcSuccesses'] = lcSuccesses
            current['lcVariables'] = lcVariables
            current['countSuccesses'] = countSuccesses
            current['elapsedInMsLC'] += self._ms_since(start_ns)

        lcLosses["globalSuccesses"] = globalSuccesses
        lcLosses["globalSuccessCounter"] = torch.nansum(globalSuccesses).item()
        gurobiILPOntSolver.myLoggerTime.info('Global success counter is %i ' % (lcLosses["globalSuccessCounter"]))

        # Phase 3: compute sample losses 
        for lc in self._active_head_lcs():
            lcName = lc.lcName
            current = lcLosses[lcName]
            start_ns = perf_counter_ns()

            lossList = current['lossList']
            successesList = current['successesList']
            lcSuccesses = current['lcSuccesses']
            lcVariables = current['lcVariables']

            eliminateDuplicateSamples = False

            current['lossTensor'] = []
            current['conversionTensor'] = []
            current['lcSuccesses'] = []
            current['lcVariables'] = []
            current['loss'] = []
            current['conversion'] = []
            current['conversionSigmoid'] = []
            current['conversionClamp'] = []

            if getattr(lc, 'sampleEntries', False):
                current['lcSuccesses'] = successesList
                for i, _ in enumerate(lossList):
                    # rebuild per-entry variables
                    currentEntryVars: "OrderedDict[str, Tuple]" = OrderedDict()
                    sampleInfo = current['sampleInfo']
                    for k in sampleInfo.keys():
                        for c in sampleInfo[k][i]:
                            if len(c) > 2 and c[2] not in currentEntryVars:
                                currentEntryVars[c[2]] = c

                    usedSuccesses = successesList[i] if not sampleGlobalLoss else lcLosses["globalSuccesses"]
                    if usedSuccesses is None:
                        # No data => zero loss vector (or keep NaNs — choose your convention)
                        entryLoss = torch.full((sampleSize,), float('nan'), device=self.current_device)
                        effN = sampleSize
                    else:
                        entryLoss, effN = gurobiILPOntSolver.calulateSampleLossForVariable(
                            lc_name=lcName,
                            lc_variables=currentEntryVars,
                            successes=usedSuccesses,
                            sample_size=sampleSize,
                            eliminate_duplicate_samples=eliminateDuplicateSamples,
                        )

                    current['lossTensor'].append(entryLoss)
                    current['conversionTensor'].append(1 - entryLoss)
                    current['lcVariables'].append(currentEntryVars)
                    current['loss'].append(torch.nansum(entryLoss).item())
                    current['conversion'].append(1 - torch.nansum(entryLoss).item())
                    current['conversionSigmoid'].append(torch.sigmoid(-torch.nansum(entryLoss).item()))
                    current['conversionClamp'].append(torch.clamp(1 - torch.nansum(entryLoss).item(), min=0.0, max=1.0))
            else:
                usedSuccesses = lcLosses["globalSuccesses"] if sampleGlobalLoss else lcSuccesses
                lossTensor, lcSampleSize = gurobiILPOntSolver.calulateSampleLossForVariable(
                    lc_name=lcName,
                    lc_variables=lcVariables,
                    successes=usedSuccesses,
                    sample_size=sampleSize,
                    eliminate_duplicate_samples=eliminateDuplicateSamples,
                )
                current['loss'].append(torch.nansum(lossTensor).item())
                current['conversion'].append(1 - torch.nansum(lossTensor).item())
                current['conversionSigmoid'].append(torch.sigmoid(-torch.nansum(lossTensor).item()))
                current['conversionClamp'].append(torch.clamp(1 - torch.nansum(lossTensor).item(), min=0.0, max=1.0))
                current['lossTensor'].append(lossTensor)
                current['conversionTensor'].append(1 - lossTensor)
                current['lcSuccesses'].append(lcSuccesses)
                current['lcVariables'].append(lcVariables)

            current['elapsedInMsLC'] += self._ms_since(start_ns)

            if getattr(lc, 'sampleEntries', False):
                gurobiILPOntSolver.myLoggerTime.info('Processing time for %s with %i entries and %i variables is: %ims'
                                    % (lcName, len(lossList), len(lcVariables), current['elapsedInMsLC']))
            elif eliminateDuplicateSamples:
                gurobiILPOntSolver.myLoggerTime.info('Processing time for %s with %i entries, %i variables and %i unique samples is: %ims'
                                    % (lcName, len(lossList), len(lcVariables), lcSampleSize, current['elapsedInMsLC']))
            else:
                gurobiILPOntSolver.myLoggerTime.info('Processing time for %s with %i entries and %i variables is: %ims'
                                    % (lcName, len(lossList), len(lcVariables), current['elapsedInMsLC']))

        lcLosses['_lcCounter'] = lcCounter
        return lcLosses
