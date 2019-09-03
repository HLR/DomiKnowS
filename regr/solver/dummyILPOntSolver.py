if __package__ is None or __package__ == '': 
    from regr.solver.ilpConfig import ilpConfig
    from regr.solver.ilpOntSolver import ilpOntSolver
else:
    from .ilpConfig import ilpConfig 
    from .ilpOntSolver import ilpOntSolver

class dummyILPOntSolver(ilpOntSolver):
    def __init__(self) -> None:
        super().__init__()
        self.myIlpBooleanProcessor = None
        
    def calculateILPSelection(self, phrase, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None):
        self.myLogger.info('Returning unchanged results')
        return graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation