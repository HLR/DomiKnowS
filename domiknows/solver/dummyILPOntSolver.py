if __package__ is None or __package__ == '': 
    from domiknows.solver.ilpConfig import ilpConfig
    from domiknows.solver.ilpOntSolver import ilpOntSolver
else:
    from .ilpConfig import ilpConfig 
    from .ilpOntSolver import ilpOntSolver

class dummyILPOntSolver(ilpOntSolver):
    def __init__(self, graph, ontologiesTuple, _ilpConfig=ilpConfig) -> None:
        super().__init__(graph, ontologiesTuple, _ilpConfig=ilpConfig)
        self.myIlpBooleanProcessor = None
        
    def calculateILPSelection(self, phrase, fun=None, epsilon = 0.00001, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None):
        self.myLogger.info('Returning unchanged results')
        return graphResultsForPhraseToken, graphResultsForPhraseRelation, graphResultsForPhraseTripleRelation