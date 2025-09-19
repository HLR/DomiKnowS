import abc

from importlib.resources import files
def resource_filename(package, resource):
    return str(files(package) / resource)
import logging
from logging.handlers import RotatingFileHandler

# ontology
from owlready2 import *
    
# path to Meta Graph ontology
graphMetaOntologyPathname = resource_filename('domiknows', 'ontology/ML')

# path
from pathlib import Path
from domiknows.solver.ilpConfig import ilpConfig 

from domiknows import getRegrTimer_logger, setup_logger

class ilpOntSolver(object):
    __metaclass__ = abc.ABCMeta
    
    __negVarTrashhold = 1.0

    def __init__(self, graph, ontologiesTuple, _ilpConfig):
        self.myLogger = None
        self.myLoggerTime = None
        self.update_config(graph, ontologiesTuple, _ilpConfig)

    def update_config(self, graph=None, ontologiesTuple=None, _ilpConfig=None):
        if _ilpConfig is not None:
            self.setup_solver_logger(_ilpConfig=_ilpConfig)
        if graph is not None:
            self.myGraph = graph
        if ontologiesTuple:
            self.loadOntology(ontologiesTuple)

    def setup_solver_logger(self, _ilpConfig=None):
        # Use the encapsulated setup_logger method
        self.myLogger = setup_logger(_ilpConfig, 'ilpOntSolver.log')
        self.myLogger.info('--- Starting new run ---')

        # Get the timer logger
        self.myLoggerTime = getRegrTimer_logger(_ilpConfig)
        
    def loadOntology(self, ontologies):
        start = datetime.datetime.now()
        
        if self.myLogger is None:
            self.setup_solver_logger()
            
        self.myLogger.info('')
        self.myLogger.info('-----------------------------------------------')
        
        currentPath = Path(os.path.normpath("./")).resolve()
            
        # Check if Graph Meta ontology path is correct
        graphMetaOntologyPath = Path(os.path.normpath(graphMetaOntologyPathname))
        graphMetaOntologyPath = graphMetaOntologyPath.resolve()
        if not os.path.isdir(graphMetaOntologyPath):
            self.myLogger.error("Path to load Graph ontology: %s does not exists in current directory %s"%(graphMetaOntologyPath,currentPath))
            exit()
        
        onto_path.append(graphMetaOntologyPath)  # the folder with the Graph Meta ontology

        for currentOntology in ontologies:
            self.myLogger.info('Start Loading ontology %s'%(currentOntology.iri))
            
            if currentOntology.local is not None:
                # Check if specific ontology path is correct
                ontologyPath = Path(os.path.normpath(currentOntology.local))
                ontologyPath = ontologyPath.resolve()
                self.myLogger.info("Path to load ontology: %s is %s resolved to %s"%(currentOntology.iri, currentOntology.local, ontologyPath))

                if not os.path.isdir(ontologyPath):
                    self.myLogger.error("Path to load ontology: %s does not exists in current directory %s"%(currentOntology.iri, currentOntology.local))
                    exit()
    
                onto_path.append(ontologyPath) # the folder with the ontology for the specific  graph
                self.myLogger.info("Ontology: %s is appended"%(currentOntology.iri))

            # Load specific ontology
            try :
                self.myOnto = get_ontology(currentOntology.iri)
                self.myOnto.load(only_local = True, fileobj = None, reload = False, reload_if_newer = False)
            except FileNotFoundError as e:
                self.myLogger.warning("Error when loading - %s from: %s"%(currentOntology.iri, currentOntology.local))
        
            end = datetime.datetime.now()
            elapsed = end - start
            self.myLogger.info('Finished loading ontology - elapsed time: %ims'%(elapsed.microseconds/1000))
        
        return self.myOnto

    @abc.abstractmethod
    def calculateILPSelection(self, phrase, fun=None, epsilon = 0.00001, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None, minimizeObjective = False, hardConstrains = []):
        #self, *conceptsRelations, fun = fun, epsilon = epsilon, minimizeObjective = minimizeObjective, ignorePinLCs = ignorePinLCs
        pass