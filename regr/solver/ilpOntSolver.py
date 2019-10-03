import abc

from pkg_resources import resource_filename
from typing import Dict
from torch import Tensor
import logging

# ontology
from owlready2 import *

from regr.graph import DataNode
    
# path to Meta Graph ontology
graphMetaOntologyPathname = resource_filename('regr', 'ontology/ML')

# path
from pathlib import Path

class ilpOntSolver(object):
    __metaclass__ = abc.ABCMeta
    
    __negVarTrashhold = 1.0
    
    def setup_solver_logger(self, log_filename='ilpOntSolver.log'):
        logger = logging.getLogger(__name__)
    
        # create file handler and set level to info
        ch = logging.FileHandler(log_filename)
        logger.setLevel(logging.DEBUG)
    
        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s:%(funcName)s - %(message)s')
    
        # add formatter to ch
        ch.setFormatter(formatter)
    
        # add ch to logger
        logger.addHandler(ch)
        
        print("Log file is in: ", ch.baseFilename)
        self.myLogger = logger

    def loadOntology(self, ontologyURL, ontologyPathname=None):
        start = datetime.datetime.now()
        
        if self.myLogger is None:
            self.setup_solver_logger()
            
        self.myLogger.info('')
        self.myLogger.info('-----------------------------------------------')
        self.myLogger.info('Start Loading ontology %s'%(ontologyURL))
        
        currentPath = Path(os.path.normpath("./")).resolve()
        
        # Check if Graph Meta ontology path is correct
        graphMetaOntologyPath = Path(os.path.normpath(graphMetaOntologyPathname))
        graphMetaOntologyPath = graphMetaOntologyPath.resolve()
        if not os.path.isdir(graphMetaOntologyPath):
            self.myLogger.error("Path to load Graph ontology: %s does not exists in current directory %s"%(graphMetaOntologyPath,currentPath))
            exit()
            
        if ontologyPathname is not None:
            # Check if specific ontology path is correct
            ontologyPath = Path(os.path.normpath(ontologyPathname))
            ontologyPath = ontologyPath.resolve()
            if not os.path.isdir(ontologyPath):
                self.myLogger.error("Path to load ontology: %s does not exists in current directory %s"%(ontologyURL,currentPath))
                exit()

            onto_path.append(graphMetaOntologyPath)  # the folder with the Graph Meta ontology
            onto_path.append(ontologyPath) # the folder with the ontology for the specific  graph
    
        # Load specific ontology
        try :
            self.myOnto = get_ontology(ontologyURL)
            self.myOnto.load(only_local = True, fileobj = None, reload = False, reload_if_newer = False)
        except FileNotFoundError as e:
            self.myLogger.warning("Error when loading - %s from: %s"%(ontologyURL, ontologyPathname))
    
        end = datetime.datetime.now()
        elapsed = end - start
        self.myLogger.info('Finished loading ontology - elapsed time: %ims'%(elapsed.microseconds/1000))
        
        return self.myOnto

    @abc.abstractmethod
    def calculateILPSelection(self, phrase, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None): pass
