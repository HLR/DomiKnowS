import abc

from pkg_resources import resource_filename
import logging
from logging.handlers import RotatingFileHandler

# ontology
from owlready2 import *
    
# path to Meta Graph ontology
graphMetaOntologyPathname = resource_filename('regr', 'ontology/ML')

# path
from pathlib import Path

if __package__ is None or __package__ == '': 
    from regr.solver.ilpConfig import ilpConfig 
else:
    from .ilpConfig import ilpConfig 

class ilpOntSolver(object):
    __metaclass__ = abc.ABCMeta
    
    __negVarTrashhold = 1.0
    
    def setup_solver_logger(self, _ildConfig = ilpConfig):
        
        if _ildConfig is not None:
            if 'log_name' in _ildConfig:
                logName = _ildConfig['log_name']
            else:
                logName = __name__
                
            if 'log_level' in _ildConfig:
                logLevel = _ildConfig['log_level']
            else:
                logLevel = logging.DEBUG

            if 'log_filename' in _ildConfig:
                logFilename = _ildConfig['log_filename']
            else:
                logFilename='ilpOntSolver.log'
                
            if 'log_filesize' in _ildConfig:
                logFilesize = _ildConfig['log_filesize']
            else:
                logFilesize=5*1024*1024*1024
                
            if 'log_backupCount' in _ildConfig:
                logBackupCount = _ildConfig['log_backupCount']
            else:
                logBackupCount=4
            
        logger = logging.getLogger(logName)
    
        # Create file handler and set level to info
        ch = RotatingFileHandler(logFilename, mode='a', maxBytes=logFilesize, backupCount=logBackupCount, encoding=None, delay=0)
        #ch = logging.FileHandler(logFilename)
        logger.setLevel(logLevel)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s:%(funcName)s - %(message)s')
    
        # Add formatter to ch
        ch.setFormatter(formatter)
    
        # Add ch to logger
        logger.addHandler(ch)
        
        print("Log file is in: ", ch.baseFilename)
        self.myLogger = logger

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
    def calculateILPSelection(self, phrase, graphResultsForPhraseToken=None, graphResultsForPhraseRelation=None, graphResultsForPhraseTripleRelation=None): pass
    
    @abc.abstractclassmethod
    def inferILPConstrains(self, model_trail, *conceptsRelations): pass