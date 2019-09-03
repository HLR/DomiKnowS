if __package__ is None or __package__ == '':
    from regr.solver.ilpConfig import ilpConfig 
    from regr.solver.ilpOntSolver import ilpOntSolver
    from regr.solver.gurobiILPOntSolver import gurobiILPOntSolver
    from regr.solver.gekkoILPOntSolver import gekkoILPOntSolver
    from regr.solver.dummyILPOntSolver import dummyILPOntSolver
else:
    from .ilpConfig import ilpConfig 
    from .ilpOntSolver import ilpOntSolver 
    from .gurobiILPOntSolver import gurobiILPOntSolver
    from .gekkoILPOntSolver import gekkoILPOntSolver
    from .dummyILPOntSolver import dummyILPOntSolver

class ilpOntSolverFactory:
    __instances = {}
    __classes = {}
    
    @classmethod
    def getClass(cls, *SolverClasses):
        if SolverClasses not in cls.__classes:
            class ImplClass(*SolverClasses): pass
            cls.__classes[SolverClasses] = ImplClass
        return cls.__classes[SolverClasses]

    @staticmethod
    def getOntSolverInstance(graph, *SupplementalClasses, _iplConfig = ilpConfig):
        if (graph is not None) and (graph.ontology is not None):
            
            if graph.ontology.iri not in ilpOntSolverFactory.__instances:
                
                if _iplConfig is not None:
                    if ilpConfig['ilpSolver'] == "Gurobi":
                        SolverClass = ilpOntSolverFactory.getClass(gurobiILPOntSolver, *SupplementalClasses)
                    elif ilpConfig['ilpSolver'] == "GEKKO":
                        SolverClass = ilpOntSolverFactory.getClass(gekkoILPOntSolver, *SupplementalClasses)
                    else:
                        SolverClass = ilpOntSolverFactory.getClass(dummyILPOntSolver, *SupplementalClasses)
                    ilpOntSolverFactory.__instances[graph.ontology.iri] = SolverClass()

                ilpOntSolverFactory.__instances[graph.ontology.iri].setup_solver_logger() 

                ilpOntSolverFactory.__instances[graph.ontology.iri].myGraph = graph
                    
                ilpOntSolverFactory.__instances[graph.ontology.iri].loadOntology(graph.ontology.iri, graph.ontology.local)
                
                ilpOntSolverFactory.__instances[graph.ontology.iri].ilpSolver = _iplConfig['ilpSolver']
                
                ilpOntSolverFactory.__instances[graph.ontology.iri].myLogger.info("Returning new ilpOntSolver for %s using %s"%(graph.ontology.iri,_iplConfig['ilpSolver']))
                return ilpOntSolverFactory.__instances[graph.ontology.iri]
            else:
                ilpOntSolverFactory.__instances[graph.ontology.iri].myLogger.info("Returning existing ilpOntSolver for %s using %s"%(graph.ontology.iri,_iplConfig['ilpSolver']))
            
                return ilpOntSolverFactory.__instances[graph.ontology.iri]
        else:
            return None