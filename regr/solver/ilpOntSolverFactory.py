if __package__ is None or __package__ == '':
    from regr.solver.ilpConfig import ilpConfig 
    from regr.solver.ilpOntSolver import ilpOntSolver
else:
    from .ilpConfig import ilpConfig 
    from .ilpOntSolver import ilpOntSolver

class ilpOntSolverFactory:
    __instances = {}
    __classes = {}
    
    @classmethod
    def getClass(cls, *SolverClasses):
        if SolverClasses not in cls.__classes:
            class ImplClass(*SolverClasses): pass
            cls.__classes[SolverClasses] = ImplClass
        return cls.__classes[SolverClasses]

    @classmethod
    def getOntSolverInstance(cls, graph, *SupplementalClasses, _iplConfig=ilpConfig) -> ilpOntSolver:
        if (graph is not None) and (graph.ontology is not None):

            if graph.ontology.iri not in cls.__instances:

                if _iplConfig is not None:
                    if ilpConfig['ilpSolver'] == "Gurobi":
                        if __package__ is None or __package__ == '':
                            from regr.solver.gurobiILPOntSolver import gurobiILPOntSolver
                        else:
                            from .gurobiILPOntSolver import gurobiILPOntSolver
                        SolverClass = cls.getClass(gurobiILPOntSolver, *SupplementalClasses)
                    elif ilpConfig['ilpSolver'] == "GEKKO":
                        if __package__ is None or __package__ == '':
                            from regr.solver.gekkoILPOntSolver import gekkoILPOntSolver
                        else:
                            from .gekkoILPOntSolver import gekkoILPOntSolver
                        SolverClass = cls.getClass(gekkoILPOntSolver, *SupplementalClasses)
                    else:
                        if __package__ is None or __package__ == '':
                            from regr.solver.dummyILPOntSolver import dummyILPOntSolver
                        else:
                            from .dummyILPOntSolver import dummyILPOntSolver
                        SolverClass = cls.getClass(dummyILPOntSolver, *SupplementalClasses)
                    cls.__instances[graph.ontology.iri] = SolverClass()

                cls.__instances[graph.ontology.iri].setup_solver_logger() 

                cls.__instances[graph.ontology.iri].myGraph = graph

                cls.__instances[graph.ontology.iri].loadOntology(graph.ontology.iri, graph.ontology.local)

                cls.__instances[graph.ontology.iri].ilpSolver = _iplConfig['ilpSolver']

                cls.__instances[graph.ontology.iri].myLogger.info("Returning new ilpOntSolver for %s using %s"%(graph.ontology.iri,_iplConfig['ilpSolver']))
                return cls.__instances[graph.ontology.iri]
            else:
                cls.__instances[graph.ontology.iri].myLogger.info("Returning existing ilpOntSolver for %s using %s"%(graph.ontology.iri,_iplConfig['ilpSolver']))

                return cls.__instances[graph.ontology.iri]
        else:
            return None
