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
        if graph is None:
            return None
        
        if not isinstance(graph, set):
            graph = {graph}
            
        ontologies = []
        
        for currentGraph in graph:
            if currentGraph.ontology is not None:
                if currentGraph.ontology not in ontologies:
                    ontologies.append(currentGraph.ontology)
        
        if not ontologies:
            return None
            
        ontologiesTuple = (*ontologies, )
        
        if ontologiesTuple not in cls.__instances:

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
                    
                cls.__instances[ontologiesTuple] = SolverClass()

            cls.__instances[ontologiesTuple].setup_solver_logger() 

            cls.__instances[ontologiesTuple].myGraph = graph

            cls.__instances[ontologiesTuple].loadOntology(ontologiesTuple)

            cls.__instances[ontologiesTuple].ilpSolver = _iplConfig['ilpSolver']

            cls.__instances[ontologiesTuple].myLogger.info("Returning new ilpOntSolver for %s using %s"%(ontologiesTuple,_iplConfig['ilpSolver']))
            return cls.__instances[ontologiesTuple]
        else:
            cls.__instances[ontologiesTuple].myLogger.info("Returning existing ilpOntSolver for %s using %s"%(ontologiesTuple,_iplConfig['ilpSolver']))

            return cls.__instances[ontologiesTuple]
