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
    def getOntSolverInstance(cls, graph, *SupplementalClasses, _iplConfig=ilpConfig, **kwargs) -> ilpOntSolver:
        if graph is None:
            return None

        if not isinstance(graph, set):
            graph = {graph}

        ontologies = []

        graphOntologyError = set()
        for currentGraph in graph:
            if (currentGraph.ontology is not None) and (currentGraph.ontology.iri is not None):
                if currentGraph.ontology not in ontologies:
                    ontologies.append(currentGraph.ontology)
            else:
                graphOntologyError.add(currentGraph)

        if not ontologies:
            return None

        ontologiesTuple = (*ontologies, )

        if _iplConfig['ilpSolver'] == "Gurobi":
            if __package__ is None or __package__ == '':
                from regr.solver.gurobiILPOntSolver import gurobiILPOntSolver
            else:
                from .gurobiILPOntSolver import gurobiILPOntSolver
            SolverClass = cls.getClass(gurobiILPOntSolver, *SupplementalClasses)
        elif _iplConfig['ilpSolver'] == "GEKKO":
            if __package__ is None or __package__ == '':
                from regr.solver.gekkoILPOntSolver import gekkoILPOntSolver
            else:
                from .gekkoILPOntSolver import gekkoILPOntSolver
            SolverClass = cls.getClass(gekkoILPOntSolver, *SupplementalClasses)
        elif _iplConfig['ilpSolver'] == "mini":
            if __package__ is None or __package__ == '':
                from regr.solver.gurobi_solver import GurobiSolver
            else:
                from .gurobi_solver import GurobiSolver
            SolverClass = cls.getClass(GurobiSolver, *SupplementalClasses)
        elif _iplConfig['ilpSolver'] == "mini_debug":
            if __package__ is None or __package__ == '':
                from regr.solver.gurobi_solver_debug import GurobiSolverDebug
            else:
                from .gurobi_solver_debug import GurobiSolverDebug
            SolverClass = cls.getClass(GurobiSolverDebug, *SupplementalClasses)
        elif _iplConfig['ilpSolver'] == "mini_log_debug":
            if __package__ is None or __package__ == '':
                from regr.solver.gurobi_log_solver_debug import GurobilogSolverDebug
            else:
                from .gurobi_log_solver_debug import GurobilogSolverDebug
            SolverClass = cls.getClass(GurobilogSolverDebug, *SupplementalClasses)
        else:
            if __package__ is None or __package__ == '':
                from regr.solver.dummyILPOntSolver import dummyILPOntSolver
            else:
                from .dummyILPOntSolver import dummyILPOntSolver
            SolverClass = cls.getClass(dummyILPOntSolver, *SupplementalClasses)

        key = (SolverClass, ontologiesTuple, frozenset(kwargs))
        if key not in cls.__instances:
            instance = SolverClass(graph, ontologiesTuple, **kwargs)
            cls.__instances[key] = instance

            if graphOntologyError:
                for currentGraph in graphOntologyError:
                    instance.myLogger.error("Problem graph %s ontology is not correctly defined %s"%(currentGraph.name,currentGraph.ontology))

            instance.myLogger.info("Creating new ilpOntSolver for %s using %s"%(ontologiesTuple,_iplConfig['ilpSolver']))

        instance = cls.__instances[key]
        instance.myLogger.info("Returning ilpOntSolver for %s using %s"%(ontologiesTuple,_iplConfig['ilpSolver']))

        return instance
