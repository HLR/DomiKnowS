if __package__ is None or __package__ == '':
    from domiknows.solver.ilpConfig import ilpConfig 
    from domiknows.solver.ilpOntSolver import ilpOntSolver
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
    def getOntSolverInstance(cls, graph, *SupplementalClasses, _ilpConfig=ilpConfig, **kwargs) -> ilpOntSolver:
        if graph is None:
            return None
        
        if not isinstance(graph, set):
            graph = {graph}
            
        ontologies = []
        reuse_model = False
        
        graphOntologyError = set()
        for currentGraph in graph:
            if (hasattr(currentGraph, 'ontology')) and (currentGraph.ontology is not None) and (currentGraph.ontology.iri is not None):
                if currentGraph.ontology not in ontologies:
                    ontologies.append(currentGraph.ontology)
            else:
                graphOntologyError.add(currentGraph)
                
            if currentGraph.reuse_model:
                reuse_model = True
                
        ontologiesTuple = (*ontologies, )
        
        if _ilpConfig is not None:
            if _ilpConfig['ilpSolver'] == "Gurobi":
                if reuse_model:
                    kwargs['reuse_model'] = True

                if __package__ is None or __package__ == '':
                    from domiknows.solver.gurobiILPOntSolver import gurobiILPOntSolver
                else:
                    from .gurobiILPOntSolver import gurobiILPOntSolver
                SolverClass = cls.getClass(gurobiILPOntSolver, *SupplementalClasses)
            elif _ilpConfig['ilpSolver'] == "Gurobi1":
                if reuse_model:
                    kwargs['reuse_model'] = True

                if __package__ is None or __package__ == '':
                    from domiknows.solver.gurobiILPOntSolver1 import gurobiILPOntSolver
                else:
                    from .gurobiILPOntSolver1 import gurobiILPOntSolver
                SolverClass = cls.getClass(gurobiILPOntSolver, *SupplementalClasses)
            elif _ilpConfig['ilpSolver'] == "GEKKO":
                if __package__ is None or __package__ == '':
                    from domiknows.solver.gekkoILPOntSolver import gekkoILPOntSolver
                else:
                    from .gekkoILPOntSolver import gekkoILPOntSolver
                SolverClass = cls.getClass(gekkoILPOntSolver, *SupplementalClasses)
            elif _ilpConfig['ilpSolver'] == "mini_prob_debug":
                if __package__ is None or __package__ == '':
                    from domiknows.solver.mini_solver_debug import MiniProbSolverDebug
                else:
                    from .mini_solver_debug import MiniProbSolverDebug
                SolverClass = cls.getClass(MiniProbSolverDebug, *SupplementalClasses)
            elif _ilpConfig['ilpSolver'] == "mini_debug":
                if __package__ is None or __package__ == '':
                    from domiknows.solver.mini_solver_debug import MiniSolverDebug
                else:
                    from .mini_solver_debug import MiniSolverDebug
                SolverClass = cls.getClass(MiniSolverDebug, *SupplementalClasses)
            else:
                if __package__ is None or __package__ == '':
                    from domiknows.solver.dummyILPOntSolver import dummyILPOntSolver
                else:
                    from .dummyILPOntSolver import dummyILPOntSolver
                SolverClass = cls.getClass(dummyILPOntSolver, *SupplementalClasses)

        key = (SolverClass, ontologiesTuple, frozenset(_ilpConfig), frozenset(kwargs))
        if key not in cls.__instances:
            instance = SolverClass(graph, ontologiesTuple, _ilpConfig, **kwargs)
            cls.__instances[key] = instance

            if ontologiesTuple and graphOntologyError:
                for currentGraph in graphOntologyError:
                    instance.myLogger.error("Problem - graph %s ontology is not correctly defined %s"%(currentGraph.name,currentGraph.ontology))

            if ontologiesTuple:
                instance.myLogger.info("Returning new ilpOntSolver for %s using %s"%(ontologiesTuple,_ilpConfig['ilpSolver']))
            else:
                instance.myLogger.debug("Returning generic (not based on ontology) ilpOntSolver using %s"%(_ilpConfig['ilpSolver']))

            return instance
        else:
            instance = cls.__instances[key]
            if ontologiesTuple:
                instance.myLogger.info("Returning existing ilpOntSolver for %s using %s"%(ontologiesTuple,_ilpConfig['ilpSolver']))
            else:
                instance.myLogger.debug("Returning existing generic ilpOntSolver using %s"%(_ilpConfig['ilpSolver']))

            return instance
