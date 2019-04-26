# numpy
import numpy as np

# pandas
import pandas as pd

# Gurobi
from gurobipy import *

# ontology
from owlready2 import *
from pathlib import Path

from concept import Concept, enum
from graph import Graph
from relation import Relation, Be, Have

def loadOntology(ontologyURL):
    # Check if ontology path is correct
    ontologyPath = Path(os.path.normpath("./examples/emr"))
    if not os.path.isdir(ontologyPath.resolve()):
        print("Path to load ontology:", ontologyPath.resolve(), "does not exists")
        exit()

    onto_path.append(ontologyPath) # the folder with the ontology

    # Load ontology
    myOnto = get_ontology(ontologyURL)
    myOnto.load(only_local = True, fileobj = None, reload = False, reload_if_newer = False)
    
    return myOnto
        
def calculateIPLSelection(phrase, graph, graphResultsForPrhase):
    result = dict()

    try:
        # Create a new Gurobi model
        m = Model("decideOnClassificationResult")
        
        # get list of tokens and concepts from panda dataframe graphResultsForPrhase
        tokens = graphResultsForPrhase.index.tolist()
        conceptNames = graphResultsForPrhase.columns.tolist()
        
        # Create Gurobi variables
        x={}
        
        for token in tokens:            
            for conceptName in conceptNames: 
                x[token, conceptName]=m.addVar(vtype=GRB.BINARY,name="x_%s_%s"%(token, conceptName))
          
        m.update()
            
        # -- Set objective
        # maximize 
        m.setObjective(quicksum(quicksum(graphResultsForPrhase[conceptName][token]*x[token, conceptName] for conceptName in conceptNames)for token in tokens), GRB.MAXIMIZE)
            
        # -- Add constraints
        myOnto = loadOntology(graph.ontology)
        
        # Add constraints based on concept disjoint statments in ontology
        foundDisjoint = dict()
        for conceptName in conceptNames:
            currentConcept = myOnto.search_one(iri = "*%s"%(conceptName))
            
            if not (currentConcept is None):
                for d in currentConcept.disjoints():
                    disjointConcept = d.entities[1]._name
                    
                    if disjointConcept in foundDisjoint:
                        if conceptName in foundDisjoint[disjointConcept] :
                            continue
                        
                    for token in tokens:
                        constrainName = 'c_%s_%sDisjoint%s'%(token, currentConcept, disjointConcept)
                        m.addConstr(x[token, conceptName] + x[token, disjointConcept], GRB.LESS_EQUAL, 1, name=constrainName)
                        
                    if not (conceptName in foundDisjoint):
                        foundDisjoint[conceptName] = {disjointConcept}
                    else :
                        foundDisjoint[conceptName].add(disjointConcept)
                    
        # Token is associated with a single concept
        #for token in tokens:
        #   constrainName = 'c_%s'%(token)
        #    m.addConstr(quicksum(x[token, conceptName] for conceptName in conceptNames), GRB.LESS_EQUAL, 1, name=constrainName)
        
        m.update()
          
        print(m)
        print(m.getObjective())
        print(m.getConstrs())
        
        m.optimize()
        
        print('Obj: %g' % m.objVal)

        for v in m.getVars():
            print('%s %g' % (v.varName, v.x))
          
        # Collect results
        if m.status == GRB.Status.OPTIMAL:
            solution = m.getAttr('x', x)
            
            for token in tokens :
                for conceptName in conceptNames:
                    if solution[token, conceptName] == 1:
                        #print("The  %s is classified as %s" % (token, conceptName))
                        
                        result[token] = conceptName
                
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
       
    # return results of ipl optimization 
    return result

# --------- Testing

with Graph('global') as graph:
    graph.ontology='http://ontology.ihmc.us/ML/EMR.owl'
        
    with Graph('linguistic') as ling_graph:
        ling_graph.ontology='http://ontology.ihmc.us/ML/EMR.owl'
        phrase = Concept(name='phrase')
            
    with Graph('application') as app_graph:
        app_graph.ontology='http://ontology.ihmc.us/ML/EMR.owl'
            
        entity = Concept(name='entity')
        entity.be(phrase)
        people = Concept(name='people')
        people.be(entity)
        organization = Concept(name='organization')
        organization.be(entity)
        other = Concept(name='other')
        other.be(entity)
        location = Concept(name='location')
        location.be(entity)
        O = Concept(name='O')
        O.be(entity)
    
        work_for = Concept(name='work_for')
        work_for.be((people, organization))
    
        live_in = Concept(name='live_in')
        live_in.be((people, location))
    
        orgbase_on = Concept(name='orgbase_on')
        orgbase_on.be((organization, location))
    
        located_in = Concept(name='located_in')
        located_in.be((organization, location))

def main() :
    test_phrase = [("John", "NNP"), ("works", "VBN"), ("for", "IN"), ("IBM", "NNP")]

    test_graph = app_graph
    tokenList = ["John", "works", "for", "IBM"]
    conceptNamesList = ["people", "organization", "work_for", "other", "location", "O"]
    
    test_graphResultsForPrhase = pd.DataFrame(np.random.random_sample((len(tokenList), len(conceptNamesList))), index=tokenList, columns=conceptNamesList)
    
    iplesults = calculateIPLSelection(test_phrase, test_graph, test_graphResultsForPrhase)
    print("\nResults - ", iplesults)
    
if __name__ == '__main__' :
    main()