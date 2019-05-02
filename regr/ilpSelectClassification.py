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
        
def calculateIPLSelection(phrase, graph, graphResultsForPhraseToken, graphResultsForPhraseRelation):
    result = dict()

    try:
        # Create a new Gurobi model
        m = Model("decideOnClassificationResult")
        
        # Get list of tokens, concepts and relations from panda dataframe graphResultsForPhraseToken
        tokens = graphResultsForPhraseToken.index.tolist()
        conceptNames = graphResultsForPhraseToken.columns.tolist()
        relationNames = graphResultsForPhraseRelation.keys()
        
        # Create Gurobi variables for concept - token
        x={}
        
        for token in tokens:            
            for conceptName in conceptNames: 
                x[token, conceptName]=m.addVar(vtype=GRB.BINARY,name="x_%s_%s"%(token, conceptName))
                
        # Create Gurobi variables for relation - token, token
        y={}
        
        for relationName in relationNames:            
            for token in tokens: 
                for token1 in tokens:
                    y[relationName, token, token1]=m.addVar(vtype=GRB.BINARY,name="y_%s_%s_%s"%(relationName, token, token1))
          
        m.update()
            
        # -- Set objective
        # maximize 
        X_Q = quicksum(quicksum(graphResultsForPhraseToken[conceptName][token]*x[token, conceptName] for conceptName in conceptNames)for token in tokens)
        Y_Q = quicksum(quicksum(quicksum(graphResultsForPhraseRelation[relationName][token][token1]*y[relationName, token, token1] for relationName in relationNames)for token in tokens)for token in tokens)
        m.setObjective(X_Q + Y_Q, GRB.MAXIMIZE)
            
        # -- Add constraints
        myOnto = loadOntology(graph.ontology)
        
        # Add constraints based on concept disjoint statments in ontology
        foundDisjoint = dict() # too eliminate duplicates
        for conceptName in conceptNames:
            currentConcept = myOnto.search_one(iri = "*%s"%(conceptName))
            
            if not (currentConcept is None):
                for d in currentConcept.disjoints():
                    disjointConcept = d.entities[1]._name
                    
                    if disjointConcept in foundDisjoint:
                        if conceptName in foundDisjoint[disjointConcept]:
                            continue
                        
                    for token in tokens:
                        constrainName = 'c_%s_%sDisjoint%s'%(token, currentConcept, disjointConcept)
                        m.addConstr(x[token, conceptName] + x[token, disjointConcept], GRB.LESS_EQUAL, 1, name=constrainName)
                        
                    if not (conceptName in foundDisjoint):
                        foundDisjoint[conceptName] = {disjointConcept}
                    else:
                        foundDisjoint[conceptName].add(disjointConcept)
                        
        # Add constraints based on relations domain and range
        for relationName in graphResultsForPhraseRelation :
            currentRelation = myOnto.search_one(iri = "*%s"%(relationName))
            
            if not (currentRelation is None):
                currentRelationDomain = currentRelation.get_domain() # domains_indirect()
                currentRelationRange = currentRelation.get_range()
                
                for domain in currentRelationDomain:
                    if domain._name not in conceptNames:
                        continue
                    
                    for range in currentRelationRange:
                        if range.name not in conceptNames:
                            continue
                        
                        for token in tokens:
                            for token1 in tokens:
                                constrainName = 'c_%s_%s_%s'%(currentRelation, token, token1)
                                m.addConstr(y[currentRelation._name, token, token1] + x[token, domain._name] + x[token1, range._name], GRB.GREATER_EQUAL, 3 * y[currentRelation._name, token, token1], name=constrainName)
                
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
    conceptNamesList = ["people", "organization", "other", "location", "O"]
    relationNamesList = ["work_for", "live_in", "located_in"]
    
    test_graphResultsForPhraseToken = pd.DataFrame(np.random.random_sample((len(tokenList), len(conceptNamesList))), index=tokenList, columns=conceptNamesList)
    
    test_graphResultsForPhraseRelation = dict()
    for relationName in relationNamesList :
        current_graphResultsForPhraseRelation = pd.DataFrame(np.random.random_sample((len(tokenList), len(tokenList))), index=tokenList, columns=tokenList)
        test_graphResultsForPhraseRelation[relationName] = current_graphResultsForPhraseRelation
    
    iplResults = calculateIPLSelection(test_phrase, test_graph, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation)
    print("\nResults - ", iplResults)
    
if __name__ == '__main__' :
    main()