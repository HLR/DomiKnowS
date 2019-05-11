# numpy
import numpy as np

# pandas
import pandas as pd

# Gurobi
from gurobipy import *

# ontology
from owlready2 import *
from pathlib import Path

if __package__ is None or __package__ == '':
    from regr.concept import Concept, enum
    from regr.graph import Graph
    from regr.relation import Relation, Be, Have
else:
    from .concept import Concept, enum
    from .graph import Graph
    from .relation import Relation, Be, Have

def loadOntology(ontologyURL, ontologyPathname = "./"):
    # Check if ontology path is correct
    ontologyPath = Path(os.path.normpath(ontologyPathname))
    ontologyPath = ontologyPath.resolve()
    if not os.path.isdir(ontologyPath):
        print("Path to load ontology:", ontologyPath, "does not exists")
        exit()

    onto_path.append(ontologyPath) # the folder with the ontology

    # Load ontology
    try :
        myOnto = get_ontology(ontologyURL)
        myOnto.load(only_local = True, fileobj = None, reload = False, reload_if_newer = False)
    except FileNotFoundError as e:
        print('Error when loading - ' + ontologyURL + " from: %s"%(ontologyPath))

    return myOnto
        
def addTokenConstrains(m, myOnto, tokens, conceptNames, x, graphResultsForPhraseToken):
        
    for token in tokens:            
        for conceptName in conceptNames: 
            x[token, conceptName]=m.addVar(vtype=GRB.BINARY,name="x_%s_%s"%(token, conceptName))
            
    m.update()
     
    # Add constraints based on concept disjoint statments in ontology
    foundDisjoint = dict() # too eliminate duplicates
    for conceptName in conceptNames:
        
        currentConcept = myOnto.search_one(iri = "*%s"%(conceptName))
            
        if currentConcept is None :
            continue
            
        for d in currentConcept.disjoints():
            disjointConcept = d.entities[1]._name
                
            if currentConcept._name == disjointConcept:
                disjointConcept = d.entities[0]._name
                    
                if currentConcept._name == disjointConcept:
                    continue
                    
            if disjointConcept not in graphResultsForPhraseToken.columns:
                 continue
                    
            if conceptName in foundDisjoint:
                if disjointConcept in foundDisjoint[conceptName]:
                    continue
            
            if disjointConcept in foundDisjoint:
                if conceptName in foundDisjoint[disjointConcept]:
                    continue
                        
            for token in tokens:
                constrainName = 'c_%s_%sDisjoint%s'%(token, currentConcept, disjointConcept)
                m.addConstr(x[token, conceptName] + x[token, disjointConcept], GRB.LESS_EQUAL, 1, name=constrainName)
                  
            #print("disjointConcept %s %s"%(currentConcept._name, disjointConcept))   
                   
            if not (conceptName in foundDisjoint):
                foundDisjoint[conceptName] = {disjointConcept}
            else:
                foundDisjoint[conceptName].add(disjointConcept)
        
    m.update()
            
    X_Q = None
    for token in tokens :
        for conceptName in conceptNames :
            X_Q += graphResultsForPhraseToken[conceptName][token]*x[token, conceptName]
    
    return X_Q
    
def addRelationsConstrains(m, myOnto, tokens, conceptNames, x, y, graphResultsForPhraseRelation):
    
    relationNames = graphResultsForPhraseRelation.keys()
        
    for relationName in relationNames:            
        for token in tokens: 
            for token1 in tokens:
                if token == token1:
                    continue

                y[relationName, token, token1]=m.addVar(vtype=GRB.BINARY,name="y_%s_%s_%s"%(relationName, token, token1))
          
    m.update()

    # Add constraints based on relations domain and range
    for relationName in graphResultsForPhraseRelation :
        currentRelation = myOnto.search_one(iri = "*%s"%(relationName))
            
        if currentRelation is None:
            continue

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
                        if token == token1 :
                            continue
                                
                        constrainName = 'c_%s_%s_%s'%(currentRelation, token, token1)
                        currentConstrain = y[currentRelation._name, token, token1] + x[token, domain._name] + x[token1, range._name]
                        #print(currentConstrain)
                        m.addConstr(currentConstrain, GRB.GREATER_EQUAL, 3 * y[currentRelation._name, token, token1], name=constrainName)
        
    m.update()     
       
    Y_Q  = None
    for relationName in relationNames:
        for token in tokens:
            for token1 in tokens:
                if token == token1 :
                    continue

                Y_Q += graphResultsForPhraseRelation[relationName][token][token1]*y[relationName, token, token1]
    
    return Y_Q
    
def calculateIPLSelection(phrase, graph, graphResultsForPhraseToken, graphResultsForPhraseRelation, ontologyPathname = "./"):

    tokenResult = None
    relationsResult = None
    
    try:
        # Create a new Gurobi model
        m = Model("decideOnClassificationResult")
        m.params.outputflag = 0
        
        myOnto = loadOntology(graph.ontology, ontologyPathname)

        # Get list of tokens, concepts and relations from panda dataframe graphResultsForPhraseToken
        tokens = graphResultsForPhraseToken.index.tolist()
        conceptNames = graphResultsForPhraseToken.columns.tolist()
        
        # Create Gurobi variables for concept - token
        x={}

        # Create Gurobi variables for relation - token, token
        y={}
            
        # -- Set objective - maximize 
        Q = None
        
        X_Q = addTokenConstrains(m, myOnto, tokens, conceptNames, x, graphResultsForPhraseToken)
        if X_Q is not None:
            Q += X_Q
        
        Y_Q = addRelationsConstrains(m, myOnto, tokens, conceptNames, x, y, graphResultsForPhraseRelation)
        if Y_Q is not None:
            Q += Y_Q
        
        m.setObjective(Q, GRB.MAXIMIZE)

        # Token is associated with a single concept
        #for token in tokens:
        #   constrainName = 'c_%s'%(token)
        #    m.addConstr(quicksum(x[token, conceptName] for conceptName in conceptNames), GRB.LESS_EQUAL, 1, name=constrainName)
        
        m.update()
          
        #print(m)
        #print(m.getObjective())
        #print(m.getConstrs())
        
        m.optimize()
        
        #print('Obj: %g' % m.objVal)

        #for v in m.getVars():
        #    print('%s %g' % (v.varName, v.x))
          
        # Collect results
        tokenResult = pd.DataFrame(0, index=tokens, columns=conceptNames)
        if x or True:
            if m.status == GRB.Status.OPTIMAL:
                solution = m.getAttr('x', x)
                
                for token in tokens :
                    for conceptName in conceptNames:
                        if solution[token, conceptName] == 1:
                            #print("The  %s is classified as %s" % (token, conceptName))
                            
                            tokenResult[conceptName][token] = 1

        relationsResult = {}
        if y or True:
            if m.status == GRB.Status.OPTIMAL:
                solution = m.getAttr('x', y)
                relationNames = graphResultsForPhraseRelation.keys()
                
                for relationName in relationNames:
                    relationResult = pd.DataFrame(0, index=tokens, columns=tokens)
                    
                    for token in tokens :
                        for token1 in tokens:
                            if token == token1:
                                continue
                            
                            if solution[relationName, token, token1] == 1:
                                relationResult[token][token1] = 1
                                
                    relationsResult[relationName] = relationResult

        #print(m)

    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
       
    # return results of ILP optimization
    return tokenResult, relationsResult

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
    
    #                         peop  org   other loc   O
    phrase_table = np.array([[0.70, 0.98, 0.95, 0.02, 0.00], # John
                             [0.00, 0.50, 0.40, 0.60, 0.90], # works
                             [0.02, 0.03, 0.05, 0.10, 0.90], # for
                             [0.92, 0.93, 0.93, 0.90, 0.00], # IBM
                            ])
    test_graphResultsForPhraseToken = pd.DataFrame(phrase_table, index=tokenList, columns=conceptNamesList)
    
    test_graphResultsForPhraseRelation = dict()
    
    # work_for
    #                                    John  works for   IBM
    work_for_relation_table = np.array([[0.50, 0.20, 0.20, 0.26], # John
                                        [0.00, 0.00, 0.40, 0.30], # works
                                        [0.02, 0.03, 0.05, 0.10], # for
                                        [0.63, 0.20, 0.10, 0.90], # IBM
                                       ])
    work_for_current_graphResultsForPhraseRelation = pd.DataFrame(work_for_relation_table, index=tokenList, columns=tokenList)
    test_graphResultsForPhraseRelation["work_for"] = work_for_current_graphResultsForPhraseRelation
    
    # live_in
    #                                   John  works for   IBM
    live_in_relation_table = np.array([[0.10, 0.20, 0.20, 0.06], # John
                                       [0.00, 0.00, 0.20, 0.10], # works
                                       [0.02, 0.03, 0.05, 0.10], # for
                                       [0.10, 0.20, 0.10, 0.00], # IBM
                                       ])
    live_in_current_graphResultsForPhraseRelation = pd.DataFrame(live_in_relation_table, index=tokenList, columns=tokenList)
    test_graphResultsForPhraseRelation["live_in"] = live_in_current_graphResultsForPhraseRelation
        
    # located_in
    #                                      John  works for   IBM
    located_in_relation_table = np.array([[0.10, 0.20, 0.20, 0.06], # John
                                          [0.00, 0.00, 0.00, 0.00], # works
                                          [0.02, 0.03, 0.05, 0.10], # for
                                          [0.03, 0.20, 0.10, 0.00], # IBM
                                         ])
    located_in_current_graphResultsForPhraseRelation = pd.DataFrame(located_in_relation_table, index=tokenList, columns=tokenList)
    test_graphResultsForPhraseRelation["located_in"] = located_in_current_graphResultsForPhraseRelation
        
    tokenResult, relationsResult = calculateIPLSelection(test_phrase, test_graph, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation, ontologyPathname="./examples/emr/")
    
    print("\nResults - ")
    print(tokenResult)
    
    for name, result in relationsResult.items():
        print("\n")
        print(name)
        print(result)
    
if __name__ == '__main__' :
    main()