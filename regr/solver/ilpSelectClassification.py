# numpy
import numpy as np

# pandas
import pandas as pd

# Gurobi
from gurobipy import *

# ontology
from owlready2 import *

# path to Meta Graph ontology
graphMetaOntologyPathname = "./ontology/ML/"

# path
from pathlib import Path

if __package__ is None or __package__ == '':
    from regr.graph.concept import Concept, enum
    from regr.graph.graph import Graph
    #from regr.graph.relation import Relation, IsA, HasA
else:
    from ..graph.concept import Concept, enum
    from ..graph.graph import Graph
    #from ..graph.relation import Relation, IsA, HasA

def loadOntology(ontologyURL, ontologyPathname = "./"):
    
    currentPath = Path(os.path.normpath("./")).resolve()
    
    # Check if Graph Meta ontology path is correct
    graphMetaOntologyPath = Path(os.path.normpath(graphMetaOntologyPathname))
    graphMetaOntologyPath = graphMetaOntologyPath.resolve()
    if not os.path.isdir(graphMetaOntologyPath):
        print("Path to load Graph ontology: %s does not exists in current directory %s"%(graphMetaOntologyPath,currentPath))
        exit()
        
    # Check if specific ontology path is correct
    ontologyPath = Path(os.path.normpath(ontologyPathname))
    ontologyPath = ontologyPath.resolve()
    if not os.path.isdir(ontologyPath):
        print("Path to load ontology: %s does not exists in current directory %s"%(ontologyPath,currentPath))
        exit()

    onto_path.append(graphMetaOntologyPath)  # the folder with the Graph Meta ontology
    onto_path.append(ontologyPath) # the folder with the ontology for the specific  graph

    # Load specific ontology
    try :
        myOnto = get_ontology(ontologyURL)
        myOnto.load(only_local = True, fileobj = None, reload = False, reload_if_newer = False)
    except FileNotFoundError as e:
        print("Error when loading - %s from: %s"%(ontologyURL,ontologyURL,ontologyPath))

    return myOnto

def addTokenConstrains(m, myOnto, tokens, conceptNames, x, graphResultsForPhraseToken):
        
    # Create variables for token - concept and negative variables
    for token in tokens:            
        for conceptName in conceptNames: 
            x[token, conceptName]=m.addVar(vtype=GRB.BINARY,name="x_%s_%s"%(token, conceptName))
            x[token, conceptName+'-neg']=m.addVar(vtype=GRB.BINARY,name="x_%s_%s"%(token, conceptName+'-neg'))

    # Add constraints forcing decision between variable and negative variables 
    for conceptName in conceptNames:
        for token in tokens:
            constrainName = 'c_%s_%sselfDisjoint'%(token, conceptName)
            m.addConstr(x[token, conceptName] + x[token, conceptName+'-neg'], GRB.LESS_EQUAL, 1, name=constrainName)
            
    m.update()
     
    # -- Add constraints based on concept disjoint statements in ontology - not(and(var1, var2)) = nand(var1, var2)
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
                constrainName = 'c_%s_%s_Disjoint_%s'%(token, conceptName, disjointConcept)
                m.addConstr(x[token, conceptName] + x[token, disjointConcept], GRB.LESS_EQUAL, 1, name=constrainName)
                  
            #print("disjointConcept %s %s"%(currentConcept._name, disjointConcept))   
                        
            if not (conceptName in foundDisjoint):
                foundDisjoint[conceptName] = {disjointConcept}
            else:
                foundDisjoint[conceptName].add(disjointConcept)
       
    # -- Add constraints based on concept equivalent (sameAs) statements in ontology - and(var1, av2)
    foundEquivalent = dict() # too eliminate duplicates
    for conceptName in conceptNames:
        
        currentConcept = myOnto.search_one(iri = "*%s"%(conceptName))
            
        if currentConcept is None :
            continue
            
        for equivalentConcept in currentConcept.equivalent_to:
            if equivalentConcept.name not in graphResultsForPhraseToken.columns:
                 continue
                    
            if conceptName in foundEquivalent:
                if equivalentConcept.name in foundEquivalent[conceptName]:
                    continue
            
            if equivalentConcept.name in foundEquivalent:
                if conceptName in foundEquivalent[equivalentConcept.name]:
                    continue
                        
            for token in tokens:
                constrainName = 'c_%s_%s_Equivalent_%s'%(token, conceptName, equivalentConcept.name)
                _varAnd = m.addVar(name="andVar_%s"%(constrainName))
    
                m.addGenConstrAnd(_varAnd, [x[token, conceptName], x[token, equivalentConcept.name]], constrainName)
                #m.addConstr(x[token, conceptName] + x[token, equivalentConcept], GRB.LESS_EQUAL, 1, name=constrainName) #?
                                         
            if not (conceptName in foundEquivalent):
                foundEquivalent[conceptName] = {equivalentConcept.name}
            else:
                foundEquivalent[conceptName].add(equivalentConcept.name)
       

    # -- Add constraints based on concept subClassOf statements in ontology - var1 -> var2
    for conceptName in conceptNames :
        
        currentConcept = myOnto.search_one(iri = "*%s"%(conceptName))
            
        if currentConcept is None :
            continue
            
        for ancestorConcept in currentConcept.ancestors(include_self = False) :
            if ancestorConcept.name not in graphResultsForPhraseToken.columns :
                 continue
                        
            for token in tokens:
                constrainName = 'c_%s_%s_Ancestor_%s'%(token, currentConcept, ancestorConcept.name)
                m.addGenConstrIndicator(x[token, conceptName], True, x[token, ancestorConcept.name], GRB.EQUAL, 1)
       
    # -- Add constraints based on concept intersection statements in ontology - and(var1, var2, var3, ..)
    for conceptName in conceptNames :
        
        currentConcept = myOnto.search_one(iri = "*%s"%(conceptName))
            
        if currentConcept is None :
            continue
            
        for conceptConstruct in currentConcept.constructs(Prop = None) :
            if type(conceptConstruct) is And :
                
                for token in tokens:
                    
                    constrainName = 'c_%s_%s_Intersection'%(token, conceptName)
                    _varAnd = m.addVar(name="andVar_%s"%(constrainName))
    
                    andList = []
                
                    for currentClass in conceptConstruct.Classes :
                        andList.append(x[token, currentClass.name])

                    andList.append(x[token, conceptName])
                    
                    m.addGenConstrAnd(_varAnd, andList, constrainName)
    
    # -- Add constraints based on concept union statements in ontology -  or(var1, var2, var3, ..)
    for conceptName in conceptNames :
        
        currentConcept = myOnto.search_one(iri = "*%s"%(conceptName))
            
        if currentConcept is None :
            continue
            
        for conceptConstruct in currentConcept.constructs(Prop = None) :
            if type(conceptConstruct) is Or :
                
                for token in tokens:
                    
                    constrainName = 'c_%s_%s_Union'%(token, conceptName)
                    _varOr = m.addVar(name="orVar_%s"%(constrainName))
    
                    orList = []
                
                    for currentClass in conceptConstruct.Classes :
                        orList.append(x[token, currentClass.name])

                    orList.append(x[token, conceptName])
                    
                    m.addGenConstrOr(_varOr, orList, constrainName)
    
    # -- Add constraints based on concept objectComplementOf statements in ontology - var1 + var 2 = 1
    for conceptName in conceptNames :
        
        currentConcept = myOnto.search_one(iri = "*%s"%(conceptName))
            
        if currentConcept is None :
            continue
            
        for conceptConstruct in currentConcept.constructs(Prop = None) :
            if type(conceptConstruct) is Not :
                
                for token in tokens:                    
                    for currentClass in conceptConstruct.Classes :
                        constrainName = 'c_%s_%s_ComplementOf_%s'%(token, conceptName, currentClass)
                        
                        m.addConstr(x[token, conceptName] + x[token, currentClass], GRB.EQUAL, 1, name=constrainName)

    # -- Add constraints based on concept disjonitUnion statements in ontology - 
    
    # -- Add constraints based on concept oneOf statements in ontology - 

    m.update()
            
    # Add objectives
    X_Q = None
    for token in tokens :
        for conceptName in conceptNames :
            X_Q += graphResultsForPhraseToken[conceptName][token]*x[token, conceptName]
            X_Q += (1-graphResultsForPhraseToken[conceptName][token])*x[token, conceptName+'-neg']

    return X_Q
    
def addRelationsConstrains(m, myOnto, tokens, conceptNames, x, y, graphResultsForPhraseRelation):
    
    relationNames = graphResultsForPhraseRelation.keys()
        
    for relationName in relationNames:            
        for token in tokens: 
            for token1 in tokens:
                if token == token1:
                    continue

                y[relationName, token, token1]=m.addVar(vtype=GRB.BINARY,name="y_%s_%s_%s"%(relationName, token, token1))
                y[relationName+'-neg', token, token1]=m.addVar(vtype=GRB.BINARY,name="y_%s-neg_%s_%s"%(relationName, token, token1))
          
    # Add constraints forcing decision between variable and negative variables 
    for relationName in relationNames :
        for token in tokens: 
            for token1 in tokens:
                if token == token1:
                    continue
                
                constrainName = 'c_%s_%s_%sselfDisjoint'%(token, token1, relationName)
                m.addConstr(y[relationName, token, token1] + y[relationName+'-neg', token, token1], GRB.LESS_EQUAL, 1, name=constrainName)

    m.update()

    # -- Add constraints based on property domain and range statements in ontology
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
                        m.addConstr(currentConstrain, GRB.GREATER_EQUAL, 3 * y[currentRelation._name, token, token1], name=constrainName)

    # -- Add constraints based on property subProperty statements in ontology

    # -- Add constraints based on property allValueFrom statements in ontology

    # -- Add constraints based on property hasValueFrom statements in ontology

    # -- Add constraints based on property objectHasSelf statements in ontology

    # -- Add constraints based on property minCardinality statements in ontology
    
    # -- Add constraints based on property maxCardinality statements in ontology

    # -- Add constraints based on property exactCardinality statements in ontology
    
    # -- Add constraints based on property dataSomeValuesFrom statements in ontology
    
    # -- Add constraints based on property dataHasValue statements in ontology

    # -- Add constraints based on property dataAllValuesFrom statements in ontology

    # -- Add constraints based on property allValueFrom statements in ontology

    # -- Add constraints based on property equivalentProperty statements in ontology
    
    # -- Add constraints based on property disjointProperty statements in ontology

    # -- Add constraints based on property inverseProperty statements in ontology
    
    # -- Add constraints based on property functionalProperty statements in ontology
    
    # -- Add constraints based on property inverseProperty statements in ontology

    # -- Add constraints based on property inverseFunctiona;Property statements in ontology

    # -- Add constraints based on property reflexiveProperty statements in ontology

    # -- Add constraints based on property irreflexiveProperty statements in ontology

    # -- Add constraints based on property symetricProperty statements in ontology

    # -- Add constraints based on property asymetricProperty statements in ontology
    
    # -- Add constraints based on property transitiveProperty statements in ontology

    # -- Add constraints based on property symetricProperty statements in ontology

    # -- Add constraints based on property key statements in ontology

    m.update()

    # Add objectives
    Y_Q  = None
    for relationName in relationNames:
        for token in tokens:
            for token1 in tokens:
                if token == token1 :
                    continue

                Y_Q += graphResultsForPhraseRelation[relationName][token1][token]*y[relationName, token, token1]
                Y_Q += (1-graphResultsForPhraseRelation[relationName][token1][token])*y[relationName+'-neg', token, token1]
    
    return Y_Q
    
def calculateILPSelection(phrase, graph, graphResultsForPhraseToken, graphResultsForPhraseRelation, ontologyPathname = "./"):

    tokenResult = None
    relationsResult = None
    
    try:
        # Create a new Gurobi model
        m = Model("decideOnClassificationResult")
        m.params.outputflag = 0
        
        myOnto = loadOntology(graph.ontology, ontologyPathname)

        # Get list of tokens and concepts from panda dataframe graphResultsForPhraseToken
        tokens = graphResultsForPhraseToken.index.tolist()
        conceptNames = graphResultsForPhraseToken.columns.tolist()
        
        # Gurobi variables for concept - token
        x={}

        # Gurobi variables for relation - token, token
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
          
        # Collect results for concepts
        tokenResult = pd.DataFrame(0, index=tokens, columns=conceptNames)
        if x or True:
            if m.status == GRB.Status.OPTIMAL:
                solution = m.getAttr('x', x)
                
                for token in tokens :
                    for conceptName in conceptNames:
                        if solution[token, conceptName] == 1:
                            #print("The  %s is classified as %s" % (token, conceptName))
                            
                            tokenResult[conceptName][token] = 1

        # Collect results for relations
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
                                relationResult[token1][token] = 1
                                
                    relationsResult[relationName] = relationResult
                    
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
        
    tokenResult, relationsResult = calculateILPSelection(test_phrase, test_graph, test_graphResultsForPhraseToken, test_graphResultsForPhraseRelation, ontologyPathname="./examples/emr/")
    
    print("\nResults - ")
    print(tokenResult)
    
    if relationsResult != None :
        for name, result in relationsResult.items():
            print("\n")
            print(name)
            print(result)
    
if __name__ == '__main__' :
    main()