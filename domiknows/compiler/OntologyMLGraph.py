import os
from string import Template
from pathlib import Path
from owlready2 import *

# path to Meta Graph ontology
graphMetaOntologyPathname = "./ontology/ML/"

class OntologyMLGraphCreator :
    'Class building graph based on ontology'
    
    # Templates for the elements of the Python graph
    graphImportTemlate =  Template('\tfrom domiknows import Graph, Concept\n\n')
    graphHeaderTemplate = Template('\twith Graph(\'${graphName}\') as ${graphVar}:\n')
    graphOntologyTemplate = Template('\t${graphVar}.ontology=\'${ontologyURL}\'\n\n')
    conceptTemplate = Template('\t$conceptName = Concept(name=\'${conceptName}\')\n')
    subclassTemplate = Template('\t${conceptName}.be(${superClassName})\n')
    relationTemplate = Template('\t${relationName}.be((${domanName}, ${rangeName}))\n')
    
    def __init__(self, name) :
        self.name = name
    
    def loadOntology(self, ontologyURL, ontologyPathname = "./"):
        
        # Check if Graph Meta ontology path is correct
        graphMetaOntologyPath = Path(os.path.normpath(graphMetaOntologyPathname))
        graphMetaOntologyPath = graphMetaOntologyPath.resolve()
        if not os.path.isdir(graphMetaOntologyPath):
            print("Path to load Graph ontology: %s does not exists in current directory %s"%(graphMetaOntologyPath,currentPath))
            exit()
        
        # Check if ontology path is correct
        ontologyPath = Path(os.path.normpath(ontologyPathname))
        ontologyPath = ontologyPath.resolve()
        if not os.path.isdir(ontologyPath):
            print("Path to load ontology: %s does not exists in current directory %s"%(ontologyPath,currentPath))
            exit()

        onto_path.append(graphMetaOntologyPath)
        onto_path.append(ontologyPath) # the folder with the ontology

        # Load ontology
        try :
            myOnto = get_ontology(ontologyURL)
            myOnto.load(only_local = True, fileobj = None, reload = False, reload_if_newer = False)
        except FileNotFoundError as e:
            print('Error when loading - ' + ontologyURL + " from: %s"%(ontologyPath))

        return myOnto

    def buildSubGraph(self, ontology, myOnto, graphRootClass, tabSize, graphFile) :
        print("\tBuilding subgraph for --- %s ----".expandtabs(2) %graphRootClass)
        
        # Collect all concept from this subgrapoh
        subGraphConcepts = []
        
        # Get graphName and graphVar for the subgraph from annotations of graphRootClass
        graphFile.write(self.graphHeaderTemplate.substitute(graphName=graphRootClass.graphName.first(), graphVar=graphRootClass.graphType.first()).expandtabs(tabSize))

        # Increase tab for generated code
        tabSize+=tabSize
        
        graphFile.write(self.graphOntologyTemplate.substitute(graphVar=graphRootClass.graphType.first(), ontologyURL=ontology).expandtabs(tabSize));

        # Add root concept to the graph
        graphFile.write(self.conceptTemplate.substitute(conceptName=graphRootClass._name).expandtabs(tabSize))
        for parent in graphRootClass.is_a : # immediate parent without self
            if parent != owl.Thing :
                graphFile.write(self.subclassTemplate.substitute(conceptName=graphRootClass._name, superClassName=parent._name).expandtabs(tabSize))

        subGraphConcepts.append(graphRootClass)
        
        # Add concepts in the subclass tree to the subgraph
        self.parseSubGraphOntology(graphRootClass, subGraphConcepts, tabSize, graphFile)
        
        # Add relations for every concepts found to this subgraph
        for subGraphConcept in subGraphConcepts:
            for ont_property in myOnto.object_properties() :
                domain = ont_property.get_domain().first() # Domain of the relation - assuming single domain   
                if ont_property.get_domain().first()._name == subGraphConcept._name : # if concept is a domain of this property
                    graphFile.write("\n")
                    graphFile.write(self.conceptTemplate.substitute(conceptName=ont_property._name).expandtabs(tabSize))

                    if ont_property.get_range().first() != None : # Check if property range is defined
                        graphFile.write(self.relationTemplate.substitute(relationName=ont_property._name, domanName = domain._name, rangeName=ont_property.get_range().first()._name).expandtabs(tabSize))

    def parseSubGraphOntology(self, ontConceptClass, subGraphConcepts, tabSize, graphFile):
        isLeaf = True

        ontConceptSubclasses = ontConceptClass.subclasses(only_loaded = False, world = None) # all the subclasses of the current concept
        for subClass in ontConceptSubclasses :
            print("\tCurrent subclasse".expandtabs(4), subClass) # immediate subclasses without self
            
            if subClass.graphType : # Check if this is a start of a new subgraph
                continue            # Skip it ands stop the parsing of this subtree
            
            isLeaf = False
            
            # Write concept and subclass relation to the subgraph
            graphFile.write(self.conceptTemplate.substitute(conceptName=subClass._name).expandtabs(tabSize))
            graphFile.write(self.subclassTemplate.substitute(conceptName=subClass._name, superClassName=ontConceptClass._name).expandtabs(tabSize))
            subGraphConcepts.append(subClass)
            
            # Recursively build subgraph for the current subclass
            self.parseSubGraphOntology(subClass, subGraphConcepts, tabSize , graphFile)
        
        if isLeaf :
            print("\tLeaf".expandtabs(6))
    
    def buildGraph(self, ontology, fileName="graph.py", ontologyPathname = "./") :
        
        myOnto = self.loadOntology(ontology, ontologyPathname)
        
        print("\n---- Building graph for ontology:", ontology)
        
        # Get root graph concepts
        rootGraphConcepts = set() # Set of found graph root concepts
        for cont_class in myOnto.classes():  
            if cont_class.graphType :          # Check if node annotated with graphType property
                rootGraphConcepts.add(cont_class)
         
        # Search imported ontology as well
        for currentOnt in myOnto.imported_ontologies :
            for cont_class in currentOnt.classes():  
                if cont_class.graphType :          
                    rootGraphConcepts.add(cont_class)
                    
        graphFile = open(fileName, "w")
        
        # Write Global Graph header
        graphFile.write(self.graphImportTemlate.substitute().expandtabs(0));
        graphFile.write(self.graphHeaderTemplate.substitute(graphName='global', graphVar='graph').expandtabs(0));
        graphFile.write(self.graphOntologyTemplate.substitute(graphVar='graph', ontologyURL=ontology).expandtabs(4));

        print("\nFound root graph concepts - \n")
        for rootConcept in rootGraphConcepts :
            # Build subgraph for each found graph root concept
            self.buildSubGraph(ontology, myOnto, rootConcept, 4, graphFile)
            graphFile.write("\n")

        graphFile.close()
        
        return graphFile.name
    
# --------- Testing

emrExamplePath = "examples/emr/"

def main() :
     #-- EMR
    emrOntologyMLGraphCreator = OntologyMLGraphCreator("EMR")
    emrGraphFileName = emrOntologyMLGraphCreator.buildGraph("http://ontology.ihmc.us/ML/EMR.owl", "EMRGraph.py",  ontologyPathname = emrExamplePath)
    
    emrGraphFile = open(emrGraphFileName, 'r')
    print("\nGraph build based on ontology - Python source code - %s\n\n" %emrGraphFileName, emrGraphFile.read())

if __name__ == '__main__' :
    main()