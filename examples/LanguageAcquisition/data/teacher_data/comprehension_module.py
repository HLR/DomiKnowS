
import os
import subprocess


from pyswip import *

class Comprehension_Module():
    
    def __init__(self,filename=None):
        
        self.prolog = Prolog()
        
        if filename != None:
            self.prolog.consult(filename)
            
        
    def reset_kb(self, filename):
        
        fp = open(filename)
        
        print("Removing all facts and rules from the prolog interpreter!")
        
        for line in fp:
            if "%" in line or line == '\n':
                continue
            
            line = line.strip().replace(".","")
            
            if ":-" in line:
                line = "({})".format(line)
            
            #print(line)
            #self.prolog.retractall(line)
            self.remove_assertion(line)
            
    def load_kb(self,filename):
        
        fp = open(filename)
        
        
        print("Adding all facts and rules into the prolog interpreter!")
        for line in fp:
            if "%" in line or line == '\n':
                continue
            
            line = line.strip().replace(".","") # remove the "."
            
            
            if ":-" in line:
                line = "({})".format(line)
            
            #print(line)
            
            self.add_assertion(line)
            #self.prolog.assertz(line)
            
        
    def write_consult_file(self,filename, P):
        '''
            L is the list of predicates to write in the file
        '''
        out = open(filename,"w")
        for p in P:
            print(p,file=out)
            
        out.close()
    
    def add_assertion(self,A):
        '''
            This method adds a single fact to the knowledge base.
        '''
        
        if isinstance(A,str):
            A = [A]
            
        for assertion in A:
            self.prolog.assertz(assertion)
        
    def remove_assertion(self,A):
        '''
            This method adds a single fact to the knowledge base.
        '''
        
        if isinstance(A,str):
            A = [A]
        
        for assertion in A: 
            self.prolog.retractall(assertion)
        
    def run_query(self,query):
        '''
            This function receives a query string to run using the prolog module.
        '''
        
        x = list(self.prolog.query(query))
        
        print("Run the query output:", x)
        
        if x == []:
            return(False)
        elif x == [{}]:
            return(True)
        else:
            return(x)
        
        
    def entails(self, M):
        '''
            This method evaluates the meaning of an utterance with the given situation.
            
            Returns:
            
            True if the situation entails the meaning
            
            False otherwise (Prolog returns false if a predicate is not defined in the situation)
        
        '''

        # Replace the variable in the meaning
        # Need to create a new list with the modified values
        meaning = [m.replace("x","X") for m in M]
        
        
        # Check if the meaning has the correct number of shapes before running the query
        shapes = ['sq1','st1','he1','ci1','el1','tr1']
        positions = ['ab2','le2']
        
        has_shape = False
        
        count = 0
        sps = []
        for m in meaning:
            if m[:3] in shapes:
                count += 1
                sps.append(m)
                has_shape = True
            
            if m[:3] in ('ab2','le2'):
                sps.append(m)
                
        
        # 0) No shape was found
        if count == 0:
            return False
        
        # 1) At least one shape is present
        # 2) If two shapes are present, then a position needs to be in between
        elif not (len(sps) == 1 and sps[0][:3] in shapes) and not (len(sps) == 3 and (sps[0][:3] in shapes and sps[1][:3] in positions and sps[2][:3] in shapes)): 
            return False
        
        
        # Create the query with the meaning.
        # Prolog considers a capitalized letter to be a variable.
        query = "entails('{}')".format(",".join(meaning))
       
        # Run the query
        output = self.run_query(query)
        
        
        return output
        
        
        
    def denotes(self, M):
        '''
            This function implements the query 'mapping' that determines if there is a unique
            mapping between the constants t1,t2,... from the situation and the 
            variables x1,x2,... from the provided meaning.
            
            p (Prolog): Prolog Interpreter
            meaning (list): The predicate sequence to map its variables.
            
            Returns:
                isdenoted
                    True -> If the established mapping is unique 
                            (1-to-1 mapping between variable and constant).
                    False -> Otherwise.
                
                denotingerror
                    True -> If the query returns a mapping in which the same
                            variable is mapped to two constants or two variables
                            map to the same constant.
                    False -> Otherwise.
                    
                hasshape
                    True -> If the meaning contains a shape or it contains two
                            shapes with a position in between.
                    False -> Otherwise.
                
        '''
        
        # Replace the variable in the meaning
        # Need to create a new list with the modified values
        meaning = [m.replace("x","X") for m in M]
        
        # Determine if meaning can be passed for mapping
        
        shapes = ['sq1','st1','he1','ci1','el1','tr1']
        positions = ['ab2','le2']
        
        has_shape = False
        
        count = 0
        sps = []
        for m in meaning:
            if m[:3] in shapes:
                count += 1
                sps.append(m)
                has_shape = True
            
            if m[:3] in ('ab2','le2'):
                sps.append(m)
                
        
        # 0) No shape was found
        if count == 0:
            return False, False, False, None
        
        # 1) At least one shape is present
        # 2) If two shapes are present, then a position needs to be in between
        elif not (len(sps) == 1 and sps[0][:3] in shapes) and not (len(sps) == 3 and (sps[0][:3] in shapes and sps[1][:3] in positions and sps[2][:3] in shapes)): 
            return False, False, has_shape, None
        
        
        
        # Create the query with the meaning.
        # Prolog considers a capitalized letter to be a variable.
        query = "mapping([{}])".format(",".join(meaning))
        
        # Run the denotes predicate rule
        #query = "denotes(({}))".format(",".join(meaning))
        
        # Run the query
        output = self.run_query(query)
        
        # Assume that the output is a list of dictionaries, need to
        # build a dictionary with the variables as keys and a list of
        # constants t as its value.
        
        print("Denotes query output", output)
        if output != False and output != True: 
            
            D = dict()
            for a in output:
                L = list(a.items())
                
                for T in L:
                    if T[0] not in D:
                        D[T[0]] = list()
                        
                    if T[1] not in D[T[0]]:
                        D[T[0]].append(T[1])        
                
            print("\t",D)
            
            # Analyze the map.
            
            
            # Number of variables in the dictionary
            n_var = len(D)
            
            # Single variable in the dictionary,
            # The list of constants it mapped to must have a length of 1
            if n_var == 1:
                n_c = len(D['X1'])
                if n_c != 1:
                    return False,True,has_shape,None
                
                return True,False,has_shape, D
            
            # Two variables in the dictionary
            # Each variable has one constant in their lists and the constants
            # must be different.
            elif n_var == 2:
                
                # One mapped constant to each element
                n_c1 = len(D['X1'])
                n_c2 = len(D['X2'])
                
                if n_c1 != 1 and n_c2 != 1:
                    return False,True,has_shape,None
                
                # Need to compare if the constant each paired up with is different
                v1 = D['X1'][0]
                v2 = D['X2'][0]
                
                if v1 == v2:
                    return False,True,has_shape,None
                
                return True,False,has_shape, D
            
            else:
                return False,False,has_shape, None
            
            
        return False,False,has_shape, None
    
    
    def has_predicates(self,M):
        '''
            This function implements the query 'find_predicates' which searches the
            number of predicates in the meaning that are also present in the situation.
            
            p (Prolog): Prolog Interpreter
            meaning (list): The predicate sequence to map its variables.
            
            Returns:
                
                List of predicates which are in both the situation and the meaning
        '''
        
        # Replace the variable in the meaning
        # Need to create a new list with the modified values
        meaning = [m.replace("x","X") for m in M]
        
        
        # Create the query with the meaning
        query = "find_predicates([{}],B)".format(",".join(meaning))
        
        # Run the query
        output = self.run_query(query)
        
        #print(output)
        # Clean the found predicate outputs
        
        print(output[0]['B'])
        line = [s.decode('utf-8') for s in output[0]['B']]
        
        #replace the brackets
        #line = line.replace("[","").replace("]","")
        
        
        # Replace the single quotes
        #line = line.replace("\'","")
        
        
        #Convert string to list
        #line = line.split(",")
        
        print(line)
        
        L = [x.replace("X","x") for x in line]
        L.reverse()
        
        
        return L
