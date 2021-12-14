'''

'''

import pickle

EPSILON = ''

class Transducer():
    
    '''
        M = (Sigma, O, S, q0, F)
        
        Sigma = Alphabet of input symbols
        S = Set of States
        O = Alphabet of output symbols
        q0 = Initial state
        F = Set of final states
        D = transition function
    '''
    
    def __init__(self, Sigma=set(), S=set(), O=set(), q0=None, F=set(), D=dict()):
        '''
        
        '''
        
        # Initialize the Components of the Finite State Transducer
        self.S = S
        self.Sigma = Sigma
        self.O = O
        self.q0 = q0
        self.F = F
        self.D = D # Dictionary of transitions

    def evaluate(self, sequence):
        '''
            Traverse the transducer with the provided input until reached the
            end or an error occured
        '''
        
        try:
            output = []
            q = self.q0
            accepted = False # Assume the sequence is not valid.
            
            
            for v in sequence: 
                #print(q,v)
                if (q,v) in self.D:
                    if self.D[(q,v)][1] != '':
                        output.append(self.D[(q,v)][1])
                    q = self.D[(q,v)][0]
                else:
                    output.append("<err>")
                    
            # If the utterance reaches an accepting state, the utterance is valid.
            if q in self.F:
                print(q)
                accepted = True
            
            return output, accepted
        
        except KeyError:
            print("error with the key! {} {}".format(q,v))
            return None, False
    
    
    def __str__(self):
        
        s = ("Sigma: {}\n".format(self.Sigma))
        s += ("S: {}\n".format(self.S))
        s += ("O: {}\n".format(self.O))
        s += ("q0: {}\n".format(self.q0))
        s += ("F: {}\n".format(self.F))
        s += ("D:\n")
        
        for key,value in self.D.items():
            s += ("\t({},{}) -> ({},{})\n".format(key[0],key[1],value[0],value[1]))
        
        return s
    
    def __copy(self,other):
        
        self.Sigma = other.Sigma
        self.S = other.S
        self.O = other.O
        self.q0 = other.q0
        self.F = other.F
        self.D = other.D
    
    def save(self,filename):
        '''
        '''
        
        fp_out = open(filename,'wb')
        pickle.dump(self,fp_out)
    
    
    def load(self,filename):
        
        fp_in = open(filename,'rb')
        A = pickle.load(fp_in)
        self.__copy(A)
        
        
    def set_sigma(self, Sigma):
        
        self.Sigma = Sigma
    
    def set_S(self,S):
        
        self.S = S
    
    def set_O(self,O):
        
        self.O = O
    
    def set_F(self,F):
        
        self.F = F
    
    def set_D(self,D):
        self.D = D
    
    def set_q0(self,q0):
        
        self.q0 = q0
    
## Test the class 
#S = {1,2,3}
#Sigma = {'a','b'}
#O = {'1','2'}
#q0 = 1
#F = {3}
#D = {('a',1):('1',2), ('b',2):('2',3)}
#    
#T = Transducer(Sigma,S,O,q0,F,D)
#
#print(T)
#
## Evaluate one input sequence
#seq = ['a','b']
#
#output = T.evaluate(seq)
#
#print(output)
