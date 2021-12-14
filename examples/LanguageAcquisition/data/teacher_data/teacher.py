'''
    
    Teacher network
    
    This model uses two transducers: 
        1) meaning transducer which converts an utterance into its logical form
        2) utterance transducer which converts the logical form into an utterance

    It also has another a comprehension module as a tool that analyzes the logical
    form of a learner's utterance with the observed situation and does logical
    entailment to determine the equivalency between the logical forms.
'''

# Import the pytorch Seq2Seq models
import network_models
import pickle
import data_loader
import random

#import statement
import torch
from torch import nn

import torch.nn.functional as F

from comprehension_module import Comprehension_Module
from transducer import Transducer as T


# I have not tested using GPU with the code yet.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Teacher():
    '''
        This class create a teacher model with two encoders and one decoder.
    '''
    
    def __init__(self, meaning_generator=None, comp_module=None, utterance_generator=None):
        
        
        if meaning_generator != None:
            self.meaning_generator = meaning_generator
        else:
            self.meaning_generator = T() #meaning transducer
            
        # Need to add the comprehension module
        if comp_module != None:
            self.cm = comp_module
        else:
            self.cm = Comprehension_Module()
        
        if utterance_generator != None:
            self.utterance_generator = utterance_generator
        else:
            self.utterance_generator = T() #meaning transducer
        
    
    def load_utterance_generator(self,u_generator):
        
        self.utterance_generator = u_generator
        
    def load_meaning_generator(self,m_generator):
        
        self.meaning_generator = m_generator
        
    def load_comp_module(self, comp_module):
        
        self.cm = comp_module
        
        
    def load_vocabulary(self, V):
        '''
            This method assigns the vocabulary list used by the learner model.
        '''
        
        self.vocabulary = V
    
    def load_predicates(self, P):
        '''
            This method assign the predicates used by the learner model.
        '''
        
        self.predicates = P
        
    def add_spaces(self, meaning):
        # Need to add an empty space as the first element
        meaning.insert(0,'')
        
        # Add the empty spaces between predicates with non-predicate words.
        if "le2(x1,x2)" in meaning:
            index = meaning.index("le2(x1,x2)")
            
            
            # insert one space after this element
            meaning.insert(index+1,'')
            meaning.insert(index+1,'')
            
            # insert two empty strings before this predicate
            meaning.insert(index,'')
            meaning.insert(index,'')
            
        elif "le2(x2,x1)" in meaning:
            index = meaning.index("le2(x2,x1)")
            
            # insert one space after this element
            meaning.insert(index+1,'')
            meaning.insert(index+1,'')
            
            # insert two empty strings before this predicate
            meaning.insert(index,'')
            meaning.insert(index,'')
            
        elif 'ab2(x1,x2)' in meaning:
            index = meaning.index("ab2(x1,x2)")
            
            # insert one space after this element
            meaning.insert(index+1,'')
            
        elif 'ab2(x2,x1)' in meaning:
            index = meaning.index("ab2(x2,x1)")
            
            # insert one space after this element
            meaning.insert(index+1,'')
        
    
    def evaluate(self, utterance_tensor, utterance, situation_tensor, situation, max_length=12, max_situation_length = 12):
        '''
            This function has the current model predict an utterance.
            
            Parameters
                model: Learner model
                utterance: Word sequence input for the model's utterance encoder
                situation: Predicate sequence for the model's situation encoder (observed sequences)
                logical: List of predicates
                vocabulary: List of words
                word_pred_map: Dictionary of word and predicate correspondence.
                max_length: Max number of elements the output utterance may have.
        '''
    
        
        # Pseudoteacher start by finding the meanings from the learner's utterance
        # with the meaning transducer.
        
        
        # Map of predicates and words
        print("Teacher Evaluation of learner's utterance.")
        print("Utterance:", utterance) 
        print("Situation:", situation)
        meaning, accepted = self.meaning_generator.evaluate(utterance.split())
        print('Meaning:',meaning)
        print('Accepted:', accepted)
        
        # How to determine if the meaning is valid? 
        # If the transducer returned an <err> predicate, then invalid
        valid = 0.0
        same = False
        
        # : Invalid Learner utterance!
        message = "Syntax Error"
        
        # Address syntax error: If the transducer did not reach the accepting states
        # then the utterance does not follow the grammar.
        if not accepted or "<err>" in meaning:
            
            # Need to load the situation into the comprehension module
            self.cm.add_assertion(situation)
            
            m_check = []
            for m in meaning:
                if "le2" in m or "ab2" in m or m == "<err>":
                    m_check.append(m)
                else:
                    m_check.append(m[:3])
                    
            s_check = []
            for s in situation:
                
                sit = s.replace("(t1","(x1").replace("(t2","(x2").replace("t2)","x2)").replace("t1)","x1)")
                
                if "le2" in s or "ab2" in s:
                    s_check.append(s)
                else:
                    s_check.append(s[:3])
            
            # m_check = [m[:3] for m in lst]
            # s_check = [s[:3] for s in situation]
            
            # If the meaning has a perfect match to the situation, then correct
            print("m_check", m_check)
            print("s_check", s_check)
            
            
            # If the meaning has the same predicates in the same order as the situation
            # Utterance has an error in form!
            if (m_check == s_check):
                
                # Remove the assertion
                self.cm.remove_assertion(situation)
                
                # "Error in form: Learner utterance describes the observed situation, but not in the expected grammer!"
                
                message = "Error in form"
            
                # For now, we can replace the situation with the meaning
                meaning = [s.replace("(t1","(x1").replace("(t2","(x2").replace("t2)","x2)").replace("t1)","x1)") for s in situation]
                self.add_spaces(meaning)
                
                utterance, accept = self.utterance_generator.evaluate(meaning)
            
                decoded_words = utterance
                
                return decoded_words, valid, message, same
                
            
            # In case that the predicates are not an exact match
            # Remove the assertion anyway
            self.cm.remove_assertion(situation)
            
            
            # Otherwise, the teacher detects a syntax error.
            message = "Syntax Error"
            
            # For now, we can replace the situation with the meaning
            meaning = [s.replace("(t1","(x1").replace("(t2","(x2").replace("t2)","x2)").replace("t1)","x1)") for s in situation]
            self.add_spaces(meaning)
            
            utterance,accept = self.utterance_generator.evaluate(meaning)
        
            decoded_words = utterance
            
            return decoded_words, valid, message, same
            
                
    
        # The other errors or correct
        if accepted:
            
            # Meaning could be extracted from the utterance
            
            valid = 1.0
            
            # Let us use the comprehension module
            
        
            # 1) Load the current situation into the module
            self.cm.add_assertion(situation)
            
            # Need to address the case where le2(t1,t2) != le2(t2,t1)
            # Need to address the case where le2(t1,t2) != le2(t2,t1)
        
            m_check = []
            for m in meaning:
                if "le2" in m or "ab2" in m:
                    m_check.append(m)
                else:
                    m_check.append(m[:3])
                    
            s_check = []
            for s in situation:
                
                sit = s.replace("(t1","(x1").replace("(t2","(x2").replace("t2)","x2)").replace("t1)","x1)")
                
                if "le2" in s or "ab2" in s:
                    s_check.append(sit)
                else:
                    s_check.append(sit[:3])
                    
        
            # m_check = [m[:3] for m in meaning]
            # s_check = [s[:3] for s in situation]
            # If the meaning has a perfect match to the situation, then correct
            
            print("m_check", m_check)
            print("s_check", s_check)
            
            
            # Use the entailment to see if the situation entails the situation
            entails_out = self.cm.entails(meaning)
            
            if entails_out:
                # Remove the assertion
                self.cm.remove_assertion(situation)
                
                # "Correct: Learner utterance describes the observed situation!"
                return [], valid, "Correct", same
            
            else:
                    
                # Check error in meaning, error in form, and uninterpretable.
                
                # 1) replace every variable x to X
                meaning2 = [m.replace('x1','X1').replace('x2','X2') for m in meaning]
                
                # 2) the meaning must denote the utterance
                # isdenoted, denoting_error, has_shapes, mapping = self.cm.denotes(meaning2)
                
                # 3) Determine the number of predicates from the meaning that are 
                # present in the situation
                predicates_found = self.cm.has_predicates(meaning2)
                
                # 4) Count the number of predicates
                number_of_predicates = len(meaning2) 
                
                # 5) Evaluate the meaning of the learner's utterance
                
               
                    
                m_found = []
                for m in meaning:
                    if "le2" in m or "ab2" in m:
                        m_found.append(m)
                    else:
                        m_found.append(m[:3])
                        
                p_found = []
                for p in predicates_found:
                    # Need to replace the t with x
                    pred = p.replace("(t1","(x1").replace("(t2","(x2").replace("t2)","x2)").replace("t1)","x1)")
                        
                    
                    if "le2" in p or "ab2" in p:
                        p_found.append(pred)
                    else:
                        p_found.append(pred[:3])
                
                    
                print("p_found", p_found)
                print("m_found", m_found)
                
                   
                # This case is when the utterance is uninterpretable
                if p_found == []:
                    message = "Uninterpretable"
                    
                    # For now, let us use the situation as feedback
                    meaning = [s.replace("(t1","(x1").replace("(t2","(x2").replace("t2)","x2)").replace("t1)","x1)") for s in situation]
            
                    self.add_spaces(meaning)
                    
                    print("Meaning for generation:", meaning)
                    
                    utterance, accept = self.utterance_generator.evaluate(meaning)
    
                    decoded_words = utterance
                    
                    # Remove the assertion
                    self.cm.remove_assertion(situation)
            
                    
                    return decoded_words, valid, message, same
    
                
                
                # Address error in form
                if p_found == m_found:
                    message = "Error in form"
                    
                    # Teacher needs to use the learner's meaning and
                    # generate the correct utterance.
                    
                    # For now, let us use the situation as feedback
                    meaning = [s.replace("(t1","(x1").replace("(t2","(x2").replace("t2)","x2)").replace("t1)","x1)") for s in situation]
            
                    self.add_spaces(meaning)
                    
                    print("Meaning for generation:", meaning)
                    
                    utterance, accept = self.utterance_generator.evaluate(meaning)
    
                    decoded_words = utterance
                    
                    # Remove the assertion
                    self.cm.remove_assertion(situation)
            
                    
                    return decoded_words, valid, message, same
    
                
                #if not(all(elem in p_found for elem in m_found)):
                if p_found != m_found:
                    message = "Error in meaning"
                    print(message)
                    print("Predicates found:",predicates_found)
                
                    
                    # The error in meaning finds the predicates that has the same
                    # predicates as the 
                
                    # build the new predicates from the predicates found
                    meaning = predicates_found
                    
                    # Find all predicates in the situation that has the same 
                    # constant as the one in the meaning
                    var = meaning[0][3:]
                    
                    print(var)
                    
                    # Find all predicates with the same predicate
                    N = [s.replace("(t2)","(t1)") for s in situation if var in s]
                    
                    print("Data from N:", N)
                    
                    # remove the left and above predicates
                    N = [n for n in N if ("le2" not in n and "ab2" not in n)]
                    
                    
                    print("Data from N:", N)
                    
                    if "x" in var or N == []:
                        meaning = situation
                    else:
                        meaning = N
                        
                        
                    #Need to replace the constants with variables.
                    meaning = [s.replace("(t1","(x1").replace("(t2","(x2").replace("t2)","x2)").replace("t1)","x1)") for s in meaning]
                    
                    # For now, let us use the situation as feedback
                    #meaning = [s.replace("(t1","(x1").replace("(t2","(x2").replace("t2)","x2)").replace("t1)","x1)") for s in situation]
            
                    
                    self.add_spaces(meaning)
                    
                    #print("Meaning for generation:", meaning)
                    
                    utterance, accept = self.utterance_generator.evaluate(meaning)
    
                    decoded_words = utterance
                    
                    # Remove the assertion
                    self.cm.remove_assertion(situation)
            
                    
                    return decoded_words, valid, message, same
        
        self.cm.remove_assertion(situation)
    
    def load_model(self,filename):
        
        model = pickle.load(open(filename,"rb")) 
        
        self.meaning_generator = model.meaning_generator
        self.utterance_generator = model.utterance_generator
        self.cm = model.cm
    
    
    def save_model(self,filename):
        
        pickle.dump(self,open(filename,"wb"))
        
    
    def __str__(self):
        
        out = "Learner Model Description\n"
        
        out += str(self.meaning_generator) +"\n"
        out += str(self.cm) + "\n"
        out += str(self.utterance_generator)
        
        return out
        
    
    
# Test the teacher model

# def test_save_load():
#     # Get the transducer
#     meaning_T = T()
#     meaning_T.load("./dataset/meaning_transducer.T")
    
#     utterance_T = T()
#     utterance_T.load("./dataset/utterance_transducer.T")
    
#     comp_module = Comprehension_Module()
    
#     teacher = Teacher(meaning_T, comp_module, utterance_T)
    
#     # Save the teacher model
#     teacher.save_model("./dataset/teacher_model.p")
    
#     # Load the teacher
#     teacher2 = Teacher()
#     teacher2.load_model("./dataset/teacher_model.p")
    
#     print(teacher)
#     print(teacher2)


# def test_model_train():
    
#     kb_filename = "logical_entailment.pl"
    
#     # Load the vocabulary
#     vocabulary = data_loader.build_list("./dataset/vocabulary.txt")
    
#     # Load the predicates
#     predicates = data_loader.build_list("./dataset/predicates.txt")
    
#     # Get the vocabulary and predicate sizes
#     predicates.insert(0,"<sos>")
#     predicates.append("<eos>")
#     predicates.remove('')
    
    
#     vocabulary.append("<eos>")
    
#     v_size = len(vocabulary)
    
#     p_size = len(predicates)
    
#     # Create the learner
    
#     # Initialize dimension sizes
#     situation_encoder_size = (p_size, 100)
#     situation_decoder_size = (100, v_size)
    
#     learner_model = learner.Learner(situation_encoder_size, situation_decoder_size)
   
#     meaning_T = T()
#     meaning_T.load("./dataset/meaning_transducer.T")
    
#     utterance_T = T()
#     utterance_T.load("./dataset/utterance_transducer.T")
    
#     comp_module = Comprehension_Module()
#     # Load the prolog commands into the teacher comprehension module
#     # Load the new logical_entailment file
#     comp_module.load_kb(kb_filename)
    
#     teacher_model = Teacher(meaning_T, comp_module, utterance_T)
     
   
#     # Initialize optimizers
#     learning_rate = 0.001 # consistent with literature.
    
    
#     # Add them to the learner
#     learner_model.load_vocabulary(vocabulary)
#     learner_model.load_predicates(predicates)
    
#     learner_model.load_optimizers(learning_rate)
    
#     # Next is to load the dataset used for the interaction procedure
#     interact_data_path = "dataset\\interaction_set.txt"
#     interact_fp = open(interact_data_path,"r")
#     interact_data = data_loader.load_data_logic(interact_fp)
    
#     # Next is to load the dataset used for the teacher test procedure
#     learner_test_path = "dataset\\test_set.txt"
#     learner_test_fp = open(learner_test_path,"r")
#     learner_test_data = data_loader.load_data_logic(learner_test_fp)
    
    
#     # Build the tensors for each set.
    
#     tensor_interact_data = data_loader.build_data_tensors_logic(interact_data, vocabulary, predicates)
    
#     # Build the tensor list for the test data
#     tensor_learner_test_data = data_loader.build_data_tensors_logic(learner_test_data, vocabulary, predicates)
    
    
#     # Test with one pair to see if the data is generated properly
#     r = 0
#     pair = (tensor_interact_data[r][0],tensor_interact_data[r][1])

#     #print(interact_data[r])
    
#     # get the ground truth sentence, utterance, and situation
#     truth_sentence = " ".join(interact_data[r][1])
#     utterance_tensor = pair[1]
#     situation_tensor = pair[0]
    
#     target_tensor = utterance_tensor
    
    
#     utterance = interact_data[r][1]
#     situation = interact_data[r][0]
    
#     times = 100
    
#     for i in range(times):
        
#         # 1) Learner generates utterance from situation alone
#         # learner utterance needs to be empty here!
#         learner_sentence = learner_model.generate_utterance(situation_tensor)[0]
#         learner_sentence = [w for w in learner_sentence if w != '']
#         learner_sentence = " ".join(learner_sentence)
        
    
#         # filter learner interactive sentence
#         fin_s = learner_sentence.split()
#         # if len(fin_s) >= 12:
#         #     fin_s = [w for w in fin_s[:11]]
        
#         # Need to build the tensor with the learner's output
#         utterance_tensor_learner = data_loader.create_input_tensor(fin_s, vocabulary)
        
        
#         # Teacher evaluate the utterance
#         learn_str = "\tLearner Sentence: "
        
        
#         # 2) Teacher receives learner's utterance for its own generation
#         teacher_sentence, teacher_valid, teacher_feedback, same_predicates = teacher_model.evaluate(utterance_tensor_learner, learner_sentence, situation_tensor, situation)
#         teacher_sentence = [w for w in teacher_sentence if w != '']
#         teacher_sentence  = " ".join(teacher_sentence)
    
        
#         # Print the teacher's utterance
#         teach_str = "\tTeacher Sentence: "
        
        
#         # Display the results of the current interaction
#         print("#"*80)
#         print("\tTraining Interaction #{}".format(i+1))
#         print(learn_str, learner_sentence)
#         print(teach_str, teacher_sentence)
#         print("\tValid: ", teacher_valid)
#         print("\tTeacher Feedback: ", teacher_feedback)
#         # print("\tSentence Accuracy: ", sen_accuracy)
#         # print("\tWord Accuracy: ", w_accuracy)
#         print("\tGround truth sentence:", truth_sentence)
        
#         # Train the learner model
#         learner_model.train(situation_tensor,target_tensor)
        
    
    
    
    
#     # After the program has completed running, remove all predicates from
#     # the knowledge base.
#     teacher_model.cm.reset_kb(kb_filename)

    
    
# test_model_train()

    