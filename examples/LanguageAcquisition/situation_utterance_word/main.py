import sys

# Add the DomiKnows main directory into the current system path
if '../../..' not in sys.path:
    sys.path.append('../../..')
print("sys.path - %s"%(sys.path))

if "../data/teacher_data" not in sys.path:
    sys.path.append("../data/teacher_data")
    

# Import Pytorch functionality
import torch
from torch import nn
import torch.nn.functional as F

# Import other modules
import time
import pickle
import teacher
import data_loader

from comprehension_module import Comprehension_Module
from transducer import Transducer as T

# Import DomiKnows modules and functions
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.program import SolverPOIProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from regr.program.loss import NBCrossEntropyLoss


from regr.sensor.pytorch.sensors import FunctionalSensor, JointSensor


import logging


# Graph declaration: Need to build a simple situation-utterance model
from graph import graph, utterance, word, word_label, utterance_contains_word

from model import LearnerModel, GenerateLabel 
from sensors import SituationRepSensor 
from sensors import create_words
from readers import InteractionReader

device = 'cpu'

# Functions to use.
def index2word(predicted_words, vocabulary):
        
        # Need to iterate through each element in the tensor for one dimension
        
        L = []
        
        for i in range(predicted_words.size(0)):
            
            w_prop = F.log_softmax(predicted_words[i], dim=1)
            
            # compute the argmax
            index = torch.argmax(w_prop)
            
            # Get the word from the index
            word = vocabulary[index]
            
            L.append(word)
            
        print("Conversion from index to string:", " ".join(L))
        return L # use edges to generate words with indices 
    
def separator_words(predicted, expected):
    '''
        This function needs to convert the output probabilities into a single index
        so that the local/argmax
    '''
    # Check the length of the predicted and expected
    p_len = len(predicted) # output
    e_len = len(expected) # Ground truth
    
    flag = False
    
    print("Prediction size (before):", len(predicted))
    print("Expected size (before):", len(expected))
    
    # Add padding to the ground truth if the prediction is shorter than ground truth
    if p_len<e_len:
        # print("cat problem",predicted.shape())
        print("Model output is smaller than expected!")
        predicted=torch.cat((predicted,torch.zeros((e_len - p_len,1))))
        flag = True
        
    elif p_len > e_len:
        print("Model output is larger than expected!")
        predicted=predicted[:e_len]
        flag = True
        
    # if flag:
    #     print("Prediction size (after):", len(predicted))
    #     print("Expected size (after):", len(expected))
    #     print("Expected labels:", expected)
        
    #     # Compute the softmax for each element in the sequence
    #     L = []
    #     for i in range(len(predicted)):
            
    #         print(predicted[i])
            
    #         w_prop = F.log_softmax(predicted[i], dim=0)
            
    #         # compute the argmax
    #         index = torch.argmax(w_prop)
            
    #         L.append(index.item())
            
    #     print("Predicted indices:", L)
            
        
    #     exit()
    
    return predicted

def separator_extra(predicted, expected):
    # Check the length of the predicted and expected
    p_len = len(predicted)
    e_len = len(expected)
    return predicted[e_len:]
    
def accuracy_similarity(model_sentence, truth_sentence, cat_map = None):
    '''
        This function evaluates if the model sentence is an exact match to the truth sentence.
        If cat_map != None, then it means the similarity function converts the words of each sentence to its category.
        
        
    '''
    if cat_map != None:
    
        #Replace the similar words from the model sentence to their category word 
        wordlist = model_sentence.split()
        for index in range(len(wordlist)):
            word = wordlist[index]
            
            # replace the word to the corresponding category
            wordlist[index] = cat_map[word]
         
        model_sentence = " ".join(wordlist)
        
        #Replace the similar words from the model sentence to their category word 
        wordlist = truth_sentence.split()
        for index in range(len(wordlist)):
            word = wordlist[index]
            
            # replace the word to the corresponding category
            wordlist[index] = cat_map[word]
         
        truth_sentence = " ".join(wordlist)
        
        
    # Accuracy (similarity)  
    try:
        if model_sentence == truth_sentence:
            return 1
        else:
            return 0
    except:
        return 0    

def word_accuracy(hypothesis, reference, word_hist=None, total_hist=None, cat_map=None):
    
    count = 0 # number of correct words in the hypothesis compared to reference
    
    # Split the sentences into a list
    reference = reference.split()
    hypothesis = hypothesis.split()
    
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    
    min_len = min(len(reference), len(hypothesis))
    max_len = max(len(reference), len(hypothesis))
    
    
    if cat_map == None:
        
        # Need to iterate through the entire reference sentence and evaluate
        # the total number of expected words.
        for i in range(ref_len):
        
            # Update the count of corrects if the current index is valid
            # for the reference and hypothesis
            if i < hyp_len:
                
                if reference[i] == hypothesis[i]:
                    count += 1
                    
                    if word_hist != None:
                        # Update the histogram of the correct categories
                        # Does this help determine if the model learned something?
                        if reference[i] not in word_hist:
                            word_hist[reference[i]] = 0
                        
                        word_hist[reference[i]] += 1
                       
            if total_hist != None:
                # The number of times the current category appeared in the data
                if reference[i] not in total_hist:
                        total_hist[reference[i]] = 0
                    
                total_hist[reference[i]] += 1
            
            
    
    else:
        # Need to iterate through the entire reference sentence and evaluate
        # the total number of expected words.
        for i in range(ref_len):
        
            # Update the count of corrects if the current index is valid
            # for the reference and hypothesis
            if i < hyp_len:
                
                if cat_map[reference[i]] == cat_map[hypothesis[i]]:
                    count += 1
                    
                    # Update the histogram of the correct categories
                    # Does this help determine if the model learned something?
                    if cat_map[reference[i]] not in word_hist:
                        word_hist[cat_map[reference[i]]] = 0
                    
                    word_hist[cat_map[reference[i]]] += 1
                    
            # The number of times the current category appeared in the data
            if cat_map[reference[i]] not in total_hist:
                    total_hist[cat_map[reference[i]]] = 0
                
            total_hist[cat_map[reference[i]]] += 1
                
            
    # Get the score divided on the number of correct elements
    try:
        score = count / len(reference)
    except:
        print("Error in reference")
        score = 0.0
        
    return count, len(reference)


def model_declaration():
    
    
    graph.detach()
    
    # Adding Sensors and Learner modules
    
    # Read the situation and utterance
    # The data is already split into a list of words/predicates
    # Ground truth: string lists
    utterance['tokenized_text_situation'] = ReaderSensor(keyword='situation')
    utterance['tokenized_text_utterance'] = ReaderSensor(keyword='utterance')
    
    # The "word" concept contains three properties:
    # 1) relation tensor: Tensor of ones that connect each predicate to each word
    # 2) 'sit': list of predicates
    # 3) 'utt': list of words
    word[utterance_contains_word, 'situation_token', 'utterance_token'] = JointSensor(utterance['tokenized_text_situation'], utterance['tokenized_text_utterance'],forward=create_words)
    
    # Build the embedding for the situation
    # Tensor of predicates indices
    utterance['situation_vectorized'] = SituationRepSensor('tokenized_text_situation')
    
   
    # Create the module learner with contains the hidden layer with the results of the encoding/decoding of the model
    
    # Prepare the parameters for the encoder/decoder
    
    # learning_rate = 0.001
    # max_length = 12
    vocabulary = [x.strip() for x in open("../data/vocabulary.txt")]
    predicates = [x.strip() for x in open("../data/predicates.txt")]
    
    embedding_dim = 36
    hidden_dim = 100
    pred_size = len(predicates)
    vocab_size = len(vocabulary)
    
    encoder_dim = (pred_size,100)
    decoder_dim = (100,vocab_size)
    
    
    # Compute the output of the Seq2Seq model
    word['word_probabilities'] = ModuleLearner(utterance['situation_vectorized'], module=LearnerModel(vocabulary, predicates, encoder_dim, decoder_dim))
    
    
    # Assign the label to the word concept
    word[word_label] = FunctionalSensor(utterance_contains_word, 'utterance_token', forward=GenerateLabel(device, vocabulary), label=True)
    
    
    # Convert the word probabilities to word indices.
   
    # How to evaluate sentences
    word[word_label]=FunctionalSensor(word["word_probabilities"],word['utterance_token'],forward=separator_words)
    
    
    # Run the module learner
    # word[word_label] = ModuleLearner(utterance['situation_vectorized'], module=LearnerModel(vocabulary, predicates, encoder_dim, decoder_dim))
    
    # Get the label without using the output of the learner
    # word[word_label]=FunctionalSensor(None, utterance['tokenized_text_utterance'][0], forward=GenerateLabel(device, vocabulary), label=True)
    
    
    # Convert the word probabilities to words.
    word['output_text'] = FunctionalSensor(word_label, vocabulary, forward=index2word)
    
    
    # Another functional sensor to explore is one that can create a label length independent.
    # Need to 
    
    program = SolverPOIProgram(graph,inferTypes=['local/argmax'],loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker(DatanodeCMMetric('local/argmax')))
    
    return program


def main():
    
    '''
        Current run of the model.
        
        Training Situation-Utterance pairs: 10,000
        Testing Situation-Utterance pairs: 1,000
        
    '''
    
    logging.basicConfig(level=logging.INFO)
    program = model_declaration()
    
    #graph.visualize("image")
    
    # Load the training file
    train_filename = "../data/training_set.txt"
    test_filename = "../data/test_set.txt"
    train_size = 1 #10000
    test_size = 5 #100
    
    
    # Get the list of predicates and words
    words = [x.strip() for x in open("../data/vocabulary.txt")]
    
    train_dataset = list(iter(InteractionReader(train_filename,"txt")))[:train_size]
    test_dataset = list(iter(InteractionReader(test_filename,"txt")))[:test_size]



    # Load pre-existing teacher model with the semantic evaluation
    teacher_pathname = "../data/teacher_data/teacher_model.p"
    kb_filename = "../data/teacher_data/logical_entailment.pl"
    
    # Load the teacher model
    # teacher_pathname = arguments[1]
    model_fp = open_file(teacher_pathname,"rb")
    
    
    # If the teacher model is not found, stop the program
    if model_fp == None:
    
        print("Invalid value for argument 2!")
        print("The model needs to exist!")
        
        #Uncomment when passed to the HPCC computer
        exit()
            
    else:        
        teacher_model = pickle.load(model_fp)
        
        
    # Load the prolog commands into the teacher comprehension module
    # Load the new logical_entailment file
    teacher_model.cm.load_kb(kb_filename)
    
    # Set the device to 'cpu' 
    device = 'cpu'
    learning_rate = 0.001
    
    
    sentence_results=[]
    sentence_labels=[]
    
    word_results=[]
    word_labels=[]
    
    
    # Write the data to file
    out = open("results.txt","w") 
    
    
    epochs = 1
    interval = 100
    

    word_accuracy_lst = []
    sentence_accuracy_lst = []
    
    # Measure the procedure time
    start = time.time()
    
    for e in range(1,epochs+1):
        print("Epoch #", e, file=out)
        
        count = 1    
        for i in range(0,train_size,interval):
        
            # program.train(train_dataset[i:i+interval], train_epoch_num=1, Optim=lambda param: torch.optim.SGD(param, lr = learning_rate) , device=device)
            # program.test(test_dataset, device=device)
        
            print("\nTraining Interval #", (i+100), file=out)    
       
            # Add list of outputs and labels used during the evaluation period
            word_iter_results = []
            word_iter_labels = []
            
            sen_iter_results = []
            sen_iter_labels = []
            
            
            
            for example, sentence in zip(list(iter(test_dataset)), program.populate(test_dataset, device=device)):
                
                # Interaction name
                print("Element #",count, file=out)
                count+=1
                
                print("\nSentence:", sentence, file=out)
                print("Sentence Child Node(s):", sentence.getChildDataNodes(), file=out)
                
                temp_results = []
                temp_labels = []
                
                for w_label, w in zip(example['utterance'][0],sentence.getChildDataNodes()):
                    
                    w_token = w.getAttribute("utterance_token")
                    w_prob = w.getAttribute(word_label)
                    #w_label = w.getAttribute("word_label","label")
                    w_argmax_prob = w.getAttribute(word_label,"local/argmax")
                    
                    try:
                        w_argmax = int(torch.argmax(w.getAttribute(word_label,"local/argmax"))) #int(torch.argmax(w.getAttribute(word_label)))
                    except:
                        w_argmax = None
                    
                    
                    # Only add words to the respective sets if the label is valid
                    if w_label != None:
                        
                        
                        #w_label = w_label.item()
                        w_label = words.index(w_label)
                        
                        word_text = words[w_argmax]
                        word_text_lbl = words[w_label]
                        
                        # Print the word node
                        print("\nWord:", w, file=out)
                        
                        # Print the utterance token
                        print("Word utterance token:", w_token, file=out)
                        
                        # Print the word node's word probabilities
                        # print("Word (utterance['word_probabilities']):", w_prob, file=out)
                        
                        # Print the word's node word label (index)
                        print("Word (word_label's label property):", w_label, file=out)
                        
                        # Print the word's node argmax value
                        # print("Word attributes (local/argmax):", w_argmax_prob, file=out)
                        print("Argmax:", w_argmax, file=out)
                        
                        # Print the word text
                        print("Word output token:", word_text, file=out)
                        
                        # Append the model's argmax outputs in a temporary node to represent the utterance sequence
                        # Used to compute the sentence accuracy
                        # word_results.append(word_text)
                        temp_results.append(word_text) 
                        
                        # Append the word's node output to compute the word accuracy
                        # word_labels.append(w.getAttribute("word_label","label").item())
                        # temp_labels.append(w.getAttribute("word_label","label").item())
                        # word_labels.append(word_text_lbl)
                        temp_labels.append(word_text_lbl)
                
                # Add the current index sequence into the sentence level candidates
                # to compute the sentence level accuracy
                sentence_results.append(temp_results)
                sentence_labels.append(temp_labels)
                
                sen_iter_results.append(temp_results)
                sen_iter_labels.append(temp_labels)
                
                
                # need to clean the sentence
                learner_sentence = " ".join(clean_sentence(temp_results[:]))
                situation = example['situation'][0]
                
                print("Learner_sentence:",learner_sentence)
                print("Situation:", situation)
                
                
                # Test the learner's utterance with the teacher
                # teacher_sentence, teacher_valid, teacher_feedback, same_predicates = teacher_model.evaluate(None, learner_sentence, None, situation)
            
                # print("Teacher Sentence:", teacher_sentence)
                # print("Teacher feedback:", teacher_feedback)
            
            # Update: Use evaluation metrics like the previous experiment
            
            sen_acc = 0.0
            w_acc = 0.0
            
            w_acc_num = 0
            w_acc_den = 0
            
            for sentence,label in zip(sen_iter_results, sen_iter_labels):
            
                # print(" ".join(sentence), " ".join(label))
                sen_acc += accuracy_similarity(" ".join(sentence), " ".join(label))
            
                # Need to count the number of correct utterances in the evaluation
                result = word_accuracy(" ".join(sentence), " ".join(label))
                w_acc_num += result[0]
                w_acc_den += result[1]
            
            try:
                w_acc = w_acc_num / w_acc_den # Number of words in the current reference
            except:
                w_acc = 0.0
                
            try:
                sen_acc /= len(sentence_results)
            except:
                sen_acc = 0.0
                
            word_accuracy_lst.append(w_acc)
            sentence_accuracy_lst.append(sen_acc)
            
            # Need to use the list of the 100 test examples
            # Print the new results
            # print("\nWord results:", word_results, file=out)
            # print("Word Labels:", word_labels, file=out)
            
            
            
            print("\nSentence results:", sentence_results, file=out)
            print("Sentence Labels:", sentence_labels, file=out)
            print("Final model word accuracy is :", w_acc, file=out)
            print("Final model sentence accuracy is :", sen_acc, file=out)
            
            
            # Print the results
            # print("\nWord results:", word_results, file=out)
            # print("Word Labels:", word_labels, file=out)
            # print("Final model word accuracy is :",sum([i==j for i,j in zip(word_results,word_labels)])/len(word_results), file=out)
            
            
            # print("\nSentence results:", sentence_results, file=out)
            # print("Sentence Labels:", sentence_labels, file=out)
            # print("Final model sentence accuracy is :",sum([i==j for i,j in zip(sentence_results,sentence_labels)])/len(sentence_results), file=out)
        
    # Finish of run
    end = time.time()
    
    total_time = end - start # This is in seconds
    
    # convert to hour, minutes, and seconds
    elapsed_time = convert_time(total_time)
    
    
    # Print the final result
    print("Word Accuracy per interval:", word_accuracy_lst)
    print("Sentence Accuracy per interval:", sentence_accuracy_lst)
        
    print("Procedure elapsed time:", elapsed_time)
    
    out.close() # close the output file
    
    # After the program has completed running, remove all predicates from
    # the knowledge base.
    teacher_model.cm.reset_kb(kb_filename)
   
    
def convert_time(seconds):
    
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int((seconds % 3600) % 60)
    
    return "{:02d}:{:02d}:{:02d} hh:mm:ss".format(h,m,s)


def clean_sentence(sentence):
    
    # add the first 'the'
    sentence.insert(0,"the")
    
    # Find any 'above' or 'below'
    if "above" in sentence:
    
        ab_index = sentence.index("above")
        
        # add 'the' after it
        sentence.insert(ab_index+1,"the")
        
    if "below" in sentence:
    
        ab_index = sentence.index("below")
        
        # add 'the' after it
        sentence.insert(ab_index+1,"the")
        
    
    # Find left or right
    if "left" in sentence:
    
        ab_index = sentence.index("left")
        
        
        # add 'of' and 'the' after the word
        
        sentence.insert(ab_index+1,"the")
        sentence.insert(ab_index+1,"of")
        
        # add 'the' after it
        sentence.insert(ab_index,"the")
        sentence.insert(ab_index,"to")
        
        
    if "right" in sentence:
    
        ab_index = sentence.index("right")
        
        # add 'of' and 'the' after the word
        sentence.insert(ab_index+1,"the")
        sentence.insert(ab_index+1,"of")
        
        # add 'the' after it
        sentence.insert(ab_index,"the")
        sentence.insert(ab_index,"to")
        
    return sentence
    

def open_file(filename,option):
    '''
        This function tries to open a file and returns the file pointer object.
        Returns None if the file is not opened successfully.
    '''
    
    fp = None
    
    try:
        fp = open(filename,option)
    except:
        print("The file:", filename,"was not found!")
        
        
    return fp



    
if __name__ == "__main__":
    main()
