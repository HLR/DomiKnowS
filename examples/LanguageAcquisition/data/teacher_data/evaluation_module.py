'''
    This module contains all of the evaluation functions used by the model(s)
    across all phases of the task
    
    It will be easier for code maintenance.
'''

import sys

import pickle
import time
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

from transducer import Transducer

import data_loader


# Initialize global transducer
# A = Transducer()
# A.load("../dataset/meaning_transducer.Transducer")
#print(A)

import data_loader

# Suppress all warnings from python
# I was getting a warning when the forward pass was calling a weights function 
# but the current version uses flat_weights instead.
import warnings
warnings.filterwarnings('ignore')

# This is used to make sure that the program can use GPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_evaluate(teacher,learner, test_data, tensor_test_data, n_test = None, corrective_flag = False):
    '''
        This function runs the evaluation procedure for multiple sequences.
        
        Parameters    
            model: Teacher or learner model
            test_data: 
            logical: List of predicates.
            vocab: List of words.
            n_test : Number of elements to evaluate. The default is None.
    '''
    
    with torch.no_grad():
        # If no length is specified, run the entire data that is provided.
        if n_test == None:    
            n_test = len(test_data[1])
        
        reference_list = []
        candidate_list = []
    
        correct = [0 for indx in range(14)]
        
        # Number of correct instances of the current category
        acc_hist = {}
        acc_hist_u = {}
        
        # Total instances of the current category
        total_hist = {}
        total_hist_u = {}
    
        s_thres_count = 0
        
        # This count the number of valid and invalid utterances
        v_count = 0
        i_count = 0
        
        # Determine the other error types
        no_shape_count = 0
        not_relevant_count = 0
        uninterpretable_count = 0
        correct_count = 0
        eim_count = 0
        eif_count = 0
        de_count = 0
        
        
        avoid_train_count = 0
        same_predicates_count = 0
        
        
        for i in range(1,n_test+1):
            
            # Extract the input situation and utterance, target situation, and target utterance.
            r = i-1
            pair = (tensor_test_data[r][0],tensor_test_data[r][1])
            
            #Create the ground truth sentence
            #truth_situation = " ".join(test_data[r][2])
            truth_utterance = " ".join(test_data[r][1])
            
            
            situation_tensor = pair[0]
            
            situation = test_data[r][0]
                
            
            # evaluate using situation to utterance path of the model
            learner_utterance = learner.generate_utterance(situation_tensor)[0]
            learner_utterance = [w for w in learner_utterance if w != '<sos>']
            learner_utterance  = " ".join(learner_utterance)
            
            fin_s = learner_utterance.split()
            # if len(fin_s) >= 12:
            #     fin_s = [w for w in fin_s[:11]]
            
            # utterance_tensor_learner = data_loader.create_input_tensor(fin_s, learner.vocabulary)    
            teacher_sentence, teacher_valid, teacher_feedback, same_predicates = teacher.evaluate(learner, learner_utterance, situation_tensor, situation)   
                
            # Count the number of valid utterances
            if teacher_valid == 1:
                v_count += 1
            else:
                i_count += 1
                
                
            # The learner utterance was correct
            if same_predicates == True:
                same_predicates_count += 1
                
            
            # Count the type of teacher feedback
            if teacher_feedback.lower() == "syntax error":
                uninterpretable_count += 1
        
            elif teacher_feedback.lower() == "no shape":
                no_shape_count += 1
                
            elif teacher_feedback.lower() == "uninterpretable":
                not_relevant_count += 1
            
            elif teacher_feedback.lower() == "correct":
                correct_count += 1
            elif teacher_feedback.lower() == "error in meaning":
                eim_count += 1
            elif teacher_feedback.lower() == "error in form":
                eif_count += 1
            
            elif teacher_feedback.lower() == "denoting error":
                de_count += 1
            
            # corrective_flag == True means that learner is trained with teacher's utterance
            if corrective_flag:
                if teacher_feedback.lower() != "correct":
                    teacher_sentence = [w for w in teacher_sentence if w != '']
                    teacher_sentence  = " ".join(teacher_sentence)
                else:
                    teacher_sentence = learner_utterance
                
            else:
                teacher_sentence = truth_utterance
            
            
            #Display the situation and utterances involved in this evaluation
            # Display the utterances in question
            
            
            # Compute individual accuracies first
            # Need to add the sentence and word accuracy
            sen_accuracy = accuracy_similarity(learner_utterance, teacher_sentence)
            
            # Need to count the number of correct utterances in the evaluation
            result = word_accuracy(learner_utterance, teacher_sentence)
                
            w_accuracy = result[0] / result[1] # Number of words in the current reference
    
            #teacher_sentence = [w for w in teacher_sentence if w != '']
            #teacher_sentence  = " ".join(teacher_sentence)
            
            print("#"*80)
            print("\tEvaluation Interaction #{}".format(r+1))
            
            print("\tLearner Sentence:", learner_utterance)
            
            print("\tTeacher Sentence:", teacher_sentence)
            
            print("\tValid: ", teacher_valid)
            print("\tTeacher Feedback: ", teacher_feedback)
            
            print("\tSentence Accuracy: ", sen_accuracy)
            print("\tWord Accuracy: ", w_accuracy)
            
            print("\tGround truth Utterance:", truth_utterance)
    
    
            # Utterance accuracy: Evaluate the model output utterance with the ground truth
            # without category maps.
            
            # Need to compare category similarity between the words
            # for a fairer comparison.
            correct[0] += accuracy_similarity(learner_utterance, teacher_sentence)
        
            
            # Need to count the number of correct utterances in the evaluation
            result = word_accuracy(learner_utterance, teacher_sentence) #, acc_hist_u, total_hist_u)
                
            correct[1] += result[0] # Number of correct words per utterance 
            correct[2] += result[1] # Number of words in the current reference
            
    
        # Prepare the scores
        correct[0] /= n_test # Sentence Accuracy in the set
        # correct[1] is the number of correct words in the set (overall)
        # correct[2] is the number of words in the set (overall)
        correct[3] = correct[1] / correct[2] # Word Accuracy in the set
        
        
        correct[4] = v_count
        correct[5] = i_count
        correct[6] = same_predicates_count
        correct[7] = no_shape_count
        correct[8] = not_relevant_count
        correct[9] = uninterpretable_count
        correct[10] = correct_count
        correct[11] = eim_count
        correct[12] = eif_count
        correct[13] = de_count
        
        return correct


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
    if model_sentence == truth_sentence:
        return 1
    else:
        return 0

def word_error_rate(hypothesis, reference):
    '''
        Simplified version of the formula. It only compares if the words are different and if the length between sentences 
        are different.
    '''
    score = 0
    
    # Split the sentences into a list
    reference = reference.split()
    hypothesis = hypothesis.split()

    # Find the smallest length and the max length between both sentences
    min_len = min(len(reference), len(hypothesis))
    max_len = max(len(reference), len(hypothesis))

    #Iterate through the smallest length
    #If the elements are different to each other, increase error by 1 (substitutions)
    for i in range(min_len):
        if reference[i] != hypothesis[i]:
            score += 1
    
    # Add ones to the number of remaining elements (Insertions or deletions required)
    score += max_len - min_len

    # Divide by the reference sequence.
    score /= len(reference)
    
    return score

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
    score = count / len(reference)
    
    return count, len(reference)

