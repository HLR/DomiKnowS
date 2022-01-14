'''

This script runs the baseline of the model. The baseline is the learner model without using the entailment function as its loss value.
It does not use the prolog interpreter in the results.

'''

import sys
import math


# Add the system path with the local directory
# sys.path.append('/mnt/home/castrog4/.local/bin')


# Add the DomiKnows main directory into the current system path
if '../../..' not in sys.path:
    sys.path.append('../../..')
print("sys.path - %s" % (sys.path))

if "../data/teacher_data" not in sys.path:
    sys.path.append("../data/teacher_data")

# Import Pytorch functionality
import torch
from torch import nn
import torch.nn.functional as F

# Import other modules
import time
# import pickle
# import teacher
# import data_loader
#
# from comprehension_module import Comprehension_Module
# from transducer import Transducer as T

# Import DomiKnows modules and functions
from regr.sensor.pytorch.learners import ModuleLearner
from regr.sensor.pytorch.sensors import ReaderSensor
from regr.program import SolverPOIProgram
from regr.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from regr.program.loss import NBCrossEntropyLoss

from regr.sensor.pytorch.sensors import FunctionalSensor, JointSensor
from regr.sensor.pytorch.relation_sensors import CompositionCandidateSensor

import logging

# Graph declaration: Need to build a simple situation-utterance model
from graph import graph, situation, word, word_label, utterance_contains_words, utterance, task_sit, task_utt, task

from model import LearnerModel, GenerateLabel
from sensors import SituationRepSensor
# from sensors import create_words
from readers import InteractionReader

device = 'cpu'

# Functions to use.

# from program import Prolog_Module


def pad_utterance(utterance, length=12):
    new_utterance = utterance  # Copy the input of the utterance

    if len(new_utterance) < length:
        diff = length - len(new_utterance)

        if diff > 0:
            for k in range(diff):
                new_utterance.append("<eos>")

    return new_utterance  # return the padded text out of the extra dimension


def model_declaration():
    graph.detach()

    max_length = 12

    # Adding Sensors and Learner modules

    # Read the situation and utterance
    # text_situation is a list of one string
    # text_utterance is a list of one string
    situation['text_situation'] = ReaderSensor(keyword='situation')
    utterance['text_utterance'] = ReaderSensor(keyword='utterance')

    def create_has_a(a, b, arg1, arg2):
        return True

    task[task_sit.reversed, task_utt.reversed] = CompositionCandidateSensor(situation['text_situation'],
                                                                            utterance['text_utterance'], relations=(
        task_sit.reversed, task_utt.reversed), forward=create_has_a)

    # Need to vectorize the situation
    situation['situation_vectorized'] = SituationRepSensor('text_situation')  # 1 x predicate_length

    def create_words(words):
        pad_lst = pad_utterance(words[0].split(), 12)
        return torch.ones((len(pad_lst), 1)), pad_lst

        # [situation for i in words[0].split()]

    # Combine the elements into the word concept
    word[utterance_contains_words, 'words'] = JointSensor(utterance['text_utterance'], forward=create_words)

    # learning_rate = 0.001
    # max_length = 12
    vocabulary = [x.strip() for x in open("../data/vocabulary.txt")]
    predicates = [x.strip() for x in open("../data/predicates.txt")]

    embedding_dim = 36
    hidden_dim = 100
    pred_size = len(predicates)
    vocab_size = len(vocabulary)

    encoder_dim = (pred_size, 100)
    decoder_dim = (100, vocab_size)

    # Assign the label to the word concept
    word[word_label] = FunctionalSensor('words', forward=GenerateLabel(device, vocabulary), label=True)

    # Compute the output of the Seq2Seq model
    word[word_label] = ModuleLearner(situation['situation_vectorized'],
                                     module=LearnerModel(vocabulary, predicates, encoder_dim, decoder_dim))

    program = SolverPOIProgram(graph, poi=(task, utterance, situation, word), inferTypes=['local/argmax'],
                               loss=MacroAverageTracker(NBCrossEntropyLossCustom()),
                               metric=PRF1Tracker(DatanodeCMMetric('local/argmax')))

    return program


class NBCrossEntropyLossCustom(torch.nn.CrossEntropyLoss):
    def forward(self, input, target, *args, **kwargs):
        print("Input:", input.size())
        print("Target:", target.size())

        input = input.view(-1, input.shape[-1])
        target = target.view(-1).to(dtype=torch.long, device=input.device)
        return super().forward(input, target, *args, **kwargs)


def main():
    '''
        Current run of the model.

        Training Situation-Utterance pairs: 10,000
        Testing Situation-Utterance pairs: 1,000

    '''

    logging.basicConfig(level=logging.INFO)

    program = model_declaration()

    # graph.visualize("image")

    # Load the training file
    # train_filename = "../data/training_set_single.txt"
    # test_filename = "../data/test_set_single.txt"
    # train_size = 112
    # test_size = 10 #56

    train_filename = "../data/training_set.txt"
    test_filename = "../data/test_set.txt"
    train_size = 10000
    test_size = 1000

    interval = 100 # Of times the model trains before being evaluated

    # Get the list of predicates and words
    words = [x.strip() for x in open("../data/vocabulary.txt")]

    # train_dataset = list(iter(InteractionReader(train_filename,"txt")))[:train_size]
    # test_dataset = list(iter(InteractionReader(test_filename,"txt")))[:test_size]

    train_dataset = list(iter(InteractionReader(train_filename, "txt")))[:train_size]
    test_dataset = list(iter(InteractionReader(test_filename, "txt")))[:test_size]

    # Set the device to 'cpu'
    device = 'cpu'
    learning_rate = 0.001  # 10^-3 seems to be a great option for this setting
    train_epoch = 3
    epochs = train_epoch

    # Training with Adam optimizer generates better performance for this task than SGD.
    # program.train(train_dataset, train_epoch_num=train_epoch,
    #               Optim=lambda param: torch.optim.Adam(param, lr=learning_rate), device=device)


    sentence_results = []
    sentence_labels = []

    word_results = []
    word_labels = []

    sen_acc_inter = []
    word_acc_inter = []

    # Write the data to file
    # out = open("results.txt", "w")

    # Measure the procedure time
    start = time.time()

    # Add list of outputs and labels used during the evaluation period
    word_iter_results = []
    word_iter_labels = []

    sen_iter_results = []
    sen_iter_labels = []

    sen_acc_epoch = []
    word_acc_epoch = []

    count = 0

    # Training is done by parts, we evaluate the model after 100 interactions with the data
    for e in range(1, epochs + 1):
        # print("Epoch #", e, file=out)
        print("Epoch #", e)

        temp_w_acc = 0.0
        temp_sen_acc = 0.0

        inter_count = 0
        for i in range(0, train_size, interval):

            # Train the model for the number of interval
            program.train(train_dataset[i:i+interval], train_epoch_num=1,
                          Optim=lambda param: torch.optim.Adam(param, lr=learning_rate), device=device)

            inter_count += 1

            # print("\nTraining Interval #", (i + 100), file=out)
            print("\nTraining Interval #", (i + interval))

            # Add list of outputs and labels used during the evaluation period
            word_iter_results = []
            word_iter_labels = []

            sen_iter_results = []
            sen_iter_labels = []

            count = 1
            for example, sentence in zip(list(iter(test_dataset)), program.populate(test_dataset, device=device)):

                    # Interaction name
                    print("Element #", count)
                    count += 1

                    print("\nSentence:", sentence)
                    print("Sentence Child Node(s):", sentence.getChildDataNodes())

                    # Use find Data Nodes
                    print("Found word data nodes:", sentence.findDatanodes(select="word"))

                    print("Found situation data nodes:", sentence.findDatanodes(select="situation"))

                    # Print the properties
                    # print("Sentence 'text_situation' :", sentence.findDatanodes(select="situation")[0].getAttributes('text_situation'))
                    print("Sentence 'text_utterance' :", sentence.getAttribute('text_utterance'))

                    temp_results = []
                    temp_labels = []

                    for w_value, w in zip(example['utterance'][0], sentence.findDatanodes(select="word")):

                        w_token = w.getAttribute('words')
                        w_prob = w.getAttribute(word_label)
                        w_label = w.getAttribute("word_label", "label")
                        w_argmax_prob = w.getAttribute(word_label, "local/argmax")

                        # Move to the next label
                        if w_label.item() == -100:
                            continue

                        # print("Word token:", w_token)
                        # print("Word_probabilities:", w_prob)
                        # print("Word Label (index):", w_label)
                        # print("Word 'local/argmax' output:", w_argmax_prob)

                        try:
                            w_argmax = int(torch.argmax(
                                w.getAttribute(word_label, "local/argmax")))  # int(torch.argmax(w.getAttribute(word_label)))
                        except:
                            print("Error computing Argmax!")
                            w_argmax = None

                        # print("Word 'local/argmax' index:", w_argmax)

                        # Only add words to the respective sets if the label is valid
                        if w_label != None and w_label != "<eos>":
                            w_label = words[w_label.item()]

                            # w_label = w_label.item()
                            w_label = words.index(w_label)

                            word_text = words[w_argmax]
                            word_text_lbl = words[w_label]

                            # Print the word node
                            print("\nWord:", w)

                            # Print the utterance token
                            print("Word utterance token:", w_token)

                            # Print the word node's word probabilities
                            # print("Word (utterance['word_probabilities']):", w_prob, file=out)

                            # Print the word's node word label (index)
                            print("Word (word_label's label property):", w_label)

                            # Print the word's node argmax value
                            # print("Word attributes (local/argmax):", w_argmax_prob, file=out)
                            print("Argmax:", w_argmax)

                            # Print the word text
                            print("Word output token:", word_text)

                            # Append the model's argmax outputs in a temporary node to represent the utterance sequence
                            # Used to compute the sentence accuracy
                            word_results.append(word_text)
                            temp_results.append(word_text)

                            # Append the word's node output to compute the word accuracy
                            # word_labels.append(w.getAttribute("word_label","label").item())
                            # temp_labels.append(w.getAttribute("word_label","label").item())
                            word_labels.append(word_text_lbl)
                            temp_labels.append(word_text_lbl)

                    # # Clean the temp labels
                    temp_results = [w for w in temp_results if w != "<eos>"]
                    temp_labels = [w for w in temp_labels if w != "<eos>"]

                    # Add the current index sequence into the sentence level candidates
                    # to compute the sentence level accuracy
                    sentence_results.append(temp_results)
                    sentence_labels.append(temp_labels)

                    sen_iter_results.append(temp_results)
                    sen_iter_labels.append(temp_labels)

                    # need to clean the sentence
                    # learner_sentence = " ".join(clean_sentence(temp_results[:]))
                    # situation = example['situation'][0]

                    # print("Learner_sentence:",learner_sentence)
                    # print("Situation:", situation)

                    # Test the learner's utterance with the teacher
                    # teacher_sentence, teacher_valid, teacher_feedback, same_predicates = teacher_model.evaluate(None, learner_sentence, None, situation)

                    # print("Teacher Sentence:", teacher_sentence)
                    # print("Teacher feedback:", teacher_feedback)


            # Need to compute the accuracy for each run with test set after every n interactions
            sen_acc, sen_tot = 0.0, 0
            w_acc = 0.0

            w_acc_num = 0
            w_acc_den = 0

            print("Interval Results\n")

            count = 1
            for s, sl in zip(sen_iter_results, sen_iter_labels):
                print("Test example #", count, "\n")
                print("\tPredicted sentence:", " ".join(s))
                print("\tGround Truth:", " ".join(sl))
                print()

                count += 1

                current_sentence_acc = accuracy_similarity(" ".join(s), " ".join(sl))
                result = word_accuracy(" ".join(s), " ".join(sl))
                w_acc_num += result[0]
                w_acc_den += result[1]

                print("Current Word Accuracy: {}/{} = {}".format(result[0], result[1], (result[0] / result[1])))
                print("Current Sentence Accuracy: {}".format(current_sentence_acc))

                sen_tot += current_sentence_acc

            # Tally the final results for the training interval
            try:
                w_acc = w_acc_num / w_acc_den  # Number of words in the current reference
            except:
                w_acc = 0.0

            try:
                sen_acc = sen_tot / len(sen_iter_results)
            except:
                sen_acc = 0.0

            print("\nWord Accuracy: {}/{} = {}".format(w_acc_num, w_acc_den, w_acc))
            print("Sentence Accuracy: {}/{} = {}".format(sen_tot, len(sen_iter_results), sen_acc))

            # Append the current accuracy scores to their respective lists
            word_acc_inter.append(w_acc)
            sen_acc_inter.append(sen_acc)

            temp_w_acc += w_acc
            temp_sen_acc += sen_acc

        # Compute the epoch accuracy of the model
        word_acc_epoch.append(temp_w_acc/inter_count)
        sen_acc_epoch.append(temp_sen_acc/inter_count)

    # Print the final results
    print("Word Accuracy Interaction:", word_acc_inter)
    print("Sentence Accuracy Interaction:", sen_acc_inter)
    print("Word Accuracy Epoch:", word_acc_epoch)
    print("Sentence Accuracy Epoch:", sen_acc_epoch)

    # Finish of run
    end = time.time()

    total_time = end - start  # This is in seconds

    # convert to hour, minutes, and seconds
    elapsed_time = convert_time(total_time)

    print("Procedure elapsed time:", elapsed_time)

    # out.close()  # close the output file



def convert_time(seconds):
    seconds = math.ceil(seconds)  # Set the math to the next largest integer

    print("Total seconds: ", seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int((seconds % 3600) % 60)

    return "{:02d}:{:02d}:{:02d} hh:mm:ss".format(h, m, s)


def clean_sentence(sentence):
    # add the first 'the'
    sentence.insert(0, "the")

    # Find any 'above' or 'below'
    if "above" in sentence:
        ab_index = sentence.index("above")

        # add 'the' after it
        sentence.insert(ab_index + 1, "the")

    if "below" in sentence:
        ab_index = sentence.index("below")

        # add 'the' after it
        sentence.insert(ab_index + 1, "the")

    # Find left or right
    if "left" in sentence:
        ab_index = sentence.index("left")

        # add 'of' and 'the' after the word

        sentence.insert(ab_index + 1, "the")
        sentence.insert(ab_index + 1, "of")

        # add 'the' after it
        sentence.insert(ab_index, "the")
        sentence.insert(ab_index, "to")

    if "right" in sentence:
        ab_index = sentence.index("right")

        # add 'of' and 'the' after the word
        sentence.insert(ab_index + 1, "the")
        sentence.insert(ab_index + 1, "of")

        # add 'the' after it
        sentence.insert(ab_index, "the")
        sentence.insert(ab_index, "to")

    return sentence


def open_file(filename, option):
    '''
        This function tries to open a file and returns the file pointer object.
        Returns None if the file is not opened successfully.
    '''

    fp = None

    try:
        fp = open(filename, option)
    except:
        print("The file:", filename, "was not found!")

    return fp


def accuracy_similarity(model_sentence, truth_sentence, cat_map=None):
    '''
        This function evaluates if the model sentence is an exact match to the truth sentence.
        If cat_map != None, then it means the similarity function converts the words of each sentence to its category.


    '''
    if cat_map != None:

        # Replace the similar words from the model sentence to their category word
        wordlist = model_sentence.split()
        for index in range(len(wordlist)):
            word = wordlist[index]

            # replace the word to the corresponding category
            wordlist[index] = cat_map[word]

        model_sentence = " ".join(wordlist)

        # Replace the similar words from the model sentence to their category word
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
    count = 0  # number of correct words in the hypothesis compared to reference

    # Split the sentences into a list
    reference = reference.split()
    hypothesis = hypothesis.split()

    ref_len = len(reference)
    hyp_len = len(hypothesis)

    denom = 0

    min_len = min(len(reference), len(hypothesis))
    max_len = max(len(reference), len(hypothesis))

    if cat_map == None:

        # Need to iterate through the entire reference sentence and evaluate
        # the total number of expected words.
        for i in range(ref_len):

            if reference[i] == "<eos>":
                break

            denom += 1

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

            if reference[i] == "<eos>":
                break

            denom += 1

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
        score = count / denom
    except:
        # print("Error in reference")
        score = 0.0

    return count, denom


if __name__ == "__main__":
    main()

