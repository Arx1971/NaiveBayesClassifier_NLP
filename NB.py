import sys
import os
import re
from collections import Counter
from string import punctuation
from pre_process import *
import math
import datetime

train_file = os.getcwd() + '/movie-review-HW2/aclImdb/train'
test_file = os.getcwd() + '/movie-review-HW2/aclImdb/test'

FILE = open('prediction_file.txt', "w+")


def sum_of_values(dictionary):
    total = 0
    for key, values in dictionary.items():
        total += values
    return total


def dictionary_frequency_viewer(dictionary):
    for values, key in dictionary.items():
        print(values, key)


def merge_vocabulary(vocabulary_1, vocabulary_2):
    x = Counter(vocabulary_1)
    y = Counter(vocabulary_2)
    x.update(y)
    return dict(x)


def set_dictionary(filename):
    dictionary = dict()
    file = open(filename, 'r')
    review = file.read()
    words = review.split()
    for word in words:
        if word in dictionary:
            dictionary[word] += 1
        else:
            dictionary[word] = 1

    return dictionary


def naive_byes_classifier_bag_of_words_model(vocabulary, filepath, test_review, number_of_word_in_class,
                                             total_vocabulary_size):
    prob = 0.0
    with open(filepath + test_review, "r") as reviews:
        review = reviews.read()
        review = re.sub(r'[`=~!@#$%^&*()_+\[\]{};\\:"|<,./<>?^]', ' ', review)
        words = review.split()
        for word in words:
            word = word.lower()
            word = word.strip(punctuation)
            word = word.strip()
            if word in vocabulary:
                prob += math.log(float((vocabulary[word] + 1) / (number_of_word_in_class + total_vocabulary_size)), 2)
            else:
                prob += math.log(float((1) / (number_of_word_in_class + total_vocabulary_size)), 2)

    return prob


def small_training_corpus():
    small_test_corpus = read_all_file_name("small_corpus/test")
    small_action_corpus_vocabulary = set_dictionary('small_action_train_file.txt')
    small_comedy_corpus_vocabulary = set_dictionary('small_action_train_file.txt')
    small_corpus_vocabulary = merge_vocabulary(small_action_corpus_vocabulary, small_comedy_corpus_vocabulary)

    action_class_prob = naive_byes_classifier_bag_of_words_model(small_action_corpus_vocabulary,
                                                                 "small_corpus/test/",
                                                                 small_test_corpus[0],
                                                                 sum_of_values(small_action_corpus_vocabulary),
                                                                 len(small_corpus_vocabulary)) + math.log(
        float(3 / 5), 2)

    comedy_class_prob = naive_byes_classifier_bag_of_words_model(small_comedy_corpus_vocabulary,
                                                                 "small_corpus/test/",
                                                                 small_test_corpus[0],
                                                                 sum_of_values(small_comedy_corpus_vocabulary),
                                                                 len(small_corpus_vocabulary)) + math.log(
        float(2 / 5), 2)

    print("Log Probabilities for Action Class: ", action_class_prob)
    print("Log Probabilities for Comedy Class: ", comedy_class_prob)
    if action_class_prob > comedy_class_prob:
        print("Document Belong to Action Class. ")
    else:
        print("Document Belong to Comedy Class. ")


def probability_method(test_files, neg_vocabulary, pos_vocabulary, filepath, training_vocabulary, total_neg_train_file,
                       total_pos_train_file, original_label):
    neg_counter_nr = 0
    pos_counter_nr = 0
    sum_of_pos_file = sum_of_values(pos_vocabulary)
    sum_of_neg_file = sum_of_values(neg_vocabulary)
    total_train_file = total_neg_train_file + total_pos_train_file
    len_of_train = len(training_vocabulary)
    for i in range(0, len(test_files)):
        neg_class_prob = naive_byes_classifier_bag_of_words_model(neg_vocabulary,
                                                                  filepath,
                                                                  test_files[i],
                                                                  sum_of_neg_file,
                                                                  len_of_train) + math.log(
            float(total_neg_train_file / total_train_file), 2)

        pos_class_prob = naive_byes_classifier_bag_of_words_model(pos_vocabulary,
                                                                  filepath,
                                                                  test_files[i],
                                                                  sum_of_pos_file,
                                                                  len_of_train) + math.log(
            float(total_pos_train_file / total_train_file), 2)

        if pos_class_prob > neg_class_prob:
            pos_counter_nr += 1
            FILE.write(test_files[i] + " Original Label: " + original_label + " Predict Label: + " + "\n")
        else:
            neg_counter_nr += 1
            FILE.write(test_files[i] + " Original Label: " + original_label + " Predict Label: - " + "\n")

    return neg_counter_nr, pos_counter_nr


def naive_byes_classifier():
    pos_vocabulary = set_dictionary('pos_training_file.txt')
    neg_vocabulary = set_dictionary('neg_training_file.txt')
    vocabulary = merge_vocabulary(neg_vocabulary, pos_vocabulary)

    test_pos_file_name = read_all_file_name(test_file + "/pos")
    test_neg_file_name = read_all_file_name(test_file + "/neg")

    neg_test_arr = probability_method(test_neg_file_name, neg_vocabulary, pos_vocabulary,
                                      test_file + "/neg/", vocabulary,
                                      12500, 12500, '-')
    # print("Total Number of negative review in neg class: ", neg_test_arr[0], "Probability: ",
    #       float(neg_test_arr[0] / 12500))
    #
    # print("Total Number of positive review in neg class: ", neg_test_arr[1], "Probability: ",
    #       float(neg_test_arr[1] / 12500))

    pos_test_arr = probability_method(test_pos_file_name, neg_vocabulary, pos_vocabulary,
                                      test_file + "/pos/", vocabulary,
                                      12500, 12500, '+')
    # print("Total Number of negative review in pos class: ", pos_test_arr[0], "Probability: ",
    #       float(pos_test_arr[0] / 12500))
    #
    # print("Total Number of positive review in pos class: ", pos_test_arr[1], "Probability: ",
    #       float(pos_test_arr[1] / 12500))

    total_accuracy = float((neg_test_arr[0] + pos_test_arr[1]) /
                           (neg_test_arr[0] + neg_test_arr[1] + pos_test_arr[0] + pos_test_arr[1])) * 100

    FILE.write("Total Accuracy: " + str(total_accuracy) + "%")


small_training_corpus()
a = datetime.datetime.now()
naive_byes_classifier()
print(datetime.datetime.now() - a)
