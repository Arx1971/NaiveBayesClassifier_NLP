import sys
import os
import re
from collections import Counter
from string import punctuation

import math
import datetime

train_file = os.getcwd() + '/movie-review-HW2/aclImdb/train'
test_file = os.getcwd() + '/movie-review-HW2/aclImdb/test'

FILE = open('movie_review_output.txt', "w+")


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


def read_all_file_name(filepath):
    files = []
    for i in os.listdir(filepath):
        if i.endswith(".txt"):
            files.append(i)
    return files


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
                                             total_vocabulary_size, file, class_name):
    log_probabilities = 0.0
    prob = 1.0
    with open(filepath + test_review, errors='ignore') as reviews:
        review = reviews.read()
        review = re.sub(r'[`=~!@#$%^&*()_+\[\]{};\\:"|<,./<>?^]', ' ', review)
        words = review.split()
        for word in words:
            word = word.lower()
            word = word.strip(punctuation)
            word = word.strip()
            if word in vocabulary:
                prob *= float((vocabulary[word] + 1) / (number_of_word_in_class + total_vocabulary_size))
                val = float((vocabulary[word] + 1) / (number_of_word_in_class + total_vocabulary_size))
                file.write("P(" + word + " | " + class_name + ") = " + str(val) + "\n")
                log_probabilities += math.log(val, 2)
            else:
                prob *= float((1) / (number_of_word_in_class + total_vocabulary_size))
                val = float((1) / (number_of_word_in_class + total_vocabulary_size))
                file.write("P(" + word + " | " + class_name + ") = " + str(val) + "\n")
                log_probabilities += math.log(val, 2)

    return log_probabilities, prob


def small_training_corpus():
    file = open('movie-review-small.NB', 'w+')
    output_file = open('small_corpus_output.txt', 'w+')
    small_test_corpus = read_all_file_name("small_corpus/test")
    small_action_corpus_vocabulary = set_dictionary('action_training_file.txt')
    small_comedy_corpus_vocabulary = set_dictionary('comedy_training_file.txt')
    small_corpus_vocabulary = merge_vocabulary(small_action_corpus_vocabulary, small_comedy_corpus_vocabulary)

    action_class = naive_byes_classifier_bag_of_words_model(small_action_corpus_vocabulary,
                                                            "small_corpus/test/",
                                                            small_test_corpus[0],
                                                            sum_of_values(
                                                                small_action_corpus_vocabulary),
                                                            len(small_corpus_vocabulary),
                                                            file, "Action")
    action_class_log_probabilities = action_class[0] + math.log(float(3 / 5), 2)

    comedy_class = naive_byes_classifier_bag_of_words_model(small_comedy_corpus_vocabulary,
                                                            "small_corpus/test/",
                                                            small_test_corpus[0],
                                                            sum_of_values(
                                                                small_comedy_corpus_vocabulary),
                                                            len(small_corpus_vocabulary),
                                                            file, "Comedy")

    comedy_class_log_probabilities = comedy_class[0] + math.log(float(2 / 5), 2)

    output_file.write("Probabilities for Action Class: " + str(action_class[1] * (3 / 5)) + "\n")
    output_file.write("Probabilities for Comedy Class: " + str(comedy_class[1] * (2 / 5)) + "\n")
    if action_class_log_probabilities > comedy_class_log_probabilities:
        output_file.write("Classifier Prediction: Action" + "\n")
    else:
        output_file.write("Classifier Prediction: Comedy" + "\n")


def log_probability_method(test_files, neg_vocabulary, pos_vocabulary, filepath, training_vocabulary,
                           total_neg_train_file,
                           total_pos_train_file, original_label):
    file = open('movie-review-BOW.NB', 'w+')
    neg_counter_nr = 0
    pos_counter_nr = 0
    sum_of_pos_file = sum_of_values(pos_vocabulary)
    sum_of_neg_file = sum_of_values(neg_vocabulary)
    total_train_file = total_neg_train_file + total_pos_train_file
    len_of_train = len(training_vocabulary)
    for i in range(0, len(test_files)):

        neg = naive_byes_classifier_bag_of_words_model(neg_vocabulary,
                                                       filepath,
                                                       test_files[i],
                                                       sum_of_neg_file,
                                                       len_of_train, file, '-')

        neg_class_log_probabilities = neg[0] + math.log(float(total_neg_train_file / total_train_file), 2)

        pos = naive_byes_classifier_bag_of_words_model(pos_vocabulary,
                                                       filepath,
                                                       test_files[i],
                                                       sum_of_pos_file,
                                                       len_of_train, file, '+')

        pos_class_log_probabilities = pos[0] + math.log(float(total_pos_train_file / total_train_file), 2)

        if pos_class_log_probabilities > neg_class_log_probabilities:
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

    neg_test_arr = log_probability_method(test_neg_file_name, neg_vocabulary, pos_vocabulary,
                                          test_file + "/neg/", vocabulary,
                                          12500, 12500, '-')

    pos_test_arr = log_probability_method(test_pos_file_name, neg_vocabulary, pos_vocabulary,
                                          test_file + "/pos/", vocabulary,
                                          12500, 12500, '+')

    total_accuracy = float((neg_test_arr[0] + pos_test_arr[1]) /
                           (neg_test_arr[0] + neg_test_arr[1] + pos_test_arr[0] + pos_test_arr[1])) * 100

    FILE.write("Total Accuracy: " + str(total_accuracy) + "%")


small_training_corpus()
a = datetime.datetime.now()
naive_byes_classifier()
print(datetime.datetime.now() - a)
