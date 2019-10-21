import sys
import os
import re
from collections import Counter
from string import punctuation
from pre_process import *
import math
import datetime

# args = sys.argv[1].split()
# train_file = args[0]
# test_file = args[1]
# train_write_file = args[2]
# FILE = open(train_write_file + 'mega_file.txt', "w+")


train_file = '/home/adnan/source-code/PycharmProjects/NaiveBayesClassifier_NLP/movie-review-HW2/aclImdb/train'
test_file = '/home/adnan/source-code/PycharmProjects/NaiveBayesClassifier_NLP/movie-review-HW2/aclImdb/test'
train_write_file = '/home/adnan/source-code/PycharmProjects/NaiveBayesClassifier_NLP/'
FILE = open('prediction_file.txt', "w")


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
    small_training_action_corpus = read_all_file_name("small_corpus/train/action")
    small_training_comedy_corpus = read_all_file_name("small_corpus/train/comedy")
    small_test_corpus = read_all_file_name("small_corpus/test")
    small_action_corpus_vocabulary = set_vocabulary(small_training_action_corpus, "small_corpus/train/action/")
    small_comedy_corpus_vocabulary = set_vocabulary(small_training_comedy_corpus, "small_corpus//train/comedy/")
    small_corpus_vocabulary = merge_vocabulary(small_action_corpus_vocabulary, small_comedy_corpus_vocabulary)

    total_number_of_training_files = (len(small_training_comedy_corpus) + len(small_training_action_corpus))
    total_action_training_files = len(small_training_action_corpus)
    total_comedy_training_files = len(small_training_comedy_corpus)
    action_class_prob = naive_byes_classifier_bag_of_words_model(small_action_corpus_vocabulary,
                                                                 "small_corpus/test/",
                                                                 small_test_corpus[0],
                                                                 sum_of_values(small_action_corpus_vocabulary),
                                                                 len(small_corpus_vocabulary)) + math.log(
        float(total_action_training_files / total_number_of_training_files), 2)

    comedy_class_prob = naive_byes_classifier_bag_of_words_model(small_comedy_corpus_vocabulary,
                                                                 "small_corpus/test/",
                                                                 small_test_corpus[0],
                                                                 sum_of_values(small_comedy_corpus_vocabulary),
                                                                 len(small_corpus_vocabulary)) + math.log(
        float(total_comedy_training_files / total_number_of_training_files), 2)

    print("Log Probabilities for Action Class: ", action_class_prob)
    print("Log Probabilities for Comedy Class: ", comedy_class_prob)
    if action_class_prob > comedy_class_prob:
        print("Document Belong to Action Class. ")
    else:
        print("Document Belong to Comedy Class. ")


def probability_method(test_files, neg_vocabulary, pos_vocabulary, filepath, training_vocabulary, total_neg_train_file,
                       total_pos_train_file):
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
            FILE.write(test_files[i] + " Document Belong to test Negative: " + str(pos_class_prob) + "\n")
        else:
            neg_counter_nr += 1
            FILE.write(test_files[i] + " Document Belong to test Positive: " + str(neg_class_prob) + "\n")

    return neg_counter_nr, pos_counter_nr


# def naive_byes_classifier():
#     training_pos_file_name = read_all_file_name("movie-review-HW2/aclImdb/train/pos")
#     training_neg_file_name = read_all_file_name("movie-review-HW2/aclImdb/train/neg")
#     test_pos_file_name = read_all_file_name("movie-review-HW2/aclImdb/test/pos")
#     test_neg_file_name = read_all_file_name("movie-review-HW2/aclImdb/test/neg")
#
#     neg_vocabulary = set_vocabulary(training_neg_file_name, 'movie-review-HW2/aclImdb/train/neg/')
#     pos_vocabulary = set_vocabulary(training_pos_file_name, 'movie-review-HW2/aclImdb/train/pos/')
#     training_vocabulary = merge_vocabulary(neg_vocabulary, pos_vocabulary)
#     total_neg_train_file = len(training_neg_file_name)
#     total_pos_train_file = len(training_pos_file_name)
#     file = open('vocabulary.txt', 'w+')
#     for keys, values in training_vocabulary.items():
#         file.write(keys + " " + str(values) + "\n")
#     file.close()
#     FILE.write("Process Test Negative Reviews\n")
#     neg_test_arr = probability_method(test_neg_file_name, neg_vocabulary, pos_vocabulary,
#                                       "movie-review-HW2/aclImdb/test/neg/", training_vocabulary,
#                                       total_neg_train_file, total_pos_train_file)
#     print("Total Number of negative review in neg class: ", neg_test_arr[0], "Probability: ",
#           float(neg_test_arr[0] / total_neg_train_file))
#
#     FILE.write("Total Number of negative review in neg class: " + str(neg_test_arr[0]) + " Probability: " +
#                str(float(neg_test_arr[0] / total_pos_train_file)) + "\n")
#
#     print("Total Number of positive review in neg class: ", neg_test_arr[1], "Probability: ",
#           float(neg_test_arr[1] / total_neg_train_file))
#
#     FILE.write("Total Number of positive review in neg class: " + str(neg_test_arr[1]) + " Probability: " +
#                str(float(neg_test_arr[1] / total_pos_train_file)) + "\n")
#
#     FILE.write("Process Test Positive Reviews\n")
#     pos_test_arr = probability_method(test_pos_file_name, neg_vocabulary, pos_vocabulary,
#                                       "movie-review-HW2/aclImdb/test/pos/", training_vocabulary,
#                                       total_neg_train_file, total_pos_train_file)
#     print("Total Number of negative review in pos class: ", pos_test_arr[0], "Probability: ",
#           float(pos_test_arr[0] / total_pos_train_file))
#
#     FILE.write("Total Number of positive review in pos class: " + str(pos_test_arr[0]) + " Probability: " +
#                str(float(pos_test_arr[0] / total_pos_train_file)) + "\n")
#
#     print("Total Number of positive review in pos class: ", pos_test_arr[1], "Probability: ",
#           float(pos_test_arr[1] / total_pos_train_file))
#
#     FILE.write("Total Number of positive review in pos class: " + str(pos_test_arr[1]) + " Probability: " +
#                str(float(pos_test_arr[1] / total_pos_train_file)))


def naive_byes_classifier():
    training_pos_file_name = read_all_file_name(train_file + "/pos")
    training_neg_file_name = read_all_file_name(train_file + "/neg")
    test_pos_file_name = read_all_file_name(test_file + "/pos")
    test_neg_file_name = read_all_file_name(test_file + "/neg")

    neg_vocabulary = set_vocabulary(training_neg_file_name, train_file + '/neg/')
    pos_vocabulary = set_vocabulary(training_pos_file_name, train_file + '/pos/')
    training_vocabulary = merge_vocabulary(neg_vocabulary, pos_vocabulary)
    total_neg_train_file = len(training_neg_file_name)
    total_pos_train_file = len(training_pos_file_name)
    file = open('vocabulary.txt', 'w+')
    for keys, values in training_vocabulary.items():
        file.write(keys + " " + str(values) + "\n")
    file.close()
    FILE.write("Process Test Negative Reviews\n")
    neg_test_arr = probability_method(test_neg_file_name, neg_vocabulary, pos_vocabulary,
                                      test_file + "/neg/", training_vocabulary,
                                      total_neg_train_file, total_pos_train_file)
    print("Total Number of negative review in neg class: ", neg_test_arr[0], "Probability: ",
          float(neg_test_arr[0] / total_neg_train_file))

    FILE.write("Total Number of negative review in neg class: " + str(neg_test_arr[0]) + " Probability: " +
               str(float(neg_test_arr[0] / total_pos_train_file)) + "\n")

    print("Total Number of positive review in neg class: ", neg_test_arr[1], "Probability: ",
          float(neg_test_arr[1] / total_neg_train_file))

    FILE.write("Total Number of positive review in neg class: " + str(neg_test_arr[1]) + " Probability: " +
               str(float(neg_test_arr[1] / total_pos_train_file)) + "\n")

    FILE.write("Process Test Positive Reviews\n")
    pos_test_arr = probability_method(test_pos_file_name, neg_vocabulary, pos_vocabulary,
                                      test_file + "/pos/", training_vocabulary,
                                      total_neg_train_file, total_pos_train_file)
    print("Total Number of negative review in pos class: ", pos_test_arr[0], "Probability: ",
          float(pos_test_arr[0] / total_pos_train_file))

    FILE.write("Total Number of positive review in pos class: " + str(pos_test_arr[0]) + " Probability: " +
               str(float(pos_test_arr[0] / total_pos_train_file)) + "\n")

    print("Total Number of positive review in pos class: ", pos_test_arr[1], "Probability: ",
          float(pos_test_arr[1] / total_pos_train_file))

    FILE.write("Total Number of positive review in pos class: " + str(pos_test_arr[1]) + " Probability: " +
               str(float(pos_test_arr[1] / total_pos_train_file)))


small_training_corpus()
a = datetime.datetime.now()
naive_byes_classifier()
print(datetime.datetime.now() - a)

# python3.6 /home/adnan/source-code/PycharmProjects/NaiveBayesClassifier_NLP/NB.py '/home/adnan/source-code/PycharmProjects/NaiveBayesClassifier_NLP/movie-review-HW2/aclImdb/train /home/adnan/source-code/PycharmProjects/NaiveBayesClassifier_NLP/movie-review-HW2/aclImdb/test /home/adnan/source-code/PycharmProjects/NaiveBayesClassifier_NLP/'
