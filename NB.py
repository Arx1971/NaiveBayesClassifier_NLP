import sys
import os
import re
from collections import Counter
from string import punctuation
from pre_process import *


# args = sys.argv[1].split()
# train_file = args[0]
# test_file = args[1]
# train_write_file = args[2]
#
# print(train_file, test_file, train_write_file)
# print(os.path.isdir())


def naive_byes_classifier_bag_of_words_model(vocabulary, filepath, test_review, number_of_word_in_class,
                                             total_vocabulary_size):
    prob = 1.0
    with open(filepath + test_review, "r") as reviews:
        review = reviews.read()
        review = re.sub(r'[`=~!@#$%^&*()_+\[\]{};\\:"|<,./<>?^]', ' ', review)
        words = review.split()
        for word in words:
            word = word.lower()
            word = word.strip(punctuation)
            word = word.strip()
            if word in vocabulary:
                prob *= float((vocabulary[word] + 1) / (number_of_word_in_class + total_vocabulary_size))
            else:
                prob *= float((1) / (number_of_word_in_class + total_vocabulary_size))

    return prob


def small_training_corpus():
    small_training_action_corpus = read_all_file_name("small_corpus/train/action")
    small_training_comedy_corpus = read_all_file_name("small_corpus/train/comedy")
    small_test_corpus = read_all_file_name("small_corpus/test")
    small_action_corpus_vocabulary = set_vocabulary(small_training_action_corpus, "small_corpus/train/action/")
    small_comedy_corpus_vocabulary = set_vocabulary(small_training_comedy_corpus, "small_corpus//train/comedy/")
    small_corpus_vocabulary = merge_vocabulary(small_action_corpus_vocabulary, small_comedy_corpus_vocabulary)

    # print(small_corpus_vocabulary)
    # print(small_action_corpus_vocabulary)
    # print(small_comedy_corpus_vocabulary)

    total_number_of_training_files = (len(small_training_comedy_corpus) + len(small_training_action_corpus))
    total_action_training_files = len(small_training_action_corpus)
    total_comedy_training_files = len(small_training_comedy_corpus)
    action_class_prob = naive_byes_classifier_bag_of_words_model(small_action_corpus_vocabulary,
                                                                 "small_corpus/test/",
                                                                 small_test_corpus[0],
                                                                 sum_of_values(small_action_corpus_vocabulary),
                                                                 len(small_corpus_vocabulary)) * float(
        total_action_training_files / total_number_of_training_files)

    comedy_class_prob = naive_byes_classifier_bag_of_words_model(small_comedy_corpus_vocabulary,
                                                                 "small_corpus/test/",
                                                                 small_test_corpus[0],
                                                                 sum_of_values(small_comedy_corpus_vocabulary),
                                                                 len(small_corpus_vocabulary)) * float(
        total_comedy_training_files / total_number_of_training_files)
    print("Probabilities for Action Class: ", action_class_prob)
    print("Probabilities for Comedy Class: ", comedy_class_prob)
    if action_class_prob > comedy_class_prob:
        print("Document Belong to Action Class. ")
    else:
        print("Document Belong to Comedy Class. ")


def probability_method(test_files, neg_vocabulary, pos_vocabulary, filepath, training_vocabulary, total_neg_train_file,
                       total_pos_train_file):
    neg_counter_nr = 0
    pos_counter_nr = 0
    total_test_file = total_neg_train_file + total_pos_train_file
    for i in range(0, len(test_files)):
        neg_class_prob = naive_byes_classifier_bag_of_words_model(neg_vocabulary,
                                                                  filepath,
                                                                  test_files[i],
                                                                  sum_of_values(neg_vocabulary),
                                                                  len(training_vocabulary)) * float(
            total_neg_train_file / total_test_file)

        pos_class_prob = naive_byes_classifier_bag_of_words_model(pos_vocabulary,
                                                                  filepath,
                                                                  test_files[i],
                                                                  sum_of_values(pos_vocabulary),
                                                                  len(training_vocabulary)) * float(
            total_pos_train_file / total_test_file)

        if pos_class_prob > neg_class_prob:
            pos_counter_nr += 1
        else:
            neg_counter_nr += 1

    return neg_counter_nr, pos_counter_nr


def naive_byes_classifier():
    training_pos_file_name = read_all_file_name("movie-review-HW2/aclImdb/train/pos")
    training_neg_file_name = read_all_file_name("movie-review-HW2/aclImdb/train/neg")
    test_pos_file_name = read_all_file_name("movie-review-HW2/aclImdb/test/pos")
    test_neg_file_name = read_all_file_name("movie-review-HW2/aclImdb/test/neg")

    neg_vocabulary = set_vocabulary(training_neg_file_name, 'movie-review-HW2/aclImdb/train/neg/')
    pos_vocabulary = set_vocabulary(training_pos_file_name, 'movie-review-HW2/aclImdb/train/pos/')
    training_vocabulary = merge_vocabulary(neg_vocabulary, pos_vocabulary)
    total_neg_train_file = len(training_neg_file_name)
    total_pos_train_file = len(training_pos_file_name)

    neg_test_arr = probability_method(test_neg_file_name, neg_vocabulary, pos_vocabulary,
                                      "movie-review-HW2/aclImdb/test/neg/", training_vocabulary,
                                      total_neg_train_file, total_pos_train_file)
    print("Total Number of negative review in neg class: ", neg_test_arr[0], "Probability: ",
          float(neg_test_arr[0] / total_neg_train_file))
    print("Total Number of positive review in neg class: ", neg_test_arr[1], "Probability: ",
          float(neg_test_arr[1] / total_neg_train_file))
    pos_test_arr = probability_method(test_pos_file_name, neg_vocabulary, pos_vocabulary,
                                      "movie-review-HW2/aclImdb/test/pos/", training_vocabulary,
                                      total_neg_train_file, total_pos_train_file)
    print("Total Number of negative review in pos class: ", pos_test_arr[0], "Probability: ",
          float(pos_test_arr[0] / total_pos_train_file))
    print("Total Number of positive review in pos class: ", pos_test_arr[1], "Probability: ",
          float(pos_test_arr[1] / total_pos_train_file))


small_training_corpus()
naive_byes_classifier()
