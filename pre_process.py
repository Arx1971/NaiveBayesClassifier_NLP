import os
import re
from collections import Counter
from string import punctuation


#  regular expression r'[\s+`=~!@#$%^&*()_+\[\]{};\--\\:"|<,./<>?^]'

def file_name_viewer(file_name):
    for name in file_name:
        print(name)
    print(len(file_name))


def sum_of_values(dictionary):
    total = 0
    for key, values in dictionary.items():
        total += values
    return total


def dictionary_frequency_viewer(dictionary):
    for values, key in dictionary.items():
        print(values, key)


def read_all_file_name(filepath):
    files = []
    for i in os.listdir(filepath):
        if i.endswith(".txt"):
            files.append(i)
    return files


def set_vocabulary(review, filepath):
    vocabulary = dict()
    for review_name in review:
        with open(filepath + review_name, "r") as reviews:
            review = reviews.read()
            review = re.sub(r'[`=~!@#$%^&*()_+\[\]{};\\:"|<,./<>?^]', ' ', review)
            words = review.split()
            for word in words:
                word = word.lower()
                word = word.strip(punctuation)
                word = word.strip()
                if len(word) is not 0:
                    if word in vocabulary:
                        vocabulary[word] += 1
                    else:
                        vocabulary[word] = 1
    return vocabulary


def merge_vocabulary(vocabulary_1, vocabulary_2):
    x = Counter(vocabulary_1)
    y = Counter(vocabulary_2)
    x.update(y)
    return dict(x)