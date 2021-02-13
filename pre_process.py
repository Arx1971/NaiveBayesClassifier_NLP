import os
import re
from string import punctuation

train_file = os.getcwd() + '/movie-review-HW2/aclImdb/train'
test_file = os.getcwd() + '/movie-review-HW2/aclImdb/test'
small_file = os.getcwd() + '/small_corpus/train/'


def write_file(filepath, file_review, filename):
    file = open(filename, 'w+')
    for review_name in file_review:

        with open(filepath + review_name, errors='ignore') as reviews:
            review = reviews.read()
            review = re.sub(r'[`=~!@#$%^&*()_+\[\]{};\\:"|<,./<>?^]', ' ', review)
            words = review.split()
            for word in words:
                word = word.lower()
                word = word.strip(punctuation)
                word = word.strip()
                file.write(word + " ")

    file.close()


def read_all_file_name(filepath):
    files = []
    for i in os.listdir(filepath):
        if i.endswith(".txt"):
            files.append(i)
    return files


pos_file = read_all_file_name(train_file + '/pos')
neg_file = read_all_file_name(train_file + '/neg')

write_file(train_file + '/pos/', pos_file, 'pos_training_file.txt')
write_file(train_file + '/neg/', neg_file, 'neg_training_file.txt')

action_file = read_all_file_name(small_file + '/action')
comedy_file = read_all_file_name(small_file + '/comedy')

write_file(small_file + '/action/', action_file, 'action_training_file.txt')
write_file(small_file + '/comedy/', comedy_file, 'comedy_training_file.txt')
