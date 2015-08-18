# Taken from https://www.dataquest.io/mission/75

import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier

# Use the pandas library in python to read in the csv file.
dato = pandas.read_csv("train.csv")

# A function to get the ngrams from a list.
def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

# A function to get the html file as a list of characters.
def convert_to_characters(input_file):
  with open(input_file, "r") as file:
    return list(file.read().replace('\n',''))

characters = convert_to_characters("train.csv")
trigrams = find_ngrams(characters, 3)
print(list(trigrams))
