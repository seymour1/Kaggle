# This file will get what looks like interesting features
# and append them to the train.csv

from os import listdir
import collections

# A function to get the ngrams from a list.
def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

# A function to get the html file as a list of characters.
def convert_to_characters(input_file):
  with open(input_file, "r") as file:
    return list(file.read().replace('\n',''))

trigram_counts = {}
for f in listdir("/media/sf_VirtualBox_Shared_Folder/0")[0:2]:
  characters = convert_to_characters("/media/sf_VirtualBox_Shared_Folder/0/" + f)
  trigrams = find_ngrams(characters, 3)
  for trigram in trigrams:
    trigram_counts[''.join(trigram)] = trigram_counts.get(''.join(trigram), 0) + 1

ordered_trigram_counts = collections.OrderedDict(sorted(trigram_counts.items()))
for key, value in ordered_trigram_counts.items():
  print(key, value)
#characters = convert_to_characters("train.csv")
#trigrams = find_ngrams(characters, 3)
#print(list(trigrams))
