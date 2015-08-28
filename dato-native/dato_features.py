# This file will get what looks like interesting features
# and append them to the train.csv

from os import listdir
import collections
import operator
import math

# A function to get the ngrams from a list.
def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

# A function to get the html file as a list of characters.
def convert_to_characters(input_file):
  with open(input_file, "r") as file:
    return list(file.read().replace('\n',''))

# Get hashmaps of frequencies of n_grams for each file
bigram_files = {}
trigram_files = {}
for f in listdir("/media/sf_VirtualBox_Shared_Folder/0")[0:10]:
  bigram_files[f] = {}
  trigram_files[f] = {}
  characters = convert_to_characters("/media/sf_VirtualBox_Shared_Folder/0/" + f)
  bigrams = find_ngrams(characters, 2)
  trigrams = find_ngrams(characters, 3)
  for bigram in bigrams:
    bigram_files[f][''.join(bigram)] = bigram_files[f].get(''.join(bigram), 0) + 1
  for trigram in trigrams:
    trigram_files[f][''.join(trigram)] = trigram_files[f].get(''.join(trigram), 0) + 1

# Calculate tf_idfs for each token
# tf_idf = frequency * log(num_docs / num_docs_with_term) / num_terms_in_doc
tfidfs = {}
num_docs = len(bigram_files)
for file, tokens in bigram_files.items():
  tfidfs[file] = {}

  # Find the normalization constant
  num_terms_in_doc = 0
  for token, frequency in tokens.items():
    num_terms_in_doc += frequency

  for token, frequency in tokens.items():
    # Find number of documents with term.
    num_docs_with_term = 0
    for file2 in bigram_files:
      if bigram_files[file2].get(token, 0) != 0:
        num_docs_with_term += 1
    idf = math.log(1.0 * num_docs / num_docs_with_term)


    tfidfs[file][token] = frequency * idf / num_terms_in_doc

  ordered_bigrams = collections.OrderedDict(sorted(tfidfs[file].items(), key=operator.itemgetter(1)))
  for key, value in ordered_bigrams.items():
    print(key, value)
