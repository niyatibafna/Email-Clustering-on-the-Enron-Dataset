##
 # preprocess.py
 #
 # Archit Checker
 #
 # Take the emails from dataset/emails/processed and preprocess them to make the clean dataset
 # Preprocess: lemmatise, remove stop words, numbers, and employee names
 #
 ##

import sys
sys.path.append("lib/python3.7/site-packages/")

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import lru_cache
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
# nltk.download('punkt')

N_DOCUMENTS = 517402
# N_DOCUMENTS = 2000


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


print("Loading dataset...")
# Load Dataset
X = []
name_dict = set()
for i in range(1, N_DOCUMENTS):
    f = open("dataset/emails/processed/" + str(i + 1) + ".txt", "r")
    clean_string = ""
    for line in f:
        if line[:4] == "From":
            line = line.split(" ")
            for element in line[1].split("@")[0].split("."):
                name_dict.add(element) 
        elif line[:2] == "To":
            line = line.split(" ")
            for element in line[1].split("@")[0].split("."):
                name_dict.add(element) 
        elif line[:7] == "Subject":
            line = line.split(" ")
            clean_string += line[1]
        elif line[:10] == "X-FileName":
            break
    for line in f:
        clean_string += line.strip("\n") + "\n"
    f.close()
    X.append(clean_string)


print("Lemmatising...")
stop_words = set(stopwords.words('english')) 
stop_words.add("ect")
stop_words.add("hou")
stop_words.add("com")
stop_words.add("www")
stop_words.add("http")
stop_words.add("best")
stop_words.add("thanks")
stop_words.add("thank")
stop_words.add("dear")
stop_words.add("please")
for name in name_dict:
    stop_words.add(name)

lemmatizer = WordNetLemmatizer()
lemmatize = lru_cache(maxsize = 50000)(lemmatizer.lemmatize)
for i in range(len(X)):
    sent_text = nltk.sent_tokenize(X[i])
    tokenized_text = []
    
    # Remove the numbers
    for sentence in sent_text:
        sentence = re.sub("[^a-zA-Z]", " ", sentence)
        tokenized_text += list(nltk.word_tokenize(sentence))
    
    # Remove stopwords
    X[i] = [x.lower() for x in tokenized_text if x.lower() not in stop_words and len(x) > 2]
    
    # Pos Tag    
    X[i] = [x for x in nltk.pos_tag(X[i])]
    
    # Lemmatize
    X[i] = list(map(lambda x: lemmatize(x[0], pos = get_wordnet_pos(x[1])), X[i]))
    
    f = open("dataset/clean/"+str(i+1)+".txt", "w")
    f.write(" ".join(X[i]))
    f.close()