# Data Mining and Machine Learning 2020
# Group Number 2    ->  [Raz Malka | Shoham Yamin | Raz Itzhak Afriat]
# Research Subject  ->  One-Class SVMs for Document Classification

# This file is meant to provide implementation of required representations
# Binary, Frequency, TF-IDF, Hadamard
import const
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer

# Data Analysis Libraries
from collections import Counter
import numpy as np

def opendoc(filename: str):
    return open(file="docs/" + filename + ".txt", mode='r', encoding="utf8").read()

def opendocs(filenames: list):
    text = ""
    for filename in filenames:
        text = text + opendoc(filename)
    return text

def remove_specials(text: str):
    for w in const.special:
        text = text.replace(w, "")
    return text
    
def remove_stopwords(text: str):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w in stop_words]
    filtered_text = []
  
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_text.append(w)
    return filtered_text

def remove_prefixes_suffixes(text: str): #slow
    word_tokens = word_tokenize(text)
    filtered_text = []
    
    stemmer = LancasterStemmer()
    for w in word_tokens:
        w = stemmer.stem(w)
        filtered_text.append(w)
    return str(filtered_text)

def frequencies(text: list, m: int):   
    # Pass the split_it list to instance of Counter class. 
    counter = Counter(text)
    
    # most_common() produces k frequently encountered 
    # input values and their respective counts. 
    most_occur = counter.most_common(m)
    return most_occur

#
def r_binary(filename: str):
    text = opendoc(filename)
    print("A")

#
def r_frequency(filename: str):
    file = opendoc(filename)
    print("B")

#
def r_tfidf(filename: str):
    file = opendoc(filename)
    print("C")

#
def r_hadamard(filename: str):
    file = opendoc(filename)
    print("D")

def main():
    text = opendocs(const.books[1:7])
    text = text.lower()
    text = remove_specials(text)
    text = remove_prefixes_suffixes(text)
    text = remove_specials(text)
    text = remove_stopwords(text)
    frequency = frequencies(text, 20)
    print(frequency)

if __name__ == "__main__":
    main()