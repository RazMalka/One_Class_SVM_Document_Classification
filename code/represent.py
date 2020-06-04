# Data Mining and Machine Learning 2020
# Group Number 2    ->  [Raz Malka | Shoham Yamin | Raz Itzhak Afriat]
# Research Subject  ->  One-Class SVMs for Document Classification

# This file is meant to provide implementation of required representations
# Binary, Frequency, TF-IDF, Hadamard
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Data Analysis Libraries
from collections import Counter
import numpy as np
from scipy import stats

special = [".", ",", "’", "“", "”", "—", "?", "!", "|", "page", "chapter", "-", ";", ":"]

def opendoc(filename: str):
    return open(file="docs/" + filename + ".txt", mode='r', encoding="utf8").read()

def remove_stopwords(text: str):
    text = text.lower()
    for w in special:
        text = text.replace(w, "")

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [w for w in word_tokens if not w in stop_words]
    filtered_text = []
  
    for w in word_tokens: 
        if w not in stop_words: 
            filtered_text.append(w)
    return filtered_text

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
    text = opendoc("Book 1 - The Philosopher's Stone (A)") # All Files Should Be 'Scanned'
    text = remove_stopwords(text)
    frequency = frequencies(text, 200)
    print(frequency)

if __name__ == "__main__":
    main()