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
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import time
start_time = time.time()

def opendoc(filename: str):
    return open(file="docs/" + filename, mode='r', encoding="utf8").read()

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
    return [item[0] for item in most_occur]

# --------------------------------------------------------------------------------------
# Take Care of Books and Text
def purifyText(text: str):
    text = text.lower()
    text = remove_specials(text)
    text = remove_prefixes_suffixes(text)
    text = remove_specials(text)
    text = remove_stopwords(text)
    return text

def getBook(book: str):
    text = opendoc(book)
    return purifyText(text)

def getBooks(books: list):
    text = opendocs(books)
    return purifyText(text)

# For Binary Representation
def getBookKeywords(book: str):
    text = getBook(book)
    frequent = frequencies(text, const.m)
    print("GET_SET Process finished --- %.6s seconds ---" % (time.time() - start_time))
    return frequent

def getBookSetKeywords(books: list):
    text = getBooks(books)
    frequent = frequencies(text, const.m)
    print("GET_SET Process finished --- %.6s seconds ---" % (time.time() - start_time))
    return frequent

# GENERAL
def getTrainSet(bookSet: int):
    if (bookSet == const.BookSet.HARRY_POTTER):
        return const.books[0:53] # 214
    if (bookSet == const.BookSet.GAME_OF_THRONES):
        return const.books[214:242] # 111
    return []

# Get Keywords according to a predefined training set
def getTrainSetKeywords(bookSet: int):
    return getBookSetKeywords(getTrainSet(bookSet))

# --------------------------------------------------------------------------------------
# Representations
def r_binary(keywords: list, books: list):
    test_keywords = []
    for i in range(len(books)):
        test_keywords.append([0] * const.m)

    i = 0
    test_books = books
    while i < len(books):
        book = getBook(test_books[i])
        for j in range(const.m):
            if (keywords[j] in book):
                test_keywords[i][j] = 1
        i += 1
    return test_keywords

#
def r_frequency(filename: str):
    file = opendoc(filename)
    print("B")

def r_tfidf(books: list):
    vectorizer = TfidfVectorizer()

    booksVector = [str(getBook(b)) for b in books]
    vectors = vectorizer.fit_transform(booksVector)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)
    return df.values    # actual conversion to numpy ndarray

#
def r_hadamard(filename: str):
    file = opendoc(filename)
    print("D")

def main():
    print(r_tfidf(getTrainSet(const.BookSet.HARRY_POTTER)))

if __name__ == "__main__":
    main()