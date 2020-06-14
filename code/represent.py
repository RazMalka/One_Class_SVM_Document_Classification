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
from sklearn.preprocessing import normalize
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

def frequencies(text: list, m: int, return_index: int):   
    # Pass the split_it list to instance of Counter class. 
    counter = Counter(text)
    # return index 0 => words, 1 => frequencies
    # most_common() produces k frequently encountered 
    # input values and their respective counts. 
    most_occur = counter.most_common(m)
    return [item[return_index] for item in most_occur]

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
    frequent = frequencies(text, const.m, 0)
    return frequent

def getBookSetKeywords(books: list):
    text = getBooks(books)
    frequent = frequencies(text, const.m, 0)
    return frequent

# GENERAL
def getTrainSet(bookSet: int):
    if (bookSet == const.BookSet.HARRY_POTTER):
        return const.books[29:70] # 25% of training set
    if (bookSet == const.BookSet.GAME_OF_THRONES):
        return const.books[214:242] # 111
    return []

# Get Keywords according to a predefined training set
def getTrainSetKeywords(bookSet: int):
    return getBookSetKeywords(getTrainSet(bookSet))

# --------------------------------------------------------------------------------------
# Representations

# Binary representation of a specific document - choose the m dimensional binary vector 
# where the ith entry is 1 if the ith keyword appears in the document and 0 if it does not.
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

# frequency representation - choose the m dimensional real valued vector, 
# where the ith entry is the normalized frequency of appearance of the ith keyword in the specific document.
def r_frequency(keywords: list, books: list):
    test_keywords = []
    for i in range(len(books)):
        test_keywords.append([0] * const.m)

    i = 0
    test_books = books
    while i < len(books):
        book = getBook(test_books[i])
        for j in range(const.m):
            if (keywords[j] in book):
                test_keywords[i][j] = book.count(keywords[j])
        i += 1
    
    return normalize(test_keywords)

# tf-idf representation (“term frequency inverse document frequency”).
# It is used as a weighting factor in searches of information retrieval and text mining.
# The tf–idf value increases proportionally to the number of times a word appears in the document,
# and is offset by the number of documents in the corpus that contain the word.
def r_tfidf(vectorizer: TfidfVectorizer, books, onlyTransform: int):
    dataBooksVector = [str(getBook(b)) for b in books]
    if onlyTransform == 0:
        x_data = vectorizer.fit_transform(dataBooksVector)
    else:
        x_data = vectorizer.transform(dataBooksVector)
    feature_names = vectorizer.get_feature_names()
    dense = x_data.toarray()
    denselist = dense.tolist()
    return pd.DataFrame(denselist, columns=feature_names).values

# Hadamard product representation - consists of the m dimensional vector, 
# where the ith entry is the product of the frequency of the ith keyword in the document, 
# and its frequency over all documents (in the training set).
def r_hadamard(keywords: list, train_books: list, test_books: list):
    return frequencies(getBooks(train_books), const.m, 1) * r_frequency(keywords, test_books)

def main():
    print(r_tfidf(getTrainSet(const.BookSet.HARRY_POTTER)))

if __name__ == "__main__":
    main()