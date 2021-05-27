from nltk.sem.logic import TypeResolutionException
import numpy as np
import pandas as pd
from nltk.corpus import stopwords         
from nltk.stem import snowball      
from nltk.tokenize import RegexpTokenizer, word_tokenize
import matplotlib.pyplot as plt 
import re
import string

def get_pos_neg(dataset):
    ## get all the positive and negative labelled reviews
    pos = []
    neg = []
    for index, revs in dataset.iterrows():
        if revs.Recommended == 1:
            pos.append( revs )
        else:
            neg.append( revs) 
    # positives = 18540; negatives = 4101
    return pos, neg 

def preprocess(revs):
    ## takes reviews and returns pre_processed reviews and dictionary of unique words
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = snowball.SnowballStemmer('english')
    stop_words = stopwords.words('english')
    dictionary = set()
    for index, review in revs.iterrows():
        #1 tokenize 
        review['Text'] = tokenizer.tokenize(review['Text'])
        #2 lowercase
        review['Text'] = [w.lower() for w in review['Text']]
        #3 remove stopwords 
        review['Text'] = [wrd for wrd in review['Text'] if not wrd in stop_words ]
        #4 stem
        review['Text'] = [stemmer.stem(x) for x in review['Text']]
        revs['Text'][index] = review['Text']
        # print( revs['Text'][index] )
        # add unique words to dictionary
        dictionary.update(review['Text'])
    
    return dictionary, revs

# The keys are a tuple (word, label) and 
# the values are the corresponding frequency. 
# The labels we'll use here are 1 for positive and 0 for negative.

def make_count(dictionary, data):
    word_freq = {}
    for index, row in data.iterrows():
        for word in row['Text']:
            if word in dictionary:
                # check frequency of that word in the specific review
                freq = len( [i for i, x in enumerate(row['Text']) if x == word] )
                temp = (word, row['Recommended'])
                # word_freq[ temp ] += 1
                if temp in word_freq.keys():
                    word_freq[ temp ] += 1
                else:
                    word_freq[ temp ] = 1
    return word_freq


data = pd.read_csv('Reviews1.csv')
# print( type( data.head()) )
words, data = preprocess(data)
positive_reviews, negative_reviews = get_pos_neg( data ) 
# print(len(positive_reviews), " " , len(negative_reviews))

# training set = 80% 
# testing set = 20%
# positives = 18540; negatives = 4101

train_pos = positive_reviews[:14832]
test_pos = positive_reviews[14832:]

train_neg = negative_reviews[:3281]
test_neg = negative_reviews[3281:]

training_set = train_pos + train_neg
test_set = test_pos + test_neg

ratings = make_count(words, data)
# print(ratings[('silki', 1)])
print(training_set)
print(test_set)