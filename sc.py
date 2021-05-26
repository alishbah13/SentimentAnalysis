import numpy as np
import pandas as pd
from nltk.corpus import stopwords         
from nltk.stem import snowball      
from nltk.tokenize import RegexpTokenizer, word_tokenize
import matplotlib.pyplot as plt 
import re
import string


data = pd.read_csv('Reviews1.csv')
# print( type( data.head()) )

def get_pos_neg(dataset):
    ## get all the positive and negative labelled reviews
    pos = []
    neg = []
    for index, revs in dataset.iterrows():
        if revs.Recommended == 1:
            pos.append( revs['Text'] )
        else:
            neg.append( revs['Text']) 
    # positives = 19314; negatives = 4172
    return pos, neg  

positive_reviews, negative_reviews = get_pos_neg( data ) 
print(len(positive_reviews), " " , len(negative_reviews))


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


words, data = preprocess(data)
print( data.head())

# The keys are a tuple (word, label) and 
# the values are the corresponding frequency. 
# The labels we'll use here are 1 for positive and 0 for negative.

def make_count(dictionary, data):
    
    for index, row in data.iterrows():