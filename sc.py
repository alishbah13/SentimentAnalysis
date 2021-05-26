import numpy as np
import pandas as pd
from nltk.corpus import stopwords         
from nltk.stem import snowball      
from nltk.tokenize import RegexpTokenizer, word_tokenize
import matplotlib.pyplot as plt 
import re
import string


data = pd.read_csv('Reviews1.csv')
# print( data.head(1)['Text'] )

def preprocess(revs):
    ## takes reviews and returns pre_processed reviews and dictionary of unique words
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = snowball.SnowballStemmer('english')
    stop_words = stopwords.words('english')
    dictionary = set()
    for review in revs.iteritems():
        print(review[0])
        #name, values in df.iteritems():
        #1 tokenize 
        review = tokenizer.tokenize(review[1])
        #2 lowercase
        review = [w.lower() for w in review]
        #3 remove stopwords 
        review = [wrd for wrd in review if not wrd in stop_words ]
        #4 stem
        review = [stemmer.stem(x) for x in review]
        # print(review)
        # add unique words to dictionary
        dictionary.update(review)
    
    return list(dictionary), revs

words, reviews = preprocess(data['Text'])

# print(words)

# print( reviews[1] )



