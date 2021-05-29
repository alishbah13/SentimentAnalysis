from nltk.probability import log_likelihood
from nltk.sem.logic import TypeResolutionException
import numpy as np
import pandas as pd
from nltk.corpus import stopwords         
from nltk.stem import snowball      
from nltk.tokenize import RegexpTokenizer, word_tokenize
from statistics import mean
import matplotlib.pyplot as plt 
import re
import string
import math

def get_pos_neg(dataset):
    ## get all the positive and negative labelled reviews
    pos = [] 
    neg = []
    for index, revs in dataset.iterrows():
        if revs.Recommended == 1:
            pos.append( revs.Text )
        else:
            neg.append( revs.Text) 
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
        # add unique words to dictionary
        dictionary.update(review['Text'])
    
    return dictionary, revs

def process_review(review):
    ## takes reviews and returns processed tokens
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = snowball.SnowballStemmer('english')
    stop_words = stopwords.words('english')
    #1 tokenize 
    review_tokens = tokenizer.tokenize(review)
    #2 lowercase
    review_tokens = [w.lower() for w in review_tokens]
    #3 remove stopwords 
    review_tokens= [stemmer.stem(wrd) for wrd in review_tokens if not wrd in stop_words ]

    return review_tokens



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
                if temp in word_freq.keys():
                    word_freq[ temp ] += 1
                else:
                    word_freq[ temp ] = 1
    return word_freq


def training_naive_bayes(ratings, training_set):
    loglikelihood = {}
    logprior = 0

    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in ratings.keys()])
    V = len(vocab)

    # calculate N_pos and N_neg
    N_pos = N_neg = 0
    for pair in ratings.keys():
        if pair[1] > 0:
            N_pos += ratings[pair]
        else:
            N_neg += ratings[pair]

    D = len(training_set)
    D_pos = len(train_pos)
    D_neg = D - D_pos
    logprior = math.log(D_pos)  - math.log(D_neg)

    # For each word in the vocabulary get pos and neg probability
    for word in vocab:
        if (word,1.0) in ratings:
            freq_pos = ratings[(word,1.0)]
        else:
            freq_pos = 0.0
        if (word,0.0) in ratings:
            freq_neg = ratings[(word,0.0)]
        else :
            freq_neg = 0.0

        p_w_pos = (freq_pos +1 ) / (N_pos + V)
        p_w_neg = (freq_neg +1 ) / (N_neg + V)
        loglikelihood[word] = math.log(p_w_pos)  - math.log(p_w_neg)

    return logprior, loglikelihood

def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    accuracy = 0  
    y_hats = []
    for tweet in test_x:
        if predict_review(tweet, logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0
        y_hats.append(y_hat_i)
    error = mean([abs(x - y) for (x,y) in zip(y_hats, test_y)])

    accuracy = 1 - error
    return accuracy

def get_ratings(reviews, positive_reviews, negative_reviews):
    ratings = []
    
    for rev in reviews:
        if rev in positive_reviews:
            ratings.append(1)
        elif rev in negative_reviews:
            ratings.append(0)
    
    return ratings


def predict_review(review, logprior, loglikelihood):
    review_tokens = process_review(review)
    p = 0
    p += logprior
    for word in review_tokens:
        if word in loglikelihood:
            p += loglikelihood[word]
    return p




################

data = pd.read_csv('Reviews1.csv')
positive_reviews, negative_reviews = get_pos_neg( data ) 
# positive_reviews & negative_reviews = list of strings, seprated positive & negative reviews

words, data = preprocess(data)
# words = dict of unique words 
# data = processed tweet

## training set = 80% 
## testing set = 20%
## positives = 18540; negatives = 4101

train_pos = positive_reviews[:14832]
test_pos = positive_reviews[14832:]

train_neg = negative_reviews[:3281]
test_neg = negative_reviews[3281:]

training_set = train_pos + train_neg
test_set = test_pos + test_neg


ratings = make_count(words, data) 

# ratings = freqs of words

log_prior , log_likelihood = training_naive_bayes(ratings,training_set)

labels = get_ratings(test_set, positive_reviews, negative_reviews)
print( test_naive_bayes(test_set, labels , log_prior, log_likelihood) )

print(predict_review("she smiled and was happy" , log_prior , log_likelihood ))