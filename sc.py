from nltk.sem.logic import TypeResolutionException
import numpy as np
import pandas as pd
from nltk.corpus import stopwords         
from nltk.stem import snowball      
from nltk.tokenize import RegexpTokenizer, word_tokenize
import matplotlib.pyplot as plt 
import re
import string
import math

def get_pos_neg(dataset):
    ## get all the positive and negative labelled reviews
    pos = [] 
    neg = []
    # pos_tweet = []
    # neg_tweet = []
    # tweet_classifier = []
    for index, revs in dataset.iterrows():
        if revs.Recommended == 1:
            pos.append( revs )
            # pos_tweet.append(revs.Text)
            # tweet_classifier.append(revs.Recommended)
        else:
            neg.append( revs) 
            # neg_tweet.append(revs.Text)
            # tweet_classifier.append(revs.Recommended)
    # positives = 18540; negatives = 4101
    return pos, neg 
    # return pos_tweet , neg_tweet , tweet_classifier


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

def process_tweet(tweet):
    ## takes tweets and returns processed tokens
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = snowball.SnowballStemmer('english')
    stop_words = stopwords.words('english')
    #1 tokenize 
    tweet_tokens = tokenizer.tokenize(tweet)
    #2 lowercase
    tweet_tokens = [w.lower() for w in tweet_tokens]
    #3 remove stopwords 
    tweet_tokens= [stemmer.stem(wrd) for wrd in tweet_tokens if not wrd in stop_words ]

    print(tweet_tokens)
    return tweet_tokens








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

process_tweet("i am a happy larki")


# data = pd.read_csv('Reviews1.csv')
# # print( type( data.head()) )
# words, data = preprocess(data)
# #here words is a dict of unique words and data is processed tweet
# # positive_reviews, negative_reviews , Review_classifier = get_pos_neg( data ) 
# positive_reviews, negative_reviews = get_pos_neg( data ) 
# #^ here positive and negative reviews aare just list of strings and Review classifier has coresponding recommendations
# print(len(positive_reviews), " " , len(negative_reviews))

# # training set = 80% 
# # testing set = 20%
# # positives = 18540; negatives = 4101

# train_pos = positive_reviews[:14832]
# test_pos = positive_reviews[14832:]

# train_neg = negative_reviews[:3281]
# test_neg = negative_reviews[3281:]

# training_set = train_pos + train_neg
# test_set = test_pos + test_neg

# # print (len(Review_classifier))

# ratings = make_count(words, data)  # equi to freqs 

# # print(ratings[('silki', 1)])   # If you search manually silky gives 126, here it gives 111??
# # print(training_set)
# # print(test_set)


# def training_naive_bayes(ratings, training_set):
#     loglikelihood = {}
#     logprior = 0

#     # calculate V, the number of unique words in the vocabulary
#     vocab = set([pair[0] for pair in ratings.keys()])
#     V = len(vocab)

#     # calculate N_pos and N_neg
#     N_pos = N_neg = 0
#     for pair in ratings.keys():
#         if pair[1] > 0:
#             N_pos += ratings[pair]
#         else:
#             N_neg += ratings[pair]

#     D = len(training_set)
#     D_pos = len(train_pos)
#     D_neg = D - D_pos
#     logprior = math.log(D_pos)  - math.log(D_neg)
#     # For each word in the vocabulary get pos and neg probabilitu
#     for word in vocab:
#         if (word,1.0) in ratings:
#             freq_pos = ratings[(word,1.0)]
#         else:
#             freq_pos = 0.0
#         if (word,0.0) in ratings:
#             freq_neg = ratings[(word,0.0)]
#         else :
#             freq_neg = 0.0

#         p_w_pos = (freq_pos +1 ) / (N_pos + V)
#         p_w_neg = (freq_neg +1 ) / (N_neg + V)
#         loglikelihood[word] = math.log(p_w_pos)  - math.log(p_w_neg)

#     return logprior, loglikelihood


# print("Yahan aagaya")
# # print(len(training_set) ,"    " , len(positive_reviews))
# # print(training_naive_bayes(ratings,training_set))


# def Prdeict_Tweet(tweet, logprior, loglikelihood):

#     Tweet_tokens = process_tweet(tweet)
#     p = 0
#     p += logprior
#     for word in Tweet_tokens:
#         if word in loglikelihood:
#             p += loglikelihood[word]
#     return p



