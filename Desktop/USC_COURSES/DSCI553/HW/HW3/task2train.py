# task2train.py: Content-based Recommendation System.
# 1. Build item/business profile. 2. Build user profile 3. find the cosine similarity between them.
'''
During the training process, you will construct the business and user profiles as follows:

a. Concatenating all reviews for a business as one document and parsing the document, such as
removing the punctuations, numbers, and stopwords. Also, you can remove extremely rare words to
reduce the vocabulary size. Rare words could be the ones whose frequency is less than 0.0001% of
the total number of words.

b. Measuring word importance using TF-IDF, i.e., term frequency multiply inverse doc frequency

c. Using top 200 words with the highest TF-IDF scores to describe the document

d. Creating a Boolean vector with these significant words as the business profile

e. Creating a Boolean vector for representing the user profile by aggregating the profiles of the items
that the user has reviewed

During the prediction process, you will estimate if a user would prefer to review a business by computing
the cosine distance between the profile vectors. The (user, business) pair is valid if their cosine
similarity is >= 0.01. You should only output these valid pairs.
'''

from __future__ import division
from pyspark import SparkContext
from itertools import combinations, islice
import sys
import time
import json
import random
from math import log2
import re
import json
import pickle
import collections

# Params
train_file = sys.argv[1]
model_file = sys.argv[2]
stopwords_path = sys.argv[3]

# process the stopwords.
def process_stopwords():
    stp = list(open(stopwords_path,'r'))
    stopwords = [i.replace('\n','') for i in stp]
    stopwords.extend(['',' ', "it\'s", "i\'m", "", '&', '$'])
    return stopwords


def text_preproc(text, stopwords):
    text = text.lower()
    text = text.replace("(", " ").replace("["," ").replace(","," ").replace("."," ").replace("!"," ").replace("?"," ")\
        .replace(":"," ").replace(";"," ").replace("]"," ").replace(")"," ").replace('\n','').replace("\\", " ")
    text = text.replace("can\'t", "can not").replace("won\'t", "will not")
    text = text.replace("\'m", " ").replace("\'re", " ").replace("\'ve", " ").replace("\'s", " ").replace("\'ll"," ").replace("\'d", " ")
    text = re.sub(r"\w+(n\'t).", "", text)
    text = text.strip().split(" ")
    text = re.sub(r"(\$*\d+)", "" ,text)
    text = re.sub(r"(\d+.)", " ", text)

    text = [i.strip() for i in text if i not in stopwords]
    return text

def TF(text, v):
    text = [v[w] for w in text if w in v.keys()]
    freq = collections.Counter(text)
    max_freq = freq.most_common(1)[0][1]
    TF = [(word, (freq/max_freq)) for word, freq in freq.items()]
    return TF

# content-based recommendation system.
def task2(review, stopwords):
    # STEP: A.1 = text preprocessing during combining all the 'text' for a particular 'business_id'.
    business_text  = review.repartition(20).map(lambda x: (x['business_id'],x['text'])).groupByKey().\
        mapValues(lambda x: text_preproc(' '.join(list(set(x))), stopwords))
    # Total number of documents/businesses
    num_business = business_text.count()    

    # STEP: A.2 = Remove extremenly rare words.
    vocab = business_text.flatMap(lambda x: x[1]).collect()
    rare_threshold = 0.0001*len(vocab)/100
    vocab = collections.Counter(vocab)
    vocab = [k for k,v in vocab.items() if v >= rare_threshold]
    vocab = dict([(i1,i0) for i0,i1 in enumerate(vocab)])

    # STEP: B/C/D = create business profiles using 200 significant words with highest TF.IDF
    tfidf = business_text.mapValues(lambda text: TF(text, vocab)).flatMap(lambda x: [(word, (x[0], tf)) for word, tf in x[1]])\
        .groupByKey().mapValues(list).mapValues(lambda x: [(business_id, tf*log2(num_business/len(x))) for business_id,tf in x])
    business_profile = tfidf.flatMap(lambda x: ([(business_id, (x[0], tfidf_score)) for business_id, tfidf_score in x[1]])).groupByKey()\
        .mapValues(lambda x: list(dict(sorted(x, reverse=True, key=lambda x: x[1])[:200]).keys()))


    # STEP: E = create user profiles by aggregating the profiles of the items that the user has reviewed.
    user = review.map(lambda x: (x['business_id'], x['user_id']))
    user_profile = business_profile.join(user).map(lambda x: (x[1][1], x[1][0])).groupByKey()\
        .mapValues(lambda x: list(set([item for sublist in x for item in sublist])))
    
    result = [dict(user_profile.collect()), dict(business_profile.collect())]
    return result

sc = SparkContext('local[*]', 'task2')
start = time.time()
stopwords = process_stopwords()

review = sc.textFile(train_file).map(json.loads)

result = task2(review, stopwords)
pickle.dump(result, open(model_file, "wb"))

end = time.time()
print(end-start)

