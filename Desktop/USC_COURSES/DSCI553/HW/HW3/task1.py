# task1.py: Find Similar business pairs using Min Hashing + LSH 
from __future__ import division
from pyspark import SparkContext
import sys

from itertools import combinations, islice
import time
import json
import random

# Parameters
input_path = 'data/train_review.json' #sys.argv[1]
output_path = 'task1.res' #sys.argv[2]


def minhash(n,col):
    sig = []
    random.seed(2)
    for i in range(n):
        a,b,m = random.randint(0,10000), random.randint(0,10000), random.randint(20000,30000)
        perm = [(a * i + b) % m for i in col]
        sig.append(min(perm))
    return sig

def banding(x,bands,rows):
    n = iter(range(len(x)))
    k = iter(x)
    elems = [rows]*bands
    slices = [tuple([next(n),tuple(islice(k,i))]) for i in elems]
    return slices

def jackard_similarity(x, bizset):
    C0 = set(bizset[x[0]])
    C1 = set(bizset[x[1]])
    jaccard = len(C0.intersection(C1))/len(C0.union(C1))
    return jaccard

def task1(review):
    # Return a new RDD containing the distinct elements in this RDD.
    users = review.map(lambda x: x['user_id']).distinct().collect()

    d_users = dict(zip(users,range(len(users))))
    # Pass each value in the key-value pair RDD through a map function without changing the keys; 
    # this also retains the original RDD’s partitioning.
    review = review.repartition(28).map(lambda x: (x['business_id'], d_users[x['user_id']])).groupByKey().mapValues(lambda x: list(set(x))).persist()
    business_userids = dict(review.collect())

    lsh = review.map(lambda x: (x[0],minhash(50,x[1]))).map(lambda x: (x[0],banding(x[1],50,1))).flatMap(lambda x: [(i,x[0]) for i in x[1]])

    
    candidates = lsh.groupByKey().mapValues(lambda x: sorted(set(x))).filter(lambda x: len(x[1]) > 1)
    candidates = candidates.flatMap(lambda x: combinations(x[1],2)).distinct()


    sim = candidates.map(lambda x: (x,jackard_similarity(x, business_userids))).filter(lambda x: x[1] >= 0.05)
    sim = sim.sortBy(lambda x: (x[0][0],x[0][1])).map(lambda x: dict([('b1',x[0][0]),('b2',x[0][1]),('sim',x[1])]))
    return sim.collect()
    

# Driver code
sc = SparkContext('local[*]', 'task1')
sc.setLogLevel('OFF')
start =  time.time()
review = sc.textFile(input_path).map(json.loads)
output = task1(review)


#parsed = review.first()
#print(json.dumps(parsed, indent=4, sort_keys=True))
'''
{
    "business_id": "zK7sltLeRRioqYwgLiWUIA",
    "date": "2015-12-19 07:35:30",
    "review_id": "pxOrtki0sqXps5hSyLXKpA",
    "stars": 5.0,
    "text": "Second time I've been here. First time was whatever. This time it was actually good. Way better than inn n out. It's the same type of burger that's why I put it up against that. I love that you can get grilled jalape\u00f1os. Just wish they came on the burger and not on the side.",
    "user_id": "OLR4DvqFxCKLOEHqfAxpqQ"
}
'''


# write the output to the file
with open(output_path,'w') as f:
    for line in output:
        f.write(json.dumps(line)+'\n')
end = time.time()
print("Duration: ", end-start)        




