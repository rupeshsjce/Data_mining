from __future__ import division
from pyspark import SparkContext
from itertools import combinations, islice, product
from statistics import mean
from collections import Counter, defaultdict
from math import log2, sqrt
import json
import random
import sys
import time

from utils import *

# Parameters
train_path = sys.argv[1]
model_path = sys.argv[2]
cf_type =    sys.argv[3]


def generatorJSON(partition):
    # @partition: iterator-object of partition
    with open(model_path, 'a+') as f:
        for row in iter(partition):
            f.write(json.dumps(row) + '\n')

# Driver code
sc = SparkContext('local[*]', 'task3train')
start = time.time()

review = sc.textFile(train_path).map(json.loads)
if cf_type == 'item_based':
    result = review.map(lambda x: (x['user_id'], (x['business_id'], x['stars']))).groupByKey() \
        .mapValues(lambda x: dict(list(x))).flatMap(lambda x: item_TransformPairs(x[1])).groupByKey().mapValues(list) \
        .filter(lambda x: len(x[1]) >= 3) \
        .map(lambda x: item_PearsonSimilarity(x), preservesPartitioning=True) \
        .filter(lambda x: x['sim'] > 0)
else: #'User-based'
    businesses = list(set(review.map(lambda x: x['business_id']).collect()))
    biz_to_id = dict(zip(businesses, range(len(businesses))))
    result = review.map(lambda x: (x['user_id'], (biz_to_id[x['business_id']], x['stars'])), preservesPartitioning=True) \
        .groupByKey() \
        .mapValues(lambda x: dict(list(set(x)))).filter(lambda x: len(x[1]) >= 3)
    userset = dict(result.collect())
    result = result.flatMap(lambda x: user_MinhashBanding(x, 30, 1)).groupByKey().mapValues(lambda x: sorted(set(x))) \
        .filter(lambda x: len(x[1]) > 1).flatMap(lambda x: combinations(x[1], 2)).distinct() \
        .map(lambda x: (x, user_JaccardSimilarity(x, userset)), preservesPartitioning=True) \
        .filter(lambda x: (x[1][0] >= 0.01) and (len(x[1][1]) >= 3)) \
        .map(lambda x: user_PearsonSimilarity(x, userset), preservesPartitioning=True) \
        .filter(lambda x: x['sim'] > 0)



result.foreachPartition(generatorJSON)
end = time.time()
print(end - start)


