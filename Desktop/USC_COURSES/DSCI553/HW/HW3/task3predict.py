from __future__ import division
from pyspark import SparkContext
from statistics import mean
import sys
import time
from math import log2, sqrt
import json
import os

from utils import itemBasedPrediction, userBasedPrediction

# Parameters
train_path = sys.argv[1]
test_path = sys.argv[2]
model_path = sys.argv[3]
output_path = sys.argv[4]
cf_type = sys.argv[5]

# Write to the output_path in JSON format.
def generatorJSON(partition):
    with open(output_path, 'a+') as f:
        for row in iter(partition):
            f.write(json.dumps(row) + '\n')


# Driver Code
sc = SparkContext('local[*]', 'task3predict')
start = time.time()

if cf_type == 'item_based':
    test = sc.textFile(test_path).map(json.loads).map(lambda x: (x['user_id'], x['business_id']))
    model = dict(sc.textFile(model_path).map(json.loads).map(lambda x: ((x['b1'], x['b2']), x['sim'])).collect())
    train = sc.textFile(train_path).map(json.loads).map(lambda x: (x['user_id'], (x['business_id'],x['stars'])))\
        .groupByKey().mapValues(dict)
    result = test.join(train).map(lambda x: itemBasedPrediction(x, 5, model), preservesPartitioning=True)\
        .filter(lambda x: x['stars'] > 0)
else:
    test = sc.textFile(test_path).map(json.loads).map(lambda x: (x['business_id'], x['user_id']))
    model = dict(sc.textFile(model_path).map(json.loads).map(lambda x: ((x['u1'], x['u2']), x['sim'])).collect())
    
    train = sc.textFile(train_path).map(json.loads).map(lambda x: ((x['business_id'],x['user_id']),
                                                        (x['stars'], x['text']))).distinct().mapValues(lambda x: x[0])\
        .aggregateByKey((0,0), lambda a,b: (a[0] + b, a[1] + 1),lambda a,b: (a[0] + b[0], a[1] + b[1]))\
        .mapValues(lambda v: v[0]/v[1]).map(lambda x: (x[0][0], (x[0][1], x[1])))

    userset = dict(train.map(lambda x: (x[1][0], (x[0], x[1][1]))).groupByKey()\
        .mapValues(lambda x: dict(list(set(x)))).collect())

    train = train.groupByKey().mapValues(dict)
    
    result = test.join(train).map(lambda x: userBasedPrediction(x, 50, model, userset), preservesPartitioning=True)\
        .filter(lambda x: x['stars'] > 0)


result.foreachPartition(generatorJSON)       
end = time.time()
print(end-start)