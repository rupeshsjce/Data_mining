from pyspark import SparkContext
import json
import sys
import time
import os
from operator import add
#%%
input_path = sys.argv[1]
output_path = sys.argv[2]
partition_type = sys.argv[3]
n_partitions = int(sys.argv[4])
n = int(sys.argv[5])
#%%
def default_partition(review,n):
    output = {}
    output['n_partitions'] = review.getNumPartitions()
    output['n_items'] = review.glom().map(len).collect()
    output['result'] = review.map(lambda x: (x['business_id'],1)).reduceByKey(lambda a,b:a+b)\
        .filter(lambda x: x[1]>n).map(lambda x: list(x)).collect()
    return output
#%%
def custom(biz_id):
    return hash(biz_id)
#%%
# def count_reviews(iterator):
#     for y in set(iterator):
#         yield sum(y[1])
    #return [y[0] for y in set(iterator)]
#%%
def custom_partition(n_partitions,n):
    output = {}
    review = sc.textFile(input_path).map(json.loads)\
        .map(lambda x: (x['business_id'],1)).partitionBy(n_partitions, custom)
    output['n_partitions'] = review.getNumPartitions()
    output['n_items'] = review.glom().map(len).collect()
    output['result'] = review.reduceByKey(add).filter(lambda x: x[1]>n).map(lambda x: list(x)).collect()#mapPartitions(count_reviews)
    return output
#%%
sc = SparkContext('local[*]', 'task3')

if partition_type == 'default':
    review = sc.textFile(input_path).map(json.loads).persist()
    output = default_partition(review, n)
else:
    output = custom_partition(n_partitions,n)
#%%
# start = time.time()
# output = default_partition(review,300)
# end = time.time()
# t = end-start
#%%
json_output = json.dumps(output)
with open(output_path,"w") as f:
    f.write(json_output)
#%%
# from pyspark import SparkFiles
# path = os.path.join('data/', "test.txt")
# with open(path, "w") as testFile:
#    _ = testFile.write("100")
# sc.addFile(path)
# def func(iterator):
#    with open(SparkFiles.get("test.txt")) as testFile:
#        fileVal = int(testFile.readline())
#        return [x * fileVal for x in iterator]
# hi = b.mapPartitions(func).collect()
