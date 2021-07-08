from pyspark import SparkContext
import json
import sys

def default_partition(review,n):
    output = {}
    output['n_partitions'] = review.getNumPartitions()
    output['n_items'] = review.glom().map(len).collect()
    output['result'] = review.map(lambda x: (x['business_id'],1)).reduceByKey(lambda a,b:a+b)\
        .filter(lambda x: x[1]>n).map(lambda x: list(x)).collect()
    return output

# Key here is business_id
def custom(k):
    return hash(k)

def custom_partition(n_partitions,n):
    output = {}
    review = sc.textFile(input_path).map(json.loads)
    # partitionBy works on (key,value)
    review =  review.map(lambda x: (x['business_id'],1)).partitionBy(n_partitions, custom)
    output['n_partitions'] = review.getNumPartitions()
    output['n_items'] = review.glom().map(len).collect()
    output['result'] = review.reduceByKey(lambda a,b : a+b).filter(lambda x: x[1] > n).map(lambda x: list(x)).collect()
    return output

# Argument
input_path = sys.argv[1]
output_path = sys.argv[2]
partition_type = sys.argv[3]
n_partitions = int(sys.argv[4])
n = int(sys.argv[5])

# Driver Code
sc = SparkContext('local[*]', 'task3')
if partition_type == 'default':
    review = sc.textFile(input_path).map(json.loads).persist()
    output = default_partition(review, n)
else:
    output = custom_partition(n_partitions,n)

# Output
json_output = json.dumps(output)
with open(output_path,"w") as f:
    f.write(json_output)
