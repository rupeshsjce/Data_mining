import os
import json
import codecs
import sys
import itertools

from pyspark import SparkContext
import collections

def with_spark(review, business, num):
    review = review.map(lambda x: (x['business_id'], x['stars']))
    business = business.map(lambda x: (x['business_id'], x['categories']))
    joined = review.join(business).filter(lambda x: (x[1][0] is not None) & (x[1][1] is not None))
    intermediate = joined.flatMap(lambda x: [(category.strip(),x[1][0]) for category in x[1][1].split(',')])
    # It calculates total stars and counts for each category.
    seqFunc = (lambda x, y: (x[0] + y[1], x[1] + 1))
    combFunc = (lambda x, y: (x[0] + y[0], x[1] + y[1]))
    result = intermediate.aggregateByKey((0,0), seqFunc, combFunc).map(lambda x: [x[0],x[1][0]/x[1][1]]).sortBy(lambda x: [-x[1],x[0]]).take(num)
    return result

def without_spark(review,business,num):
    category_stars = collections.defaultdict(list)
    business = list(set(map(lambda x: (x['business_id'], x['categories']),
                       filter(lambda x: x['categories'] is not None, business))))
    business_dict = dict(business)
    review = list(filter(lambda x: (x[0] is not None) & (x[1] is not None),
                         map(lambda x: (x['business_id'], x['stars']), review)))
    intermediate = map(lambda x: (x[0], x[1], business_dict[x[0]]), filter(lambda x: x[0] in business_dict.keys(), review))
    pre_result = list(itertools.chain.from_iterable(list(map(lambda x:[(i.strip(), x[1]) for i in x[-1].split(',')], intermediate))))
    
    for k, v in pre_result:
        category_stars[k].append(v)
    final_result = {k: sum(v)/len(v) for (k, v) in category_stars.items()}
    result = list(map(list, sorted(list(final_result.items()), key=lambda x: [-x[1], x[0]])[:num]))
    return result

# Arguments
review_input_path = sys.argv[1]
business_input_path = sys.argv[2]
output_path = sys.argv[3]
if_spark = sys.argv[4]
num = int(sys.argv[5])

# Driver Code
output = {}
if if_spark == "spark":
    sc=SparkContext('local[*]','task2')
    review = sc.textFile(review_input_path).map(json.loads).persist()
    business = sc.textFile(business_input_path).map(json.loads).persist()
    output['result'] = with_spark(review,business,num)
else:
    review = [json.loads(line) for line in codecs.open(review_input_path, 'r', 'utf-8-sig')]
    business = [json.loads(line) for line in codecs.open(business_input_path, 'r', 'utf-8-sig')]
    output['result'] = without_spark(review,business,num)

# write to the output file.   
json_out = json.dumps(output)
with open(output_path,"w") as f:
    f.write(json_out)
