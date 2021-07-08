from pyspark import SparkContext
import csv
import json

sc = SparkContext('local[*]', 'task1')

review = sc.textFile('review-002.json').map(json.loads).map(lambda x: (x['business_id'], x['user_id']))
business = sc.textFile('business.json').map(json.loads).filter(lambda x: x['state'] == 'NV').map(lambda x: (x['business_id'], x['state']))

joined = review.join(business).map(lambda x: (x[1][0], x[0]))

rb = joined.collect()
with open('user_business.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(['user_id', 'business_id'])
    for i in rb:
        write.writerow(i)
