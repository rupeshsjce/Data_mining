import sys
import json
from pyspark import SparkContext
import datetime

def cleanup(text):
    text = text.lower().replace("(", "").replace("[","").replace(",","").replace(".","").replace("!","").replace("?","")\
        .replace(":","").replace(";","").replace("]","").replace(")","").strip().split(" ")
    return text

def task1(review,stopwords,year,m,n):
    result = {}
    result['A'] = review.map(lambda x: x['review_id']).count()
    result['B'] = review.filter(lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d %H:%M:%S').year == year).count()
    result['C'] = review.map(lambda x: x['user_id']).distinct().count()
    result['D'] = review.map(lambda x: (x['user_id'], 1)).reduceByKey(lambda a,b: a+b)\
        .map(lambda x: [x[0],x[1]]).sortBy(lambda x: x[1], ascending=False).take(m)
    stopwords.extend(['',' '])
    result['E'] = review.flatMap(lambda x: cleanup(x['text'])).filter(lambda x: x.strip() not in stopwords)\
        .map(lambda x: (x,1)).reduceByKey(lambda a, b: a + b).sortBy(lambda x: x[1], ascending=False).map(lambda x: x[0]).take(n)
    return result


# Arguments
review_input_path = sys.argv[1]
output_path = sys.argv[2]
stopwords_path = sys.argv[3]
year = int(sys.argv[4])
m = int(sys.argv[5])
n = int(sys.argv[6])

# Driver Program
sc=SparkContext('local[*]','task1')
review = sc.textFile(review_input_path).map(json.loads).persist()
stp = list(open(stopwords_path,'r'))
stopwords = [i.replace('\n','') for i in stp]
output = task1(review,stopwords,year,m,n)

# Write the output to the output_path.
json_output = json.dumps(output)
with open(output_path,"w") as f:
    f.write(json_output)
