import itertools
from collections import Counter
import csv
import sys
from operator import add

from pyspark import SparkContext
import time
from itertools import combinations


def count_occurences(x, first_pass):
    d = Counter()
    for basket in x:
        for itemset in first_pass:
            if set(itemset).issubset(basket):
                d[itemset] += 1
    return list(d.items())


def candidates(cominations_length,last_frequent):
    prev = sorted(set(itertools.chain.from_iterable(last_frequent)))
    return combinations(prev, cominations_length)

def true_frequent(baskets, candidates, partition_thresh):
    d = Counter()
    for itemset in candidates:
        for basket in baskets:
            if set(itemset).issubset(basket):
                d[itemset] += 1
    true_freq = [k for k,v in d.items() if v >= partition_thresh]
    return true_freq

def apriori_algorithm(baskets,s,total_items):
    d_count = Counter()
    count = 0
    baskets = list(baskets)

    for basket in baskets:
        for item in basket:
            d_count[item] += 1
        count += 1 # increment the count of the basket    
    p = count/total_items # percentage of baskets in this parition wrt to total baskets
    
    all_possible_combinations = sorted([tuple([k]) for k, v in d_count.items() if v >= p*s]) 

    final_freq = []
    final_freq.extend(all_possible_combinations)
    cominations_length = 1
    more_combination_possibles = False
    if len(all_possible_combinations) >= 2:
        more_combination_possibles = True

    while more_combination_possibles:
        C = candidates(cominations_length + 1 ,all_possible_combinations)
        all_possible_combinations = sorted(true_frequent(baskets,C,p*s))
        final_freq.extend(all_possible_combinations)
        cominations_length += 1
        if len(all_possible_combinations) < 2:
            break


    return final_freq


def task1(rdd, s):
    task1_output= []
    # PHASE 1 OF THE SON ALGORITHM 
    total_items = rdd.count()
    pass1 = rdd.mapPartitions(lambda baskets: apriori_algorithm(baskets,s,total_items)).distinct()
    pass1_single = pass1.filter(lambda x: len(x)==1).sortBy(lambda x: x[0]).collect()
    pass1_multiple = pass1.filter(lambda x: len(x)>1).sortBy(lambda x: (len(x),x[0],x[1])).collect()

    # PHASE 2 OF THE SON ALGORITHM 
    temp = pass1.collect()
    pass2 = rdd.mapPartitions(lambda x: count_occurences(x,temp)).reduceByKey(add).filter(lambda x: x[1]>=s).map(lambda x: x[0])
    pass2_single = pass2.filter(lambda x: len(x)==1).sortBy(lambda x: x[0]).collect()
    pass2_multiple = pass2.filter(lambda x: len(x)>1).sortBy(lambda x: (len(x), x[0], x[1])).collect()
    task1_output.extend([pass1_single, pass1_multiple, pass2_single, pass2_multiple])
    return task1_output



# Parameters
case = int( sys.argv[1])
s = int(sys.argv[2])
input_path = sys.argv[3]
output_path = sys.argv[4]

# Driver Program
sc = SparkContext('local[*]', 'task1')
sc.setSystemProperty('spark.driver.memory', '4g')
sc.setSystemProperty('spark.executor.memory', '4g')
sc.setLogLevel("OFF")
start = time.time()
rdd = sc.textFile(input_path).mapPartitions(lambda x: csv.reader(x))
header = rdd.first() # remove the header.
rdd = rdd.filter(lambda x: x != header) # rdd without the header.

if case == 1:
    rdd = rdd.groupByKey().mapValues(lambda x: list(set(x))).map(lambda x: x[1])
if case == 2 :
    rdd = rdd.map(lambda x: [x[1],x[0]]).groupByKey().mapValues(lambda x: list(set(x))).map(lambda x: x[1])

# Perform SON Algorithm and return the output.
task1_output = task1(rdd, s)
pass1_single = task1_output[0]
pass1_multiple =  task1_output[1]
pass2_single = task1_output[2]
pass2_multiple = task1_output[3]

# Write Output
with open(output_path,"w") as f:
    f.writelines("Candidates:")
    xeleton = len(pass1_multiple[-1])+1
    for i in range(1,xeleton):
        if i==1:
            out = "\n"+str(pass1_single).replace(',)', ')').replace('[','').replace(']','').replace(', ',',')+"\n"
            f.writelines(out)
        if i>1:
            out = sorted(list(filter(lambda x: len(x)==i, pass1_multiple)))
            out = "\n"+str(out).replace(', (', ',(').replace('[','').replace(']','')+"\n"
            f.writelines(out)
    f.writelines("\nFrequent Itemsets:")
    xeleton = len(pass2_multiple[-1])+1
    for i in range(1,xeleton):
        if i==1:
            out = "\n"+str(pass2_single).replace(',)', ')').replace('[','').replace(']','').replace(', ',',')+"\n"
            f.writelines(out)
        if i>1:
            out = sorted(list(filter(lambda x: len(x)==i, pass2_multiple)))
            out = "\n"+str(out).replace(', (', ',(').replace('[','').replace(']','')+"\n"
            f.writelines(out)
f.close()
end = time.time()
diff = round(end-start,2)
print("Duration:", diff)