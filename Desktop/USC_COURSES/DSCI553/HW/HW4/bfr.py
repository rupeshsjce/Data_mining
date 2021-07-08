from collections import defaultdict
import time
import sys
from statistics import mean
from sys import getsizeof
import re
from os.path import isfile, join
import os
import random
from itertools import chain, combinations
import math, csv, json, copy

#%%

def euclideanDistance(P1, P2):
    return math.sqrt(sum((x1-x2)**2 for x1,x2 in zip(P1,P2)))

def randomSampling(data, n):
    random.seed(49)
    sample = []
    sample.extend(random.sample(data.keys(), n))
    return sample

def assignCluster(cluster_centers, data):
    assigned_cluster = {point: min((euclideanDistance(data[point],cluster_centers[C]),index) for index,C in
                               enumerate(cluster_centers))[1] for point in data}
    return assigned_cluster

def init_kmeans_centers(n_cluster, data):
    centers = randomSampling(data, 1)
    i = 1
    while(i < n_cluster):
        scores = {p: min([euclideanDistance(data[p],data[C]) for C in centers]) for p in data}
        centers.append(max(scores.items(), key=lambda x:x[1])[0])
        i += 1
    return centers

def centroid(clusterPoints, data):
    centroid = [(index,[mean(d) for d in zip(*(data[i] for i in clusterPoints[key]))]) for index,key in enumerate(clusterPoints)]
    return dict(sorted(centroid))
#%%
def kmeans(data, n_cluster):
    cluster_centers = dict(sorted((index,data[center]) for index,center in enumerate(init_kmeans_centers(n_cluster,data))))
    assigned_cluster = assignCluster(cluster_centers, data)
    clusterPoints = defaultdict(list); [clusterPoints[v].append(k) for k, v in assigned_cluster.items()]
    target = sum(sum(euclideanDistance(data[i], cluster_centers[C]) for i in points) for C, points in clusterPoints.items())
    
    k = target + 1
    iter, maxIter = 1, 200
    while target < k and iter <= maxIter:
        k = target
        cluster_centers = centroid(clusterPoints, data)
        assigned_cluster = assignCluster(cluster_centers, data)
        clusterPoints = defaultdict(list); [clusterPoints[v].append(k) for k, v in assigned_cluster.items()]
        target = sum(sum(euclideanDistance(data[i], cluster_centers[C]) for i in points) for C, points in clusterPoints.items())
        iter += 1
    return clusterPoints
#%%
def generateStats(DS, orig_data):
    return {c:{'N': len(DS[c]), 'SUM': [sum(i) for i in zip(*(orig_data[pt] for pt in DS[c]))],
               'SUMSQ': [sum(i**2 for i in dim) for dim in zip(*(orig_data[pt] for pt in DS[c]))]} for c in DS}

def updateStats(oldDS_stats, newpts, orig_data):
    for C in newpts:
        if C in oldDS_stats:
            oldDS_stats[C]['N'] += len(newpts[C])
            oldDS_stats[C]['SUM'] = [sum(i) for i in zip(oldDS_stats[C]['SUM'], *(orig_data[pt] for pt in newpts[C]))]
            newsq = {pt: [i**2 for i in orig_data[pt]] for pt in newpts[C]}
            oldDS_stats[C]['SUMSQ'] = [sum(i) for i in zip(oldDS_stats[C]['SUMSQ'], *(newsq[pt] for pt in newpts[C]))]
    return oldDS_stats

def Average_Variance(name, cluster_stats):
    n = cluster_stats['N']
    if name == 'average':
        return [i/n for i in cluster_stats['SUM']]
    return [((a/n) - (b/n)**2) for a,b in zip(cluster_stats['SUMSQ'], cluster_stats['SUM'])]

def mahaDistance(P_dim, cluster_stats):
    average = Average_Variance('average', cluster_stats)
    variance = Average_Variance('variance', cluster_stats)
    MD = math.sqrt(sum((a-b)**2/c for a,b,c in zip(P_dim, average, variance)))
    return MD

def assignClusterBFR(orig_data, leftoverData, DS_stats, d):
    DS = defaultdict(list)
    if bool(DS_stats):
        minD = {pt: min((mahaDistance(orig_data[pt], DS_stats[C]),C) for C in DS_stats) for pt in leftoverData}
        [(DS[minD[i][1]].append(i), leftoverData.pop(i)) for i in minD if minD[i][0] <= 2*math.sqrt(d)]
        DS_stats = updateStats(DS_stats, DS, orig_data)
    res = leftoverData, DS, DS_stats
    return res

def mergeCS(CS, CS_stats):
    if CS:
        CS = dict(sorted(CS.items()))
        centroid = {C: Average_Variance('average', CS_stats[C]) for C in CS_stats}
        for a, b in combinations(CS,2):
            MD_ab, MD_ba = mahaDistance(centroid[a], CS_stats[b]), mahaDistance(centroid[b], CS_stats[a])
            if MD_ab <= 2*math.sqrt(d) or MD_ba <= 2*math.sqrt(d):
                CS[a].extend(CS[b]); CS[b].extend(CS[a])
                CS_stats[a]['N'] = CS_stats[a]['N']+CS_stats[b]['N']; CS_stats[b]['N'] = CS_stats[a]['N']+CS_stats[b]['N']
                CS_stats[a]['SUM'] = [sum(i) for i in zip(CS_stats[a]['SUM'], CS_stats[b]['SUM'])]
                CS_stats[b]['SUM'] = [sum(i) for i in zip(CS_stats[a]['SUM'], CS_stats[b]['SUM'])]
                CS_stats[a]['SUMSQ'] = [sum(i) for i in zip(CS_stats[a]['SUMSQ'], CS_stats[b]['SUMSQ'])]
                CS_stats[b]['SUMSQ'] = [sum(i) for i in zip(CS_stats[a]['SUMSQ'], CS_stats[b]['SUMSQ'])]
        temp = []; merged = [i for i in zip(CS.values(), CS_stats.values())]
        [temp.append(i) for i in merged if i not in temp]
        CS, CS_stats = [{k: cluster[k] for k in range(len(cluster))} for cluster in zip(*temp)]
    return CS, CS_stats

def mergeCSRS_DS(assignedCluster, orig_data, DS_stats, CS, RS):
    DS_centroid = {C: Average_Variance('average', DS_stats[C]) for C in DS_stats}
    minD = {pt: min((euclideanDistance(RS[pt], DS_centroid[C]),C) for C in DS_centroid) for pt in RS}
    DS = defaultdict(list); [DS[minD[i][1]].append(i) for i in minD]
    minD = {pt: min((euclideanDistance(orig_data[pt], DS_centroid[C]),C) for C in DS_centroid) for
            pt in chain.from_iterable(CS.values())}
    [DS[minD[i][1]].append(i) for i in minD]
    assignedCluster.update({pt: C for C in DS for pt in DS[C]})
    del orig_data, DS_stats, CS, RS
    return assignedCluster

def reorder(CS):
    val = list(CS.values())
    return {i:val[i] for i in range(len(val))}
#%%
def initBFR(sample, n_cluster):
    init_cluster = kmeans(sample, 5*n_cluster)
    RS = []; [RS.extend(init_cluster.pop(i)) for i in range(len(init_cluster)) if len(init_cluster[i]) < 20]
    DS = kmeans({i: sample[i] for i in list(chain.from_iterable(init_cluster.values()))}, n_cluster)
    DS_stats = generateStats(DS, orig_data)
    CS = kmeans({i: sample[i] for i in RS}, 5*n_cluster)
    RS = []; [RS.extend(CS.pop(i)) for i in range(len(CS)) if len(CS[i]) == 1]
    CS = reorder(CS); RS = {i: sample[i] for i in RS}
    CS_stats = generateStats(CS, orig_data)
    assignedCluster = {pt: C for C in DS for pt in DS[C]}
    del DS, CS, sample, init_cluster
    return assignedCluster, RS, DS_stats, CS_stats

def BFR(lastornot, d, orig_data, leftoverData, assignedCluster, RS, DS_stats, CS_stats):
    # Get the DS and DS STATS
    leftoverData, DS, DS_stats = assignClusterBFR(orig_data, leftoverData, DS_stats, d)
    # Get the CS and CS STATS
    leftoverData, CS, CS_stats = assignClusterBFR(orig_data, leftoverData, CS_stats, d)
    # Update the RS
    RS.update(leftoverData); RS_all = copy.copy(RS)
    # discard the old RS data.
    del leftoverData

    addCS = kmeans(RS, 5*n_cluster)
    CS = dict(CS); track = [i for i in addCS if len(addCS[i]) == 1]
    RS = {i: RS[i] for i in list(chain.from_iterable(addCS.pop(i) for i in track))}
    addCS = reorder(addCS); addCS = dict((len(CS)+a,b) for a,b in addCS.items())
    CS.update(addCS)
    CS_stats.update(generateStats(addCS, RS_all))
    CS, CS_stats = mergeCS(CS, CS_stats) # Merge CS clusters
    assignedCluster.update({pt: C for C in DS for pt in DS[C]})
    if lastornot != 'last':
        del DS, CS, track, addCS
        return assignedCluster, RS, DS_stats, CS_stats
    else:
        del DS, track, addCS
        return assignedCluster, RS, CS, DS_stats, CS_stats


#%%
# Parameters
input_path = 'data/test3' #sys.argv[1]
n_cluster = 5 #int(sys.argv[2])
final_output =  'bfr1_output.json' #sys.argv[3]
inter_output =   'inter1_output.csv'#sys.argv[4]

#%%
# Driver Code
start = time.time()
files = [join(input_path,i) for i in os.listdir(input_path) if isfile(join(input_path, i))]
files = sorted(filter(lambda x: re.findall(r'.txt',x), files))
print(files)
header = ['round_id', 'nof_cluster_discard', 'nof_point_discard', 'nof_cluster_compression',
              'nof_point_compression', 'nof_point_retained']
with open(inter_output, 'a+', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

for iter in range(len(files)):
    data = [i.replace('\n', '').split(',') for i in open(files[iter], 'r').readlines()]
    data = {int(x[0]): [float(i) for i in x[1:]] for x in data}
    orig_data = copy.copy(data); d = len(list(data.values())[0])
    if iter == 0:
        if len(data) > 5000:
            sample = {i: data.pop(i) for i in randomSampling(data, 5000)}
            assignedCluster, RS, DS_stats, CS_stats = BFR('not last', d, orig_data, data, *initBFR(sample, n_cluster))
        else:
            assignedCluster, RS, DS_stats, CS_stats = initBFR(data)
    elif iter == len(files)-1:
        assignedCluster, RS, CS, DS_stats, CS_stats = BFR('last', d, orig_data, data, assignedCluster,
                                                          RS, DS_stats, CS_stats)
        assignedCluster = mergeCSRS_DS(assignedCluster, orig_data, DS_stats, CS, RS)
    else:
        assignedCluster, RS, DS_stats, CS_stats = BFR('not last',d, orig_data, data, assignedCluster, RS, DS_stats, CS_stats)

    intermediate = {'round_id':iter+1, 'nof_cluster_discard': len(DS_stats), 'nof_point_discard': sum(C['N'] for C in DS_stats.values()),
               'nof_cluster_compression': len(CS_stats), 'nof_point_compression': sum(C['N'] for C in CS_stats.values()),
               'nof_point_retained': len(RS)}
    with open(inter_output, 'a+', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(intermediate)
    print(iter, getsizeof(assignedCluster))

with open(final_output,'w') as f:
    f.write(json.dumps(assignedCluster))

end = time.time()
print(end-start)
