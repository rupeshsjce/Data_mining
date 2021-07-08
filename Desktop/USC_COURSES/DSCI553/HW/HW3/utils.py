from __future__ import division
from pyspark import SparkContext
from itertools import combinations, islice, product
from statistics import mean
from collections import Counter, defaultdict
import sys
import time
from math import log2, sqrt
import json
import random


# Item-based CF functions
def item_TransformPairs(x):
    business_pairs = list(combinations(sorted(x), 2))
    ratings = [(x[i[0]], x[i[1]]) for i in business_pairs]
    return list(zip(business_pairs, ratings))

# Item-based Pearson Similarity.
def item_PearsonSimilarity(x):
    b1, b2 = list(zip(*x[1]))
    b1avg, b2avg = mean(b1), mean(b2)
    numerator = sum((i - b1avg) * (j - b2avg) for i, j in x[1])
    if numerator <= 0:
        sim = -1
    else:
        denominator = sqrt(sum((rt - b1avg) ** 2 for rt in b1)) * sqrt(sum((rt - b2avg) ** 2 for rt in b2))
        sim = numerator / denominator
    return {'b1': x[0][0], 'b2': x[0][1], 'sim': sim}

# User-based Pearson Similarity.
def user_PearsonSimilarity(x, userset):
    u1, u2 = userset[x[0][0]], userset[x[0][1]]
    co_rated_biz = set(x[1][1])
    u1pf, u2pf = dict([(i, u1[i]) for i in co_rated_biz]), dict([(i, u2[i]) for i in co_rated_biz])
    u1avg, u2avg = mean(u1pf.values()), mean(u2pf.values())
    numerator = sum((a - u1avg) * (b - u2avg) for a, b in zip(u1pf.values(), u2pf.values()))
    if numerator <= 0:
        sim = -1
    else:
        denominator = sqrt(sum((rt - u1avg) ** 2 for rt in u1pf)) * sqrt(sum((rt - u2avg) ** 2 for rt in u2pf))
        sim = numerator / denominator
    return {'u1': x[0][0], 'u2': x[0][1], 'sim': sim}


def user_MinhashBanding(x, bands, rows):
    sig = []
    N = bands * rows
    random.seed(2)
    for i in range(N):
        a, b, m = random.randint(0, 10000), random.randint(0, 10000), random.randint(20000, 30000)
        perm = [(a * i + b) % m for i in x[1]]
        sig.append(min(perm))
    n, k = iter(range(len(sig))), iter(sig)
    elems = [rows] * bands
    slices = [tuple([next(n), tuple(islice(k, i))]) for i in elems]
    grps = [(i, x[0]) for i in slices]
    return grps



def user_JaccardSimilarity(x, userset):
    C0 = set(userset[x[0]])
    C1 = set(userset[x[1]])
    return (len(C0.intersection(C1)) / len(C0.union(C1)), C0.intersection(C1))


# User-Based Collaborative Filtering Functions.
def userBasedPrediction(x, N, model, userset):
    target_user, biz_ratings = x[1][0], x[1][1]
    num_user_biz = len(biz_ratings)
    if num_user_biz == 0:  # new business : no ratings
        predicted = -1
    else:    # old business (rated by some users)
        in_pairs = [tuple([target_user, i]) for i in biz_ratings]
        co_rated = dict([(u2, set(userset[u1]).intersection(set(userset[u2]))) for u1, u2 in in_pairs])
        co_rated = dict([(a,b) for a,b in co_rated.items() if b != set()])
        if co_rated == {}:      # No co-rated users
            predicted = -1
        else:
            pairs = [tuple(sorted([target_user, user])) for user in co_rated.keys()]
            weights = [model[i] if i in model.keys() else 0 for i in pairs]
            if sum(abs(wt) for wt in weights) == 0:       # New user: no rating or too few co-rated businesses
                predicted = -1
            else:
                co_rated_avg = [mean(userset[user][biz] for biz in cor_biz) for user, cor_biz in co_rated.items()]
                biz_ratings = [biz_ratings[user] for user in co_rated.keys()]
                if num_user_biz <= N:
                    topN = zip(biz_ratings, co_rated_avg, weights)
                else:
                    topN = sorted(zip(biz_ratings, co_rated_avg, weights), key=(lambda x: x[2]), reverse=True)[:N]
                try:
                    target_user_avg = mean(userset[target_user].values())
                    numerator = sum(((a - b) * c) for a, b, c in topN)
                    denominator = sum(abs(c) for a, b, c in topN)
                    predicted = target_user_avg + numerator/denominator
                except:
                    predicted = -1
    return {'user_id': target_user, 'business_id': x[0], 'stars': predicted}



def itemBasedPrediction(x, N, model):
    target, ratings = x[1][0], x[1][1]
    num_biz_user = len(ratings)
    if num_biz_user != 0:
        pairs = [tuple(sorted([target, i])) for i in ratings]
        weights = [model[i] if i in model.keys() else 0 for i in pairs]
        if sum(weights) == 0:
            predicted = -1
        else:
            if num_biz_user <= N:
                topN = list(zip(pairs, weights, ratings.values()))
            else:
                topN = sorted(zip(pairs, weights, ratings.values()), key=(lambda x: x[1]), reverse=True)[:N]
            predicted = sum(b * c for a, b, c in topN) / sum(b for a, b, c in topN)
    else:
        predicted = -1
    return {'user_id': x[0], 'business_id': target, 'stars': predicted}
