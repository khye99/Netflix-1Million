import sys
from pyspark import SparkContext
import heapq
import math

sc = SparkContext()

train = sc.textFile(sys.argv[1]).map(lambda line: line.split(',')).persist()
test = sc.textFile(sys.argv[2]).map(lambda line: line.split(',')).persist()
k = 500 # number of similar users used to predict

# apply the threshold (rating >= 3.0)
itemUser = train.filter(lambda (m,u,r): float(r) >= 3.0).map(lambda (m,u,r): (m,u))

# count the number of ratings for each user
userList = itemUser.map(lambda (m,u): (u,[m])).reduceByKey(lambda v1,v2: v1+v2)
userCount = userList.map(lambda (u,l): (u,(l,len(l))))

# get the list of rated users for each movie
movieList = train.map(lambda (m,u,r): (m,[(u,r)])).reduceByKey(lambda v1,v2: v1+v2)

# produce a list of movies for each user in the test set
testUsers = test.map(lambda (m,u,r): u).distinct().map(lambda u: (u,1))
testList = testUsers.join(userCount).map(lambda (u,(c,p)): (u,p))

movieUser = testList.map(lambda (u,(l,c)): (l,(u,c))).flatMap(lambda (l,v): [(m,v) for m in l])

# count the number of overlaps between a user in the test set and another user in the training set
cooccur = movieUser.join(movieList).flatMap(lambda (m,((u,c),l)): [(u2,(u,c)) for (u2,r) in l])
overlaps = cooccur.join(userCount).map(lambda (u2,((u1,c1),(l,c2))): ((u1,u2),(c1,c2,1))).reduceByKey(lambda (c1,c2,v1),(c3,c4,v2):(c1,c2,v1+v2))

# calculate Jaccard similarity based on counts
similarity = overlaps.map(lambda ((u1,u2),(c1,c2,o)): ((u1,u2),(float(o)/float(c1+c2-o))))

# produce the top k list for each test user
topList = similarity.filter(lambda ((u1,u2),s): u1 != u2).map(lambda ((u1,u2),s): (u1,[(u2,s)])).reduceByKey(lambda v1,v2: v1+v2).map(lambda (u,l): (u,heapq.nlargest(k,l,key=lambda e:e[1])))
simSet = topList.flatMap(lambda (u,l): [(u,p) for p in l]).map(lambda (u1,(u2,s)): ((u1,u2),s))

# get all (user, movie) pair we need to predict, then use weighted average to predict
predictTarget = test.join(movieList).flatMap(lambda (m,(u,l)): [(m,u,p) for p in l]).map(lambda (m,u1,(u2,r)): ((u1,u2),(m,r)))
predictResult = predictTarget.join(simSet).map(lambda ((u1,u2),((m,r),s)): ((u1,m),(float(r)*s,s))).reduceByKey(lambda (wr1,s1),(wr2,s2): (wr1+wr2,s1+s2)).map(lambda ((u,m),(wr,s)): ((m,u),wr/s))
    
predictResult.saveAsTextFile(sys.argv[3])

# compare our prediction with the given ratings
comparasion = test.map(lambda (m,u,r): ((m,u),float(r))).join(predictResult).map(lambda ((m,u),(r,pr)): ((m,u),r-pr)).values().persist()
count = comparasion.count()
MAE = comparasion.map(lambda v: abs(v)).sum() / count # mean absolute error
sc.parallelize([MAE]).saveAsTextFile(sys.argv[4])
RMSE = math.sqrt(comparasion.map(lambda v: v*v).sum() / count) # root mean squared error
sc.parallelize([RMSE]).saveAsTextFile(sys.argv[5])

sc.stop()