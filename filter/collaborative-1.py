
import sys
from operator import add

import heapq 

from pyspark import SparkContext
# sc = SparkContext()

if __name__ == "__main__":
    #Program Name Train, Test, Output
        # train = sc.textFile('netflix_subset/Train*')
    
    sc = SparkContext()

    #train = sc.textFile('TrainingRatings.txt').sample(False, 0.01)
    #test = sc.textFile('TestingRatings.txt')
    train = sc.textFile(sys.argv[1])#.sample(False, 0.2)
    # use small sample first
    test = sc.textFile(sys.argv[2])
    k = 50 # number of similar users used to predict

    # apply the threshold (rating >= 3.0)
    itemUser = train.map(lambda line: line.split(',')).filter(lambda rcd: float(rcd[2]) >= 3.0).map(lambda rcd: (rcd[0],rcd[1])).persist()

    # count the number of ratings for each user
    userList = itemUser.map(lambda (m,u): (u,[m])).reduceByKey(lambda v1,v2: v1+v2).persist()
    userCount = userList.map(lambda (u,l): (u,(l,len(l))))

    # get the list of rated users for each movie
    movieList = itemUser.map(lambda (m,u): (m,[u])).reduceByKey(lambda v1,v2: v1+v2)

    # count the number of overlaps between a user in the test set and another user
    testUsers = test.map(lambda line: line.split(',')).map(lambda rcd: rcd[1]).distinct().map(lambda u: (u,1))
    testList = testUsers.join(userCount).map(lambda (u,(c,p)): (u,p))
    movieUser = testList.map(lambda (user,(list,count)): (list,(user,count))).flatMap(lambda (l,v): [(m,v) for m in l])
    cooccur = movieUser.join(movieList).flatMap(lambda (m,((u,c),l)): [(u2,(u,c)) for u2 in l])
    overlaps = cooccur.join(userCount).map(lambda (u2,((u1,c1),(l,c2))): ((u1,u2),(c1,c2,1))).reduceByKey(lambda (c1,c2,v1),(c3,c4,v2):(c1,c2,v1+v2))

    # calculate Jaccard similarity based on counts
    similarity = overlaps.map(lambda ((u1,u2),(c1,c2,o)): ((u1,u2),(float(o)/float(c1+c2-o)))).persist()

    # produce the top list for each test user
    # this time we use similarity >= 0.5 as the threshold instead of a fixed number
    topList = similarity.filter(lambda ((u1,u2),s): u1 != u2).map(lambda ((u1,u2),s): (u1,[(u2,s)])).reduceByKey(lambda v1,v2: v1+v2).map(lambda (u,l): (u,heapq.nlargest(k,l,key=lambda e:e[1])))

    #
    itemUserNew = train.map(lambda line: line.split(','))
    fullMovieList = itemUserNew.map(lambda (m,u,r): (m,[(u,r)])).reduceByKey(lambda v1,v2: v1+v2)
    predictTarget = test.map(lambda line: line.split(',')).join(fullMovieList).flatMap(lambda (m,(u,l)): [(m,u,p) for p in l]).map(lambda (m,u1,(u2,r)): ((u1,u2),(m,r)))
    simSet = topList.flatMap(lambda (u,l): [(u,p) for p in l]).map(lambda (u1,(u2,s)): ((u1,u2),s))
    predictResult = predictTarget.join(simSet).persist()

    #predictResult.saveAsTextFile(sys.argv[3])

    rdd = predictResult.map(lambda ((u1,u2),((m,r),s)): ((u1,m),(float(r)*s,s))).reduceByKey(lambda (wr1,s1),(wr2,s2): (wr1+wr2,s1+s2)).map(lambda ((u,m),(wr,s)): (m,u,wr/s))
    rdd.saveAsTextFile(sys.argv[3])
    
    sc.stop()




