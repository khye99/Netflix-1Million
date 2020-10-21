from pyspark import SparkContext
sc = SparkContext()

print('Starting Script')
train = sc.textFile('netflix_subset/Train*')
test = sc.textFile('netflix_subset/Test*')

traint = train.map(lambda x: x.split(','))
testt  = test.map(lambda x: x.split(','))

testCount = test.count()
testUserCount = testt.map(lambda x: x[1]).distinct().count()
testItemCount = testt.map(lambda x: x[0]).distinct().count()
print('Number of ratings in test set ' + str(testCount))
print('Number of unique users in test set ' + str(testUserCount))
print('Number of unique items in test set ' + str(testItemCount))
print("")

itemToUserRdd = traint.map(lambda x: (x[0], x[1])).groupByKey().mapValues(list)
userToItemRdd = traint.map(lambda x: (x[1], x[0])).groupByKey().mapValues(list)

itu = {}; uti = {}

for i in itemToUserRdd.collect():
  itu[i[0]] = i[1]

for i in userToItemRdd.collect():
  uti[i[0]] = i[1]

testUserSample = testt.map(lambda x: x[1]).distinct().take(10) # take 10 users
testItemSample = testt.map(lambda x: x[0]).distinct().take(10) # take 10 items

itemCounter = 0
userCounter = 0

for user in testUserSample:
  count = 0
  for i in uti[user]:
    count += len(itu[i])
  itemCounter += float(count) / len(uti[user])

for item in testItemSample:
  count = 0
  for i in itu[item]:
    count += len(uti[i])
  userCounter += float(count) / len(itu[item])

print('Estimated average overlap of items for users ' + str(itemCounter / 10.0))
print('Estimated average overlap of users for items ' + str(userCounter / 10.0))
print("")

trainUserCount =  traint.map(lambda x: x[1]).distinct().count()
trainItemCount =  traint.map(lambda x: x[0]).distinct().count()

print('Percentage of average overlap of items for users ' + str(itemCounter * 10.0 / trainUserCount))
print('Percentage of average overlap of users for items ' + str(userCounter * 10.0 / trainItemCount))
print("")

print('Preparing Jaccard similarity preprocessing:')
Jtrain = traint.filter(lambda x: x[2] >= 3).map(lambda x: (x[0], x[1]))
Jtest = testt.filter(lambda x: x[2] >= 3).map(lambda x: (x[0], x[1]))
print('done')