# coding: utf-8
'''
http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

1.uses the probabilities of each attribute belonging to 
	each class to make a prediction
2. It simplifies the calculation of probabilities by assuming that the 
	probability of each attribute belonging to a given class value is 
	independentof all other attributes. This is a strong assumption but 
	results in a fast and effective mothod. and still gives robust results
3. The probability of a class value given a value of an attribute is called the
	conditional probability.By multiplying the conditional probabilities together
	for each attribute for a given class value, we have a probability of a 
	data instance belonging to that class.


scratch起跑线，intuitive直观的，simplify简化，comprised由..组成，observations观测 观察，
summarize概述 总结，mock假的 模拟的 伪造的，deviation偏差 误差，exponent指数，

'''

import csv, random, math

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

# These are required when making predictions to calculate the probability of 
# specific attribute values belonging to each class value.

# We can break the preparation of this summary data down into the following 
# sub-tasks:

# Separate Data By Class
# Calculate Mean
# Calculate Standard Deviation
# Summarize Dataset
# Summarize Attributes By Class
def separateByClas(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

'''
The standard deviation is calculated as the square root of the variance. 
The variance is calculated as the average of the squared differences for 
each attribute value from the mean. Note we are using the N-1 method, 
which subtracts 1 from the number of attribute values when calculating 
the variance.
'''
def mean(numbers):
	return sum(numbers) / float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg, 2) for x in numbers]) / float(len(numbers) - 1)
	return math.sqrt(variance)

def summarize(dataset):     # awesome zip(*): translate rows to colums
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClas(dataset)
	summaries = {}
	for classValue, instances, in separated.iteritems():
		summaries[classValue] = summarize(instances)  # 0/1: (mean, stdev fo each column)
	return summaries
'''
We can divide this part into the following tasks:

Calculate Gaussian Probability Density Function
Calculate Class Probabilities
Make a Prediction
Estimate Accuracy
'''	
# the key calculation-- Probability Density Function of Gaussian Distribution
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean, 2) / (2 * math.pow(stdev, 2))))
	return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def calculateClassProbability(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbability(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct / float(len(testSet))) * 100.0

def main():
	filename = 'data/pima-indians-diabetes.data.csv'
	splitRatio = 0.67
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print('Split {0} rows into train = {1} and test={2} rows').format(len(dataset), 
		len(trainingSet), len(testSet))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%').format(accuracy)

main()

