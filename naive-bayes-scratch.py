'''
This python code is re-modeled for Python3, based on 
http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
'''

import csv
import random
import math
import sys

# Reading .csv & performing clean-up
def loadCsv(filename):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

#splitting into train & test
def splitDataset(dataSet,splitRatio):
	trainSize = int(len(dataSet)*splitRatio)
	trainSet=[]
	testSet=list(dataSet)
	while (len(trainSet)<trainSize):
		index = random.randrange(len(testSet))
		trainSet.append(testSet.pop(index))
	return trainSet,testSet

#seperate training data to dictionary for fishy websites(1) & for non-fishy websites(0)
def seperateByClass(dataSet):
	seperated={}
	for i in range(len(dataSet)):
		vector = dataSet[i]
		if (vector[-1] not in seperated):
			seperated[vector[-1]]=[]
		seperated[vector[-1]].append(vector)
	return seperated

#summarizing each attribute in the dataset
def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
        summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
        del summaries[-1]
        return summaries

def summarizeByClass(dataset):
        seperated = seperateByClass(dataset)
        summaries = {}
        for classval, instances in seperated.items():
                summaries[classval] = summarize(instances)
        return summaries

#making predictions
def calculateProbability(x, mean, stdev):
        if stdev==0: return 0
        else:
                exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
                return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
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
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

#main function

filename = 'fishy_websites.csv' #Dataset name
dataset = loadCsv(filename)
trainSet, testSet = splitDataset(dataset, 0.75) #functionality same as train_test_split() from sklearn
seperated = seperateByClass(trainSet)
summary = summarizeByClass(trainSet)
predictions = getPredictions(summary, testSet)
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: '+ str(round(accuracy,2))+"%" )

