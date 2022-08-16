#read file to get data

#def gatherData(filename) --> list()

#def createValidationSet --> list()
		#remove from training data and create a validation set

#def create y list() --> list()
		# seperate target y data from x data
import numpy as np
import math
import random
def gatherData(filename):
    with open(filename+'.data') as f:
        rows = f.readlines()
    data = []
    dataY = []
    for row in rows:
        splitRow = row.split(",")
        dataX = []
        for i in range(len(splitRow)):
            if i != len(splitRow)-1:
                if splitRow[i] != "?":
                    dataX.append(float(splitRow[i]))
                else:
                    dataX.append(splitRow[i])
            else:
                YVal = float(splitRow[i][0])
                if YVal < 1:
                    dataY.append(0)
                else:
                    dataY.append(1)
                data.append(dataX)
    return (data,dataY)

def getMeans(data):
    means = []
    for i in range(len(data[0])):
        s = 0
        for d in data:
            if d[i] != "?":
                s+= float(d[i])
        means.append(round(s/len(data),3))
    return means


def fillData(data):
    means = getMeans(data)
    for d in data:
        for i in range(len(d)):
            if d[i] == "?":
                d[i] = means[i]


class LinRegPredictor(object):
    def __init__(self,eta,dataX,dataY,valX,valY,maxEpochs):
        #self.weights = [0 for i in range(len(dataX[0]))]
        self.weights = np.zeros(len(dataX[0]))
        #self.weights = np.array([random.uniform(-100,100) for i in range(len(dataX[0]))])

        self.features = []
        self.eta = eta
        self.dataX = dataX
        self.dataY = np.array(dataY)
        self.valX = valX
        self.valY = np.array(valY)
        self.maxEpochs = maxEpochs


    def trainHingeLoss(self):
        
        epochs = 0
        while epochs < self.maxEpochs:
            for i in range(len(self.dataX)):
                margin = (np.dot(self.weights,np.array(self.dataX[i]))*self.dataY[i])
                if margin > 1:
                    gradient = 0
                else: 
                    gradient = -1*np.array(self.dataX[i])*self.dataY[i]
                self.weights -= self.eta*gradient
                #print("WEIGHTS: ",self.weights)
            epochs +=1
            self.eta/=1.3
    def trainLogLoss(self):
        epochs = 0
        #use correlations as initial weights
        #self.weights = self.getCorrelations()
        while epochs < self.maxEpochs:
            for i in range(len(self.dataX)):
                margin = np.dot(self.weights,np.array(self.dataX[i]))
                if margin < 0:
                    sigmoid = 1 - 1 / (1 + math.exp(margin))
                else:
                    sigmoid = 1/(1+math.exp(-1*margin))
                gradient = np.array(self.dataX[i]).T * (sigmoid - self.dataY[i])
                #gradient = (margin -self.dataY[i])/(self.weights*(1-margin))#-1*np.array(self.dataX[i])*self.dataY[i]
                self.weights -= self.eta*gradient
                #print("WEIGHTS: ",self.weights)
            epochs +=1
            #self.eta/=1.5
            if epochs % 500 == 0 and self.eta >= 0.001:
                self.eta/=10

    def predict(self):
        predictions = []
        #print("FINAL WEIGHTS: ",self.weights)
        for dataPoint in self.valX:
            score = np.dot(self.weights,dataPoint)
            #print(score)
            if score >0:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions
    def predictUsingCorrelations(self):
        predictions = []
        #print("FINAL WEIGHTS: ",self.weights)
        for dataPoint in self.valX:
            score = np.dot(self.getCorrelations()*5000,dataPoint)
            #print(score)
            if score >0:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions  

    def getCorrelations(self):
        columns = []
        correlations = []
        for i in range(len(self.dataX[0])):
            column = []
            for d in self.dataX:
                column.append(d[i])
            columns.append(column)

        for column in columns:
            correlation = np.corrcoef(column,self.dataY)
            #print("CORRELATION: ",correlation[0][1])
            correlations.append(correlation[0][1])
        return np.array(correlations)


data = gatherData("processed.cleveland")
fillData(data[0])
stopData = int(len(data[0])*0.8)
trainSet = [data[0][i] for i in range(stopData)]
valSet = [data[0][i] for i in range(stopData,len(data[0]))]
trainY = [data[1][i] for i in range(stopData)]
valY = [data[1][i] for i in range(stopData,len(data[0]))]

linRegLearner = LinRegPredictor(1,trainSet,trainY,valSet,valY,1000)
linRegLearner.getCorrelations()
linRegLearner.trainLogLoss()
predictions = linRegLearner.predict()
#predictions = linRegLearner.predictUsingCorrelations()


print("ACTUAL: ",valY)
print("PREDIC: ",predictions)
correct = 0
for p in range(len(predictions)):
    if predictions[p] == valY[p]:
        correct+=1
print("TOTAL CORRECT: ",correct,"/",len(predictions))
print(valSet[2])
print(linRegLearner.weights)

# i = 0
# for v in valSet:
#     print("Example: ",v, "-- Prediction: ",predictions[i])
#     i+=1






	

        