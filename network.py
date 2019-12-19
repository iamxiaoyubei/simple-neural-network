import numpy as np
import math
import random
def intializeSameWeigths():
    hiddenWeights = [0.5, 0.5, 0.5, 0.5]
    outputWeights = [-0.5, -0.5]
    return hiddenWeights, outputWeights

def intializeRandomWeigths():
    hiddenWeights = [0, 0, 0, 0]
    outputWeights = [0, 0]
    for i in range(len(hiddenWeights)):
        hiddenWeights[i] = random.uniform(-1.0, 1.0)
    for i in range(len(outputWeights)):
        outputWeights[i] = random.uniform(-1.0, 1.0)
    return hiddenWeights, outputWeights

class Network:
    def __init__(self, hiddenWeights, outputWeigths):
        self.hiddenWeights = np.array(hiddenWeights)
        self.outputWeights = np.array(outputWeigths)
        return 
    
    def softmax(self, out):
        return np.exp(out)/sum(np.exp(out))

    # calculate MSE loss
    def calculateLoss(self, predict, label):
        return ((predict - label)**2).mean()

    def setInput(self, data):
        self.input = np.insert(data, 0, 1)
        return

    # calculate activation function (test ok)
    def sigmoid(self, out):
        a = 1.716
        b = 2.0/3.0
        return np.array((2*a/(1+np.exp(-b*out)))-a)

    def forward(self, data):
        # set input
        self.setInput(data)
        # calculate output from input
        # print("input:",self.input)
        # print("self.hiddenWeights:",self.hiddenWeights)
        self.hiddenOutputBeforeActivationFunc = np.dot(self.hiddenWeights, self.input)
        # print("self.hiddenOutputBeforeActivationFunc:",self.hiddenOutputBeforeActivationFunc)
        self.hiddenOutput = self.sigmoid(self.hiddenOutputBeforeActivationFunc)
        # print("self.hiddenOutput",self.hiddenOutput)
        self.hiddenOutput = np.insert(self.hiddenOutput, 0, 1)
        # print("self.hiddenOutput",self.hiddenOutput)
        self.outputBeforeActivationFunc = np.dot(self.outputWeights, self.hiddenOutput)
        self.output = self.sigmoid(self.outputBeforeActivationFunc)
        return self.output
    
    def derivativeByActivation(self, out):
        a = 1.716
        b = 2.0/3.0
        k = 1.0/(1+np.exp(-b*out))
        return 2*a*b*k*(1-k)

    # calculate gradient of each weight and do gradient descend
    def backpropagation(self, loss):
        learningRate = 0.1
        # partial loss/output
        partialLossDividedOutput = 2*(self.output - self.label)
        # partial output/outputWeights
        partialOutputDividedOutputWeights = self.derivativeByActivation(self.outputBeforeActivationFunc)*self.hiddenOutput
        # partial output/hiddenOutput
        partialOutputDividedHiddenOutput = self.derivativeByActivation(self.outputBeforeActivationFunc)*self.outputWeights[1]
        # partial hiddenOutput/hiddenWeights
        partialHiddenOutputDividedHiddenWeights = self.derivativeByActivation(self.hiddenOutputBeforeActivationFunc)*self.input
        # partial loss/outputWeights
        partialLossDividedOutputWeights = partialLossDividedOutput * partialOutputDividedOutputWeights
        # partial loss/hiddenWeights
        # print("output:", self.output)
        # print("label:",self.label)
        # print("partialLossDividedOutput ", partialLossDividedOutput)
        # print("partialOutputDividedHiddenOutput ",partialOutputDividedHiddenOutput)
        # print("partialHiddenOutputDividedHiddenWeights ", partialHiddenOutputDividedHiddenWeights)
        partialLossDividedHiddenWeights = partialLossDividedOutput * partialOutputDividedHiddenOutput * partialHiddenOutputDividedHiddenWeights
        # gradient descend on outputWeights
        self.outputWeights = self.outputWeights - learningRate * partialLossDividedOutputWeights
        # gradient descend on hiddenWeights
        self.hiddenWeights = self.hiddenWeights - learningRate * partialLossDividedHiddenWeights
        return
        
    # def trainWithSoftmax(self, data, label):
    #     output = self.forward(data)
    #     predict = self.softmax(output)
    #     loss = self.calculateLoss(predict, label)
    #     self.backpropagation(loss)
    #     return loss

    def train(self, data, label):
        self.label = np.array(label)
        output = self.forward(data)
        # print("output:", output)
        loss = self.calculateLoss(output, label)
        self.backpropagation(loss)
        return loss

    def inference(self, data):
        # inference
        output = self.forward(data)
        return np.sign(output)