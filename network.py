import numpy as np
import math
import random
random.seed(1500)
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
    print("Random Hidden Weights:", hiddenWeights)
    print("Random Output Weights:", outputWeights)
    return hiddenWeights, outputWeights

class Network:
    def __init__(self, hiddenWeights, outputWeights):
        self.hiddenWeights = np.array(hiddenWeights)
        self.outputWeights = np.array(outputWeights)
        return 
    
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
        self.hiddenOutputBeforeActivationFunc = np.dot(self.hiddenWeights, self.input)
        self.hiddenOutput = self.sigmoid(self.hiddenOutputBeforeActivationFunc)
        self.hiddenOutput = np.insert(self.hiddenOutput, 0, 1)
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
        partialLossDividedHiddenWeights = partialLossDividedOutput * partialOutputDividedHiddenOutput * partialHiddenOutputDividedHiddenWeights
        # gradient descend on outputWeights
        self.outputWeights = self.outputWeights - learningRate * partialLossDividedOutputWeights
        # gradient descend on hiddenWeights
        self.hiddenWeights = self.hiddenWeights - learningRate * partialLossDividedHiddenWeights
        return np.append(partialLossDividedOutputWeights, partialLossDividedHiddenWeights)

    def train(self, data, label):
        self.label = np.array(label)
        output = self.forward(data)
        loss = self.calculateLoss(output, label)
        gradient = self.backpropagation(loss)
        return loss, gradient

    # inference
    def inference(self, data):
        output = self.forward(data)
        return np.sign(output)