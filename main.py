from network import Network, intializeRandomWeigths, intializeSameWeigths
from dataset import createDatasetEvenly, getDataset
import matplotlib.pyplot as plt 
import random
import math
import numpy as np
# 1(a)
# variables
hiddenWeights = [[0.5, 0.3, -0.1], [-0.5, -0.4, 1.0]]
outputWeights = [1.0, -2.0, 0.5]
# create network and datasets
net1a = Network(hiddenWeights, outputWeights)
dataset = createDatasetEvenly()
# color the input space according to network result
plt.figure(1)
plt.title("1a")
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim(xmin=-5, xmax=5)
plt.ylim(ymin=-5, ymax=5)
for data in dataset:
    result = net1a.inference(data)
    if result == -1:
        color = 'b'
    else:
        color = 'r'
    plt.plot(data[0], data[1], marker='o', color=color, markersize=10)
plt.show()

# 1(b)
# create network for 1b
hiddenWeights = [[-1.0, -0.5, 1.5], [1.0, 1.5, -0.5]]
outputWeights = [0.5, -1.0, 1.0]
net1b = Network(hiddenWeights, outputWeights)
# color the input space according to network result
plt.figure(2)
plt.title("1b")
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim(xmin=-5, xmax=5)
plt.ylim(ymin=-5, ymax=5)
for data in dataset:
    result = net1b.inference(data)
    if result == -1:
        color = 'b'
    else:
        color = 'r'
    plt.plot(data[0], data[1], marker='o', color=color, markersize=10)
plt.show()

# Problem 2
# theta for problem 2
theta = 0.005
# 2(a)
# initialize all weights randomly
hiddenWeights, outputWeights = intializeRandomWeigths()
# create network for net2a
net2a = Network(hiddenWeights, outputWeights)
# get dataset according to the table
dataset, labels, indexes = getDataset()
# random seed for problem 2 to shuffle datasets
random.seed(2019)
# train the network and plot a learning curve
plt.figure(3)
trainingErrors = []
accuracyList = []
epochs = []
epoch = 0
while 1:
    trainingError = 0.0
    epochGradient = 0.0
    correctNum = 0.0
    # shuffle dataset
    random.shuffle(indexes)
    # stochastic gradient descend training
    for index in indexes:
        data = dataset[index]
        label = labels[index]
        predict = net2a.inference(data)
        labelSign = math.copysign(1, label)
        if (predict == labelSign):
            correctNum += 1
        stepTrainingError, stepGradient = net2a.train(data, label)   
        trainingError += stepTrainingError
        epochGradient += stepGradient
    trainingError = trainingError / len(dataset)
    trainingErrors.append(trainingError)
    epochs.append(epoch)
    accuracy = correctNum / len(dataset)
    accuracyList.append(accuracy)
    epochGradient = epochGradient / len(dataset)
    if np.absolute(epochGradient).mean() < theta:
        print("Break Epoch in initialize all weights randomly:", epoch)
        break
    epoch += 1
plt.subplot(211)
plt.plot(epochs, trainingErrors, color='r', linestyle='solid', marker='o', linewidth=2)
plt.subplot(212)
plt.plot(epochs, accuracyList, color='r', linestyle='solid', marker='o', linewidth=2)


# 2(b)
# weights initialized to be the same throughout each level
hiddenWeights, outputWeights = intializeSameWeigths()
# create network for net2b
net2b = Network(hiddenWeights, outputWeights)
# get dataset according to the table
dataset, labels, indexes = getDataset()
# random seed for problem 2 to shuffle datasets
random.seed(2019)
# train the network and plot a learning curve
trainingErrors = []
accuracyList = []
epochs = []
epoch = 0
while 1:
    trainingError = 0.0
    epochGradient = 0.0
    correctNum = 0.0
    # shuffle dataset
    random.shuffle(indexes)
    # stochastic gradient descend training
    for index in indexes:
        data = dataset[index]
        label = labels[index]
        predict = net2b.inference(data)
        labelSign = math.copysign(1, label)
        if (predict == labelSign):
            correctNum += 1
        stepTrainingError, stepGradient = net2b.train(data, label)   
        trainingError += stepTrainingError
        epochGradient += stepGradient
    trainingError = trainingError / len(dataset)
    trainingErrors.append(trainingError)
    epochs.append(epoch)
    accuracy = correctNum / len(dataset)
    accuracyList.append(accuracy)
    epochGradient = epochGradient / len(dataset)
    if np.absolute(epochGradient).mean() < theta:
        print("Break Epoch in weights initialized to be the same throughout each level:", epoch)
        break
    epoch += 1
plt.subplot(211)
plt.xlabel("epoch")
plt.ylabel("training_error")
plt.plot(epochs, trainingErrors, color='b', linestyle='solid', marker='o', linewidth=2)
plt.subplot(212)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(epochs, accuracyList, color='b', linestyle='solid', marker='o', linewidth=2)
plt.show()

