from network import Network, intializeRandomWeigths, intializeSameWeigths
from dataset import createDatasetEvenly, getDataset
import matplotlib.pyplot as plt 
import random
# # 1(a)
# # variables
# hiddenWeights = [[0.5, 0.3, -0.1], [-0.5, -0.4, 1.0]]
# outputWeights = [1.0, -2.0, 0.5]
# # create network and datasets
# net1a = Network(hiddenWeights, outputWeights)
# dataset = createDatasetEvenly()
# # color the input space according to network result
# plt.figure()
# plt.xlim(xmin=-5, xmax=5)
# plt.ylim(ymin=-5, ymax=5)
# for data in dataset:
#     result = net1a.inference(data)
#     if result == -1:
#         color = 'b'
#     else:
#         color = 'r'
#     plt.plot(data[0], data[1], marker='o', color=color, markersize=10)
# plt.show()

# # 1(b)
# # create network for 1b
# hiddenWeights = [[-1.0, -0.5, 1.5], [1.0, 1.5, -0.5]]
# outputWeights = [0.5, -1.0, 1.0]
# net1b = Network(hiddenWeights, outputWeights)
# # color the input space according to network result
# plt.figure()
# plt.xlim(xmin=-5, xmax=5)
# plt.ylim(ymin=-5, ymax=5)
# for data in dataset:
#     result = net1b.inference(data)
#     if result == -1:
#         color = 'b'
#     else:
#         color = 'r'
#     plt.plot(data[0], data[1], marker='o', color=color, markersize=10)
# plt.show()


# 2(a)
epoch = 100
# initialize all weights randomly
hiddenWeights, outputWeights = intializeRandomWeigths()
# create network for net2a
net2a = Network(hiddenWeights, outputWeights)
# get dataset according to the table
dataset, labels = getDataset()
# train the network and plot a learning curve
# plt.subplot(1,2,1,title="initialize weights randomly")
plt.figure()
plt.xlabel("epoch")
plt.ylabel("training_error")
trainingErrors = []
epochs = []
for i in range(epoch):
    trainingError = 0
    # shuffle dataset
    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(dataset)
    random.seed(randnum)
    random.shuffle(labels)
    # stochastic gradient descend training
    for data, label in zip(dataset, labels):
        trainingError += net2a.train(data, label)
    trainingError = trainingError / len(data)
    trainingErrors.append(trainingError)
    epochs.append(i)
plt.plot(epochs, trainingErrors, color='r', linestyle='solid', marker='o', linewidth=2)


# 2(b)
# weights initialized to be the same throughout each level
hiddenWeights, outputWeights = intializeSameWeigths()
# create network for net2b
net2b = Network(hiddenWeights, outputWeights)
# train the network and plot a learning curve
# plt.subplot(1,2,2,title="initialize same weights each level")
plt.xlabel("epoch")
plt.ylabel("training_error")
trainingErrors = []
epochs = []
for i in range(epoch):
    trainingError = 0
    # shuffle dataset
    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(dataset)
    random.seed(randnum)
    random.shuffle(labels)
    # stochastic gradient descend training
    for data, label in zip(dataset, labels):
        trainingError += net2b.train(data, label)
    trainingError = trainingError / len(data)
    trainingErrors.append(trainingError)
    epochs.append(i)
plt.plot(epochs, trainingErrors, color='b', linestyle='solid', marker='o', linewidth=2)
plt.show()