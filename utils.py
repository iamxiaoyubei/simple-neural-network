import matplotlib.pyplot as plt
import numpy as np
def activation(out):
    a = 1.716
    b = 2.0/3.0
    return (2*a/(1+np.exp(-b*out)))-a

def createDataset():
    dataset = []
    total = 1000
    interval = 20.0 / total
    for i in range(total):
        point = -10.0 + interval * i
        dataset.append(point)
    return dataset

dataset = createDataset()
plt.figure()
plt.title("activation function")
plt.xlabel("net")
plt.ylabel("f(net)")
plt.xlim(xmin=-10, xmax=10)
plt.ylim(ymin=-2, ymax=2)
for data in dataset:
    y = activation(data)
    plt.plot(data, y, marker='o', color='r')
plt.show()
