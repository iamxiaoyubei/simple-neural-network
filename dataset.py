import random
def createDatasetByRandom():
    dataset = []
    total = 10000
    for i in range(total):
        x1 = random.uniform(-5.0, 5.0)
        x2 = random.uniform(-5.0, 5.0)
        point = [x1, x2]
        dataset.append(point)
    return dataset

def createDatasetEvenly():
    dataset = []
    total = 100
    interval = 10.0 / total
    for i in range(total):
        x1 = -5.0 + i * interval
        for j in range(total):
            x2 = -5.0 + j * interval
            point = [x1, x2]
            dataset.append(point)
    return dataset

def getDataset():
    # label 1.716-> w1  -1.716-> w2
    dataset = []
    labels = []
    dataset = [[0.28, 1.31, -6.2],
        [0.07, 0.58, -0.78],
        [1.54, 2.01, -1.63],
        [-0.44, 1.18, -4.32],
        [-0.81, 0.21, 5.73],
        [1.52, 3.16, 2.77],
        [2.20, 2.42, -0.19],
        [0.91, 1.94, 6.21],
        [0.65, 1.93, 4.38],
        [-0.26, 0.82, -0.96],
        [0.011, 1.03, -0.21],
        [1.27, 1.28, 0.08],
        [0.13, 3.12, 0.16],
        [-0.21, 1.23, -0.11],
        [-2.18, 1.39, -0.19],
        [0.34, 1.96, -0.16],
        [-1.38, 0.94, 0.45],
        [-0.12, 0.82, 0.17],
        [-1.44, 2.31, 0.14],
        [0.26, 1.94, 0.08]]
    labels = [1.716,1.716,1.716,1.716,1.716,1.716,1.716,1.716,1.716,1.716,\
        -1.716,-1.716,-1.716,-1.716,-1.716,-1.716,-1.716,-1.716,-1.716,-1.716]
    return dataset, labels