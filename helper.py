import numpy as np
import torch
import multiprocessing
import math
import random
import csv
import tqdm

from PIL import Image
from torchvision import transforms

USE_CUDA = True

scale = 0.05


if torch.cuda.is_available() and USE_CUDA == True:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class geneDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        self.data = np.loadtxt(f"datasets/{file}.csv", delimiter=',', dtype=np.float32, skiprows=0)
        self.genes = torch.Tensor(self.data[:, 1:]).view((-1, 1, 32, 32))
        self.labels = self.data[:, :1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return((self.genes[idx], self.labels[idx]))

    
class encoderDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        self.data = np.loadtxt(f"datasets/{file}.csv", delimiter=',', dtype=np.float32, skiprows=0)
        self.genes = torch.Tensor(self.data[:, 1:]).view((-1, 1, 256))
        self.labels = self.data[:, :1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return((self.genes[idx], self.labels[idx]))


def processAll(repeats, rValues, DATA_LEN, BLOCK_SIZE, NOISE, train, test):
    PROCESSES = min(16, repeats)
    
    params = []
    
    for n in range(repeats):
        params.append((n, rValues, DATA_LEN, BLOCK_SIZE, NOISE, train, test))

    with multiprocessing.Pool(PROCESSES) as pool:
        results = [pool.apply_async(writeData, p) for p in params]
        
        for r in results:
            r.get()
            
    pool.close()

        
def writeData(n, rValues, DATA_LEN, BLOCK_SIZE, NOISE, train, test):
    rValues1 = rValues.copy()
    random.shuffle(rValues1)
    
    for j in range(int(DATA_LEN / BLOCK_SIZE)):
        fullTrain = []
        fullTest = []
        fullNoise = []

        for i in range(len(train)):
            rowData = []

            for h in range(BLOCK_SIZE):
                rowData.append(train[i][rValues1[h + j * BLOCK_SIZE][0] + 1])

            rowData.insert(0, float(train[i][0]))
            fullTrain.append(rowData)

        if NOISE > 1:
            for x in range(NOISE - 1):
                for i in range(len(train)):
                    rowData = []

                    for h in range(BLOCK_SIZE):
                        a = np.random.normal(loc=0, scale=scale)
                        rowData.append(train[i][rValues1[h + j * BLOCK_SIZE][0] + 1] + a)

                    rowData.insert(0, float(train[i][0]))
                    fullNoise.append(rowData)

        for i in range(len(test)):
            rowData = []

            for h in range(BLOCK_SIZE):
                rowData.append(test[i][rValues1[h + j * BLOCK_SIZE][0] + 1])

            rowData.insert(0, flip(float(test[i][0])))
            fullTest.append(rowData)

        iteration = n * int(DATA_LEN / BLOCK_SIZE) + j

        with open(f'datasets/train_full/{iteration}.csv', 'w') as f:
            writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(fullTrain)

        with open(f'datasets/test_full/{iteration}.csv', 'w') as f:
            writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(fullTest)

        if NOISE > 1:
            with open(f'datasets/noise_full/{iteration}.csv', 'w') as f:
                writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerows(fullNoise)


def flip(n):
    if n == 0.0:
        return 1.0
    
    else:
        return 0.0

                
def flipCol(data, col):
    for i in range(len(data)):
        data[i][col] = 1 - data[i][col]
            
    return 0


def getAccuracy(net, testset):
    wrong = 0
    right = 0
    total = 0

    with torch.no_grad():
        for data in testset:
            X, y = data
            
            net.eval()

            X = X.to(device)
            y = y.to(device)

            output = torch.clamp(net(X), min=0.0, max=1.0)

            for i in range(len(output)):
                if y[i][0] == round(float(output[i][0])):
                    right += 1
                else:
                    wrong += 1

                total += 1

    return (right / total)


def getAvgLoss(loss):
    avgLoss = []
    avg = []

    for item in loss:
        avg.append(item)

        if len(avg) > 500:
            avg.pop(0)

        avgLoss.append(sum(avg) / len(avg))
        
    return avgLoss


def getCondition(cstr):
    if cstr == "AD":
        return(1)
    elif cstr == "CTL":
        return(0)
    else:
        return(None)

    
def getMean(data):
    mean = []
    
    for j in range(len(data[0]) - 1):
        total = 0.0
        
        for i in range(len(data)):
            total += data[i][j + 1]
            
        mean.append(total / len(data))
        
    return mean
    
    
def getMinMax(data):
    maxValue = [-1000000.0] * (len(data[0]) - 1)
    minValue = [1000000.0] * (len(data[0]) - 1)

    for j in range(len(data[0]) - 1):
        for i in range(len(data)):
            if float(data[i][j + 1]) > maxValue[j]:
                maxValue[j] = float(data[i][j + 1])

            if float(data[i][j + 1]) < minValue[j]:
                minValue[j] = float(data[i][j + 1])
                
    return((maxValue, minValue))


def getRowMean(data):
    mean = []
    
    for i in range(len(data)):
        total = 0.0
        
        for j in range(1, len(data[i])):
            total += data[i][j]
            
        mean.append(total / (len(data[i]) - 1))
        
    return mean


def getRValue(data, col, ctl, ad):
    adTotal = 0.0
    ctlTotal = 0.0
    
    for i in range(len(data)):
        if data[i][0] == "1":
            adTotal += float(data[i][col])
            
        else:
            ctlTotal += float(data[i][col])
            
    ctlAvg = ctlTotal / ctl
    adAvg = adTotal / ad
    
    return(adAvg - ctlAvg)   


def getRValues(data, test, ctl, ad):
    rValues = []

    for j in range(len(data[0]) - 1):
        rValue = getRValue(data, j + 1, ctl, ad)
        flipCol(data, j + 1)

        if rValue < 0:
            rValue = abs(rValue)

        rValues.append((j, rValue))

    rValues.sort(key=lambda x: -x[1])

    return rValues


def getRValuesNoTest(data, ctl, ad):
    rValues = []

    for j in range(len(data[0]) - 1):
        rValue = getRValue(data, j + 1, ctl, ad)

        if rValue < 0:
            rValue = abs(rValue)

        rValues.append((j, rValue))

    rValues.sort(key=lambda x: -x[1])

    return rValues


def getRowVariance(data):
    mean = []
    
    for i in range(len(data)):
        total = 0.0
        
        for j in range(1, len(data[i])):
            total += abs(data[i][j])
            
        mean.append(total / (len(data[i]) - 1))
        
    return mean


def getTrainAccuracy(net, trainset):
    wrong = 0
    right = 0
    total = 0

    with torch.no_grad():
        net.eval()
        data = iter(trainset)
        
        X, y = data.next()

        X = X.to(device)
        y = y.to(device)

        output = torch.clamp(net(X), min=0.0, max=1.0)

        for i in range(len(output)):
            if y[i][0] == round(float(output[i][0])):
                right += 1
            else:
                wrong += 1

            total += 1

    return (right / total)


def getVariance(data):
    variance = []
    
    for j in range(len(data[0]) - 1):
        total = 0.0
        
        for i in range(len(data)):
            total += abs(data[i][j + 1])
            
        variance.append(total / len(data))
        
    return variance


def toFloat(data):
    for i in range(len(data)):
        for j in range(1, len(data[0])):
            data[i][j] = float(data[i][j])
            
    return data


def merge(dict1, dict2):
    res = {**dict1, **dict2}

    return res


def minmaxNormalize(data, maxValue=None, minValue=None):
    
    if minValue == None or maxValue == None:
        maxValue, minValue = getMinMax(data)
    
    for j in range(len(data[0]) - 1):
        for i in range(len(data)):
            data[i][j + 1] = abs((float(data[i][j + 1]) - minValue[j]) / (maxValue[j] - minValue[j]))
            
    return data


def normalize(data):
    mean = getMean(data)
    
    for j in range(len(data[0]) - 1):
        for i in range(len(data)):
            data[i][j + 1] -= mean[j]
            
    variance = getVariance(data)
    
    for j in range(len(data[0]) - 1):
        for i in range(len(data)):
            data[i][j + 1] /= variance[j]
            data[i][j + 1] /= 5
            data[i][j + 1] += 0.5

    return data


def normalizeRow(data):
    mean = getMean(data)
    
    for i in range(len(data)):
        for j in range(len(data[0]) - 1):
            data[i][j + 1] -= mean[j]
            
    variance = getVariance(data)
    
    for i in range(len(data)):
        for j in range(len(data[0]) - 1):
            data[i][j + 1] /= variance[j]
            data[i][j + 1] /= 5
            data[i][j + 1] += 0.5

    return data


def stretch(x):
    if x <= 0.5:
        x = 4 * (x ** 3)
        
    else:
        x = 1 - 4 * ((1-x) ** 3)
        
    return x


def wrongAns(output, expected, threshold):
    falseNeg = 0
    trueNeg = 0
    falsePos = 0
    truePos = 0
    total = 0
    
    for i in range(len(output)):
        if float(expected[i][0]) == 1.0:
            if float(output[i][0]) >= threshold:
                truePos += 1
            
            else:
                falseNeg += 1

        else:
            if float(output[i][0]) < threshold:
                trueNeg += 1
                
            else:
                falsePos += 1

        total += 1
    
    return(truePos, falsePos, trueNeg, falseNeg)