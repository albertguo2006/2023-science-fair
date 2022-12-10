from PIL import Image
import numpy as np
import torch
from torchvision import transforms

transform = transforms.Normalize((0.5,), (0.5,))

class geneDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        self.data = np.loadtxt(f"datasets/{file}.csv", delimiter=',', dtype=np.float32, skiprows=0)
        self.genes = transform(torch.Tensor(self.data[:, 1:]).view((-1, 1, 32, 32)))
        self.labels = self.data[:, [0]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return((self.genes[idx], self.labels[idx]))

    
def flipCol(data, col):
    for i in range(len(data)):
        data[i][col] = 1 - data[i][col]
            
    return 0


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


def getRValue(data, col, conditions, ctl, ad):
    adTotal = 0.0
    ctlTotal = 0.0
    
    for i in range(len(data)):
        if int(data[i][0]) == 1:
            adTotal += float(data[i][col])
            
        else:
            ctlTotal += float(data[i][col])
            
    ctlAvg = ctlTotal / ctl
    adAvg = adTotal / ad
    
    return(adAvg - ctlAvg)   


def getRValues(data, conditions, ctl, ad, testData):
    rValues = []

    for j in range(len(data[0]) - 2):
        rValue = getRValue(data, j + 2, conditions, ctl, ad)

        if rValue < 0:
            flipCol(data, j + 2)
            flipCol(testData, j + 2)
            rValue = abs(rValue)

        rValues.append((j, rValue))

    rValues.sort(key=lambda x: -x[1])

    return rValues


def getRValuesNoTest(data, conditions, ctl, ad):
    rValues = []

    for j in range(len(data[0]) - 1):
        rValue = getRValue(data, j + 1, conditions, ctl, ad)

        if rValue < 0:
            flipCol(data, j + 1)
            rValue = abs(rValue)

        rValues.append((j, rValue))

    rValues.sort(key=lambda x: -x[1])

    return rValues


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
        if int(expected[i][0]) > 0.5:
            if float(output[i][0]) < threshold:
                falseNeg += 1
            
            else:
                truePos += 1
        else:
            if float(output[i][0]) >= threshold:
                falsePos += 1
                
            else:
                trueNeg += 1

        total += 1
    
    return(truePos, falsePos, trueNeg, falseNeg)