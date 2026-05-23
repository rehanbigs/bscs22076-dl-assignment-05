import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

torch.manual_seed(2026)
np.random.seed(2026)

def loadIndices(filePath):
    with open(filePath, 'r') as fileObj:
        return [int(line.strip()) for line in fileObj.readlines() if line.strip()]

def setupModel():
    net = torchvision.models.resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Linear(512, 10)
    return net

def runBaseline():
    hardware = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainTransforms = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    evalTransforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    baseTrain = torchvision.datasets.CIFAR10('./data', train=True, transform=trainTransforms)
    baseVal = torchvision.datasets.CIFAR10('./data', train=True, transform=evalTransforms)
    baseTest = torchvision.datasets.CIFAR10('./data', train=False, transform=evalTransforms)

    idxTrain = loadIndices('splits/train_labeled_10percent.txt')
    idxVal = loadIndices('splits/val.txt')
    idxTest = loadIndices('splits/test.txt')

    trainLoader = DataLoader(Subset(baseTrain, idxTrain), batch_size=64, shuffle=True, num_workers=2)
    valLoader = DataLoader(Subset(baseVal, idxVal), batch_size=64, shuffle=False, num_workers=2)
    testLoader = DataLoader(Subset(baseTest, idxTest), batch_size=64, shuffle=False, num_workers=2)

    classifier = setupModel().to(hardware)
    lossFunc = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=3e-4)
    totalEpochs = 20

    lossTrainHist = []
    lossValHist = []

    for ep in range(totalEpochs):
        classifier.train()
        runningLoss = 0.0
        for images, labels in trainLoader:
            images = images.to(hardware)
            labels = labels.to(hardware)
            
            optimizer.zero_grad()
            preds = classifier(images)
            lossVal = lossFunc(preds, labels)
            lossVal.backward()
            optimizer.step()
            
            runningLoss += lossVal.item() * images.size(0)
            
        lossTrainHist.append(runningLoss / len(idxTrain))

        classifier.eval()
        valLossSum = 0.0
        correctCnt = 0
        with torch.no_grad():
            for images, labels in valLoader:
                images = images.to(hardware)
                labels = labels.to(hardware)
                
                preds = classifier(images)
                valLossSum += lossFunc(preds, labels).item() * images.size(0)
                bestGuesses = preds.max(1)[1]
                correctCnt += bestGuesses.eq(labels).sum().item()
        
        lossValHist.append(valLossSum / len(idxVal))
        acc = 100. * correctCnt / len(idxVal)
        print(f"Epoch {ep+1} complete. Validation Accuracy: {acc:.2f}%")

    os.makedirs('graphs', exist_ok=True)
    plt.figure()
    plt.plot(lossTrainHist, label='Train Loss')
    plt.plot(lossValHist, label='Validation Loss')
    plt.legend()
    plt.savefig('graphs/supervised_loss.png')
    
    classifier.eval()
    allGuesses = []
    allTrue = []
    correctTest = 0
    with torch.no_grad():
        for images, labels in testLoader:
            preds = classifier(images.to(hardware))
            bestGuesses = preds.max(1)[1]
            correctTest += bestGuesses.eq(labels.to(hardware)).sum().item()
            
            allGuesses.extend(bestGuesses.cpu().numpy())
            allTrue.extend(labels.numpy())
            
    finalAcc = 100. * correctTest / len(idxTest)
    print(f"\nFinal Supervised Test Accuracy: {finalAcc:.2f}%")
            
    matrix = confusion_matrix(allTrue, allGuesses)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=baseTest.classes)
    display.plot(cmap='Blues')
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/supervised_confusion_matrix.png')

if __name__ == '__main__':
    runBaseline()