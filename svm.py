# svm.py
# -------------

# svm implementation
import util
from sklearn import svm
import numpy as np

PRINT = True
clf = svm.LinearSVC(multi_class="ovr", max_iter=3)

class SVMClassifier:
    """
    svm classifier
    """

    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "svm"

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        curdata = []
        for i in range(len(trainingData)):
            curdata.append([])
            for x in range(28):
                for y in range(28):
                    curdata[i].append(trainingData[i][(x, y)])
        clf.fit(curdata, trainingLabels)

    def classify(self, data):
        guesses = []
        usedata = []
        for datum in range(len(data)):
                usedata.append([])
                for x in range(28):
                    for y in range(28):
                        usedata[datum].append(data[datum][(x, y)])
        guesses = clf.predict(usedata)
        return guesses

